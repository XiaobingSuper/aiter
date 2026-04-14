"""Minimal FlyDSL transport state for fused HGEMM + QuickReduce."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.distributed as dist

from .custom_all_reduce import FlyDSLAllreduce, _DEFAULT_MAX_SIZE, _is_weak_contiguous

_QR_TILE_THREADS = 256
_QR_ATOMS_PER_THREAD = 8
_QR_ATOM_BYTES = 16
_QR_COMM_TILE_BYTES = _QR_TILE_THREADS * _QR_ATOMS_PER_THREAD * _QR_ATOM_BYTES
_QR_MIN_TILE_M = 16
_QR_TILE_N = 128
_QR_MIN_LOGICAL_TILE_BYTES = _QR_MIN_TILE_M * _QR_TILE_N * 2
_QR_RANK_TILE_STRIDE = 1152


def _align_up(value: int, alignment: int) -> int:
    return (int(value) + int(alignment) - 1) & ~(int(alignment) - 1)


class FlyDSLQuickReduce(FlyDSLAllreduce):
    """IPC transport for the FlyDSL single-kernel INT4 QuickReduce path."""

    _SUPPORTED_WORLD_SIZES = {2, 4, 8}
    _SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}

    def __init__(
        self,
        *,
        group,
        device: torch.device,
        max_size: int,
        world_size: int,
        rank: int,
        full_nvlink: bool,
    ):
        self.group = group
        self.device = device
        self.max_size = int(max_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.full_nvlink = bool(full_nvlink)

        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        if self.world_size <= 1:
            raise ValueError("world_size must be > 1")
        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            raise ValueError(
                f"QuickReduce transport only supports world_size in {sorted(self._SUPPORTED_WORLD_SIZES)}"
            )

        self.max_blocks = max(1, (self.max_size + _QR_MIN_LOGICAL_TILE_BYTES - 1) // _QR_MIN_LOGICAL_TILE_BYTES)
        self.flags_phase_bytes = self.max_blocks * self.world_size * 4
        self.data_offset = _align_up(2 * self.flags_phase_bytes, 128)
        self.rank_tile_bytes = _QR_RANK_TILE_STRIDE * (_QR_ATOMS_PER_THREAD // self.world_size)
        self.transmitted_tile_bytes = self.rank_tile_bytes * self.world_size
        self.phase_bytes = _align_up(self.max_blocks * self.transmitted_tile_bytes, 128)
        alloc_size = self.data_offset + 2 * self.phase_bytes

        self._buffer_ptr = self._alloc_uncached(alloc_size)
        my_handle_bytes = self._get_mem_handle_bytes(self._buffer_ptr)
        all_handles = self._gather_object_list_via_broadcast(self.group, my_handle_bytes)

        self._buffer_bases = [None] * self.world_size
        self._buffer_ptrs = [0] * 8
        for r in range(self.world_size):
            handle_bytes = all_handles[r]
            if r == self.rank:
                base_ptr = self._buffer_ptr
            else:
                base_ptr = int(self._open_mem_handle(bytes(handle_bytes)))
                self._buffer_bases[r] = base_ptr
            self._buffer_ptrs[r] = base_ptr
        for i in range(self.world_size, 8):
            self._buffer_ptrs[i] = self._buffer_ptrs[0]

        self._gpu_buffer_ptrs_array = torch.tensor(
            self._buffer_ptrs[:8],
            dtype=torch.int64,
            device=self.device,
        )

    @contextmanager
    def capture(self):
        yield

    def required_logical_blocks(self, m: int, n: int, tile_m: int, tile_n: int) -> int:
        bm = (int(m) + int(tile_m) - 1) // int(tile_m)
        bn = int(n) // int(tile_n)
        return bm * bn

    def should_fuse(self, out: torch.Tensor) -> bool:
        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            return False
        if out.dtype not in self._SUPPORTED_DTYPES:
            return False
        if not _is_weak_contiguous(out):
            return False
        out_bytes = int(out.numel()) * int(out.element_size())
        if out_bytes % 16 != 0:
            return False
        if out_bytes > self.max_size:
            return False
        if self.world_size > 2 and not self.full_nvlink:
            return False
        return True

    def close(self):
        for base in getattr(self, "_buffer_bases", []):
            if base is not None:
                self._close_mem_handle(int(base))
        self._buffer_bases = []
        if getattr(self, "_buffer_ptr", None):
            try:
                self._free_device_mem(self._buffer_ptr)
            except Exception:
                pass
            self._buffer_ptr = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
