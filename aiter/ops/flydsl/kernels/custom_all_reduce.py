"""Minimal FlyDSL transport state for fused HGEMM + all-reduce."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.distributed as dist

_KMAXBLOCKS = 80
_DEFAULT_MAX_SIZE = 8192 * 1024 * 8 * 2  # 128 MB


def _is_weak_contiguous(tensor: torch.Tensor) -> bool:
    try:
        if tensor.is_contiguous():
            return True
        storage = tensor.untyped_storage()
        used_bytes = int(tensor.numel()) * int(tensor.element_size())
        available = int(storage.nbytes()) - int(tensor.storage_offset()) * int(
            tensor.element_size()
        )
        return available == used_bytes
    except Exception:
        return False


class FlyDSLAllreduce:
    _HIP_IPC_HANDLE_BYTES = 64
    _HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1
    _HIP_DEVICE_MALLOC_UNCACHED = 0x3
    _hip = None
    _hipIpcMemHandle_t = None
    _SIGNAL_SIZE = ((_KMAXBLOCKS * 8 * 4) * 2 + _KMAXBLOCKS * 4 + 127) & ~127
    _SUPPORTED_WORLD_SIZES = {2, 4, 8}
    _SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}

    @classmethod
    def _load_hip(cls):
        if cls._hip is not None:
            return cls._hip

        import ctypes

        for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
            try:
                cls._hip = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if cls._hip is None:
            raise RuntimeError("Failed to load HIP runtime library")

        class hipIpcMemHandle_t(ctypes.Structure):
            _fields_ = [("reserved", ctypes.c_byte * cls._HIP_IPC_HANDLE_BYTES)]

        cls._hipIpcMemHandle_t = hipIpcMemHandle_t
        cls._hip.hipIpcGetMemHandle.restype = ctypes.c_int
        cls._hip.hipIpcGetMemHandle.argtypes = [
            ctypes.POINTER(hipIpcMemHandle_t),
            ctypes.c_void_p,
        ]
        cls._hip.hipIpcOpenMemHandle.restype = ctypes.c_int
        cls._hip.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            hipIpcMemHandle_t,
            ctypes.c_uint,
        ]
        cls._hip.hipIpcCloseMemHandle.restype = ctypes.c_int
        cls._hip.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        cls._hip.hipGetErrorString.restype = ctypes.c_char_p
        cls._hip.hipGetErrorString.argtypes = [ctypes.c_int]
        cls._hip.hipExtMallocWithFlags.restype = ctypes.c_int
        cls._hip.hipExtMallocWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        cls._hip.hipFree.restype = ctypes.c_int
        cls._hip.hipFree.argtypes = [ctypes.c_void_p]
        cls._hip.hipMemset.restype = ctypes.c_int
        cls._hip.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        return cls._hip

    @classmethod
    def _hip_check(cls, err: int, *, what: str):
        if int(err) == 0:
            return
        hip = cls._load_hip()
        try:
            msg = hip.hipGetErrorString(int(err))
            msg = msg.decode("utf-8", errors="replace") if msg else f"hipError({err})"
        except Exception:
            msg = f"hipError({err})"
        raise RuntimeError(f"{what} failed: {msg}")

    @classmethod
    def _get_mem_handle_bytes(cls, base_ptr: int) -> bytes:
        import ctypes

        hip = cls._load_hip()
        handle = cls._hipIpcMemHandle_t()
        err = hip.hipIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(int(base_ptr)))
        cls._hip_check(err, what="hipIpcGetMemHandle")
        return bytes(ctypes.string_at(ctypes.byref(handle), cls._HIP_IPC_HANDLE_BYTES))

    @classmethod
    def _open_mem_handle(cls, handle_bytes: bytes) -> int:
        import ctypes

        if len(handle_bytes) != cls._HIP_IPC_HANDLE_BYTES:
            raise ValueError(f"Expected {cls._HIP_IPC_HANDLE_BYTES}B IPC handle")
        hip = cls._load_hip()
        handle = cls._hipIpcMemHandle_t()
        ctypes.memmove(
            ctypes.byref(handle),
            bytes(handle_bytes),
            cls._HIP_IPC_HANDLE_BYTES,
        )
        out_ptr = ctypes.c_void_p()
        err = hip.hipIpcOpenMemHandle(
            ctypes.byref(out_ptr),
            handle,
            ctypes.c_uint(int(cls._HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)),
        )
        cls._hip_check(err, what="hipIpcOpenMemHandle")
        return int(out_ptr.value)

    @classmethod
    def _close_mem_handle(cls, base_ptr: int) -> None:
        import ctypes

        hip = cls._load_hip()
        err = hip.hipIpcCloseMemHandle(ctypes.c_void_p(int(base_ptr)))
        cls._hip_check(err, what="hipIpcCloseMemHandle")

    @classmethod
    def _alloc_uncached(cls, size: int) -> int:
        import ctypes

        hip = cls._load_hip()
        buf = ctypes.c_void_p()
        err = hip.hipExtMallocWithFlags(
            ctypes.byref(buf),
            ctypes.c_size_t(size),
            ctypes.c_uint(cls._HIP_DEVICE_MALLOC_UNCACHED),
        )
        cls._hip_check(err, what="hipExtMallocWithFlags")
        err = hip.hipMemset(buf, 0, ctypes.c_size_t(size))
        cls._hip_check(err, what="hipMemset")
        return int(buf.value)

    @classmethod
    def _free_device_mem(cls, ptr: int) -> None:
        import ctypes

        hip = cls._load_hip()
        err = hip.hipFree(ctypes.c_void_p(int(ptr)))
        cls._hip_check(err, what="hipFree")

    @staticmethod
    def _gather_object_list_via_broadcast(group, shard_data):
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        all_data = [[None] for _ in range(world_size)]
        all_data[rank][0] = shard_data
        ranks = sorted(dist.get_process_group_ranks(group=group))
        for i, src_rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i],
                src=src_rank,
                group=group,
                device="cpu",
            )
        return [all_data[i][0] for i in range(world_size)]

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

        alloc_size = self._SIGNAL_SIZE + self.max_size
        self._meta_ptr = self._alloc_uncached(alloc_size)

        my_meta_bytes = self._get_mem_handle_bytes(self._meta_ptr)
        all_meta = self._gather_object_list_via_broadcast(
            self.group,
            (my_meta_bytes, 0),
        )

        self._meta_bases = [None] * self.world_size
        self._sg_ptrs = [0] * 8
        self._tmp_ptrs = [0] * 8
        for r in range(self.world_size):
            handle_bytes, off = all_meta[r]
            base_ptr = (
                self._meta_ptr
                if r == self.rank
                else int(self._open_mem_handle(bytes(handle_bytes)))
            )
            if r != self.rank:
                self._meta_bases[r] = base_ptr
            sg_ptr = base_ptr + off
            tmp_ptr = sg_ptr + self._SIGNAL_SIZE
            self._sg_ptrs[r] = sg_ptr
            self._tmp_ptrs[r] = tmp_ptr
        for i in range(self.world_size, 8):
            self._sg_ptrs[i] = self._sg_ptrs[0]
            self._tmp_ptrs[i] = self._tmp_ptrs[0]

        self._self_sg = self._sg_ptrs[self.rank]
        self._gpu_sg_ptrs_array = torch.tensor(
            self._sg_ptrs[:8],
            dtype=torch.int64,
            device=self.device,
        )

        ws, rk = self.world_size, self.rank
        rotated_tmp_ptrs = [self._tmp_ptrs[(rk + i) % ws] for i in range(8)]
        self._gpu_tmp_ptrs_array = torch.tensor(
            rotated_tmp_ptrs,
            dtype=torch.int64,
            device=self.device,
        )

        self.output_buffer = torch.empty(
            self.max_size,
            dtype=torch.uint8,
            device=self.device,
        )
        out_buf_base = int(self.output_buffer.untyped_storage().data_ptr())
        out_buf_off = int(self.output_buffer.data_ptr()) - out_buf_base
        my_out_handle = self._get_mem_handle_bytes(out_buf_base)
        all_out = self._gather_object_list_via_broadcast(
            self.group,
            (my_out_handle, out_buf_off),
        )

        self._output_buffer_bases = [None] * self.world_size
        self._output_buffer_ptrs = [0] * 8
        for r in range(self.world_size):
            handle_bytes, off = all_out[r]
            if r == self.rank:
                self._output_buffer_ptrs[r] = int(self.output_buffer.data_ptr())
            else:
                peer_base = int(self._open_mem_handle(bytes(handle_bytes)))
                self._output_buffer_bases[r] = peer_base
                self._output_buffer_ptrs[r] = peer_base + off
        for i in range(self.world_size, 8):
            self._output_buffer_ptrs[i] = self._output_buffer_ptrs[0]

        self._gpu_output_buffer_ptrs_array = torch.tensor(
            self._output_buffer_ptrs[:8],
            dtype=torch.int64,
            device=self.device,
        )
        self._IS_CAPTURING = False
        self._graph_out_bases: list[int] = []
        self._pending_graph_entries: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._graph_ptrs_cache: dict[int, torch.Tensor] = {}

    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            self._pending_graph_entries = []
            self._graph_ptrs_cache = {}
            yield
        finally:
            self._IS_CAPTURING = False
            if self._pending_graph_entries:
                with torch.inference_mode():
                    self._register_graph_outputs()

    @classmethod
    def _get_alloc_base_ptr(cls, dev_ptr: int) -> int:
        import ctypes

        hip = cls._load_hip()
        base = ctypes.c_void_p()
        range_start_addr = 11
        if not hasattr(hip, "_pga_setup"):
            hip.hipPointerGetAttribute.restype = ctypes.c_int
            hip.hipPointerGetAttribute.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            hip._pga_setup = True
        err = hip.hipPointerGetAttribute(
            ctypes.byref(base),
            ctypes.c_int(range_start_addr),
            ctypes.c_void_p(int(dev_ptr)),
        )
        cls._hip_check(err, what="hipPointerGetAttribute(RANGE_START_ADDR)")
        return int(base.value)

    def _get_or_create_graph_out_ptrs(self, out: torch.Tensor) -> torch.Tensor:
        ptr = int(out.data_ptr())
        cached = self._graph_ptrs_cache.get(ptr)
        if cached is not None:
            return cached

        # During graph capture the kernel immediately dereferences `out_ptrs`,
        # so seed the table with the always-valid eager output buffers. Once
        # capture finishes, `_register_graph_outputs()` mutates the same tensor
        # in-place to the graph output addresses used during replay.
        per_call_ptrs = self._gpu_output_buffer_ptrs_array.clone()
        self._pending_graph_entries.append((out, per_call_ptrs))
        self._graph_ptrs_cache[ptr] = per_call_ptrs
        return per_call_ptrs

    def _clear_graph_out_bases(self) -> None:
        for base in self._graph_out_bases:
            try:
                self._close_mem_handle(int(base))
            except Exception:
                pass
        self._graph_out_bases = []

    def _register_graph_outputs(self) -> None:
        ws, rk = self.world_size, self.rank
        entries = self._pending_graph_entries
        if not entries:
            return

        self._clear_graph_out_bases()

        my_handle_list = []
        for out, _ in entries:
            alloc_base = self._get_alloc_base_ptr(int(out.data_ptr()))
            off = int(out.data_ptr()) - alloc_base
            handle = self._get_mem_handle_bytes(alloc_base)
            my_handle_list.append((handle, off))

        all_ranks_handles = self._gather_object_list_via_broadcast(
            self.group,
            my_handle_list,
        )

        for entry_idx, (out, per_call_ptrs) in enumerate(entries):
            ptrs = [0] * 8
            for r in range(ws):
                handle_bytes, off = all_ranks_handles[r][entry_idx]
                if r == rk:
                    ptrs[r] = int(out.data_ptr())
                else:
                    peer_base = int(self._open_mem_handle(bytes(handle_bytes)))
                    self._graph_out_bases.append(peer_base)
                    ptrs[r] = peer_base + off
            for i in range(ws, 8):
                ptrs[i] = ptrs[0]
            per_call_ptrs.copy_(
                torch.tensor(ptrs[:8], dtype=torch.int64, device=self.device)
            )

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
        for bases in (self._meta_bases, self._output_buffer_bases):
            for base in bases:
                if base is not None:
                    self._close_mem_handle(int(base))
        self._clear_graph_out_bases()
        self._meta_bases = []
        self._output_buffer_bases = []
        if getattr(self, "_meta_ptr", None):
            try:
                self._free_device_mem(self._meta_ptr)
            except Exception:
                pass
            self._meta_ptr = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
