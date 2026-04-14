"""FlyDSL fused HGEMM + single-kernel INT4 QuickReduce."""

from __future__ import annotations

import os
from typing import Optional

import torch
from flydsl.expr.typing import Int32, Int64

from aiter.dist.parallel_state import get_tp_group

from ..shuffle import shuffle_weight
from .gemm_kernels import (
    FIXED_STAGE,
    _get_flydsl_shuffle_layout,
    _get_split_k_global_semaphore,
    _get_split_k_signal_state,
    _normalize_launch_stream,
    _normalize_supported_kernel_metadata,
    _to_kernel_dtype,
    _validate_hgemm_inputs,
    _validate_hgemm_tiling,
)
from .kernels.quick_reduce import (
    FlyDSLQuickReduce,
    _DEFAULT_MAX_SIZE,
    _QR_COMM_TILE_BYTES,
    _QR_MIN_LOGICAL_TILE_BYTES,
)
from .kernels.splitk_hgemm_qr import compile_hgemm_qr_kernel

__all__ = ["flydsl_hgemm_qr", "supports_hgemm_qr", "experimental_hgemm_qr_enabled"]

_BACKEND_CACHE: dict[tuple[tuple[int, ...], int, int], "_HGEMMQRBackend"] = {}
_INT4_ONLY = {"INT4"}
_SUPPORTED_BLOCK_THREADS = 256


def _quick_reduce_quantization() -> str:
    return str(os.environ.get("AITER_QUICK_REDUCE_QUANTIZATION", "")).strip().upper()


def experimental_hgemm_qr_enabled() -> bool:
    return str(os.environ.get("AITER_ENABLE_EXPERIMENTAL_FLYDSL_HGEMM_QR", "")).strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def supports_hgemm_qr(
    *,
    tile_m: int,
    tile_n: int,
    tile_k: Optional[int] = None,
    split_k: int,
    block_m_warps: int,
    block_n_warps: int,
    pack_n: int,
) -> bool:
    if _quick_reduce_quantization() not in _INT4_ONLY:
        return False
    if int(pack_n) != 1:
        return False
    if int(split_k) != 1:
        return False
    if int(tile_m) <= 0 or int(tile_n) <= 0:
        return False
    block_threads = 64 * int(block_m_warps) * int(block_n_warps)
    if block_threads != _SUPPORTED_BLOCK_THREADS:
        return False
    tile_out_bytes = int(tile_m) * int(tile_n) * 2
    if tile_out_bytes > _QR_COMM_TILE_BYTES:
        return False
    if tile_out_bytes % (block_threads * 8) != 0:
        return False
    if tile_k is not None:
        if (int(tile_m) * int(tile_k)) % (block_threads * 8) != 0:
            return False
        if (int(tile_n) * int(tile_k)) % (block_threads * 8) != 0:
            return False
    return True


def _normalize_hgemm_qr_kwargs(hgemm_kwargs: Optional[dict]) -> dict:
    if not hgemm_kwargs:
        raise ValueError("FlyDSL HGEMM QR requires an explicit kernel config")
    kwargs = dict(hgemm_kwargs)
    required = (
        "tile_m",
        "tile_n",
        "tile_k",
        "split_k",
        "block_m_warps",
        "block_n_warps",
        "b_to_lds",
        "b_preshuffle",
    )
    missing = [key for key in required if key not in kwargs]
    if missing:
        raise ValueError(
            "FlyDSL HGEMM QR requires explicit kernel config values; "
            f"missing: {', '.join(missing)}"
        )
    if not supports_hgemm_qr(
        tile_m=kwargs["tile_m"],
        tile_n=kwargs["tile_n"],
        tile_k=kwargs["tile_k"],
        split_k=kwargs["split_k"],
        block_m_warps=kwargs["block_m_warps"],
        block_n_warps=kwargs["block_n_warps"],
        pack_n=1,
    ):
        raise ValueError(
            "Current config is not supported by the single-kernel HGEMM QR path"
        )
    return kwargs


def _run_hgemm_qr(
    world_size: int,
    rank: Int32,
    buffer_ptrs: Int64,
    flags_phase_bytes: Int64,
    data_offset: Int64,
    phase_bytes: Int64,
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    shuffle_b: bool = False,
    hgemm_kwargs: Optional[dict] = None,
    stream: Optional[torch.cuda.Stream] = None,
):
    launch_stream = _normalize_launch_stream(a.device, stream)
    signal_state = _get_split_k_signal_state(launch_stream)
    semaphore = _get_split_k_global_semaphore(launch_stream)

    k = a.shape[-1]
    a = a.view(-1, k)
    m = a.shape[0]
    n = b.shape[0]
    assert b.shape[1] == k
    c = c.view(-1, n)
    assert c.shape[0] == m

    kwargs = _normalize_hgemm_qr_kwargs(hgemm_kwargs)
    if a.dtype == torch.float16:
        exe = compile_hgemm_qr_kernel(world_size, "f16", n, k, **{
            "TILE_M": kwargs["tile_m"],
            "TILE_N": kwargs["tile_n"],
            "TILE_K": kwargs["tile_k"],
            "SPLIT_K": kwargs["split_k"],
            "BLOCK_M_WARPS": kwargs["block_m_warps"],
            "BLOCK_N_WARPS": kwargs["block_n_warps"],
            "B_PRE_SHUFFLE": kwargs["b_preshuffle"],
            "B_TO_LDS": kwargs["b_to_lds"],
        })
    elif a.dtype == torch.bfloat16:
        exe = compile_hgemm_qr_kernel(world_size, "bf16", n, k, **{
            "TILE_M": kwargs["tile_m"],
            "TILE_N": kwargs["tile_n"],
            "TILE_K": kwargs["tile_k"],
            "SPLIT_K": kwargs["split_k"],
            "BLOCK_M_WARPS": kwargs["block_m_warps"],
            "BLOCK_N_WARPS": kwargs["block_n_warps"],
            "B_PRE_SHUFFLE": kwargs["b_preshuffle"],
            "B_TO_LDS": kwargs["b_to_lds"],
        })
    else:
        raise NotImplementedError(f"Unsupported dtype: {a.dtype}")

    if kwargs["b_preshuffle"] and shuffle_b:
        b = shuffle_weight(b, layout=_get_flydsl_shuffle_layout(1))

    exe_compiled = exe.compile(
        rank,
        buffer_ptrs,
        flags_phase_bytes,
        data_offset,
        phase_bytes,
        c,
        a,
        b,
        m,
        semaphore,
        signal_state,
        launch_stream,
    )
    exe_compiled(
        rank,
        buffer_ptrs,
        flags_phase_bytes,
        data_offset,
        phase_bytes,
        c,
        a,
        b,
        m,
        semaphore,
        signal_state,
        launch_stream,
    )


class _HGEMMQRBackend(FlyDSLQuickReduce):
    def hgemm_qr_fusion(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        *,
        shuffle_b: bool = False,
        hgemm_kwargs: Optional[dict] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        m, n = a.shape[0], b.shape[0]
        if c is None:
            c = torch.empty((m, n), dtype=a.dtype, device=a.device)
        if not self.should_fuse(c):
            raise ValueError("Output tensor is not supported by FlyDSL HGEMM QR")

        kwargs = _normalize_hgemm_qr_kwargs(hgemm_kwargs)
        required_blocks = self.required_logical_blocks(
            m,
            n,
            kwargs["tile_m"],
            kwargs["tile_n"],
        )
        if required_blocks > self.max_blocks:
            raise ValueError(
                "FlyDSL HGEMM QR block capacity exceeded: "
                f"requires {required_blocks}, max supported is {self.max_blocks}"
            )

        launch_stream = _normalize_launch_stream(a.device, stream)
        rank = Int32(self.rank)
        buffer_ptrs = Int64(int(self._gpu_buffer_ptrs_array.data_ptr()))
        flags_phase_bytes = Int64(int(self.flags_phase_bytes))
        data_offset = Int64(int(self.data_offset))
        phase_bytes = Int64(int(self.phase_bytes))

        with torch.cuda.stream(launch_stream):
            _run_hgemm_qr(
                self.world_size,
                rank,
                buffer_ptrs,
                flags_phase_bytes,
                data_offset,
                phase_bytes,
                c,
                a,
                b,
                shuffle_b=shuffle_b,
                hgemm_kwargs=kwargs,
                stream=launch_stream,
            )
        return c


def _get_tp_flydsl_hgemm_qr_backend(
    device: torch.device,
    *,
    max_size: int,
) -> _HGEMMQRBackend:
    tp_group = get_tp_group()
    device_index = device.index
    if device_index is None:
        raise ValueError(f"Unable to determine device index for {device}")

    ranks = tuple(tp_group.ranks)
    cache_key = (ranks, device_index, int(max_size))
    backend = _BACKEND_CACHE.get(cache_key)
    if backend is not None:
        return backend

    ca_comm = getattr(getattr(tp_group, "device_communicator", None), "ca_comm", None)
    full_nvlink = getattr(ca_comm, "fully_connected", True)
    backend = _HGEMMQRBackend(
        group=tp_group.cpu_group,
        device=device,
        max_size=max_size,
        world_size=tp_group.world_size,
        rank=tp_group.rank_in_group,
        full_nvlink=full_nvlink,
    )
    _BACKEND_CACHE[cache_key] = backend
    return backend


def flydsl_hgemm_qr(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 64,
    pack_n: int = 1,
    split_k: int = 1,
    block_m_warps: int = 2,
    block_n_warps: int = 2,
    stages: int = FIXED_STAGE,
    async_copy: bool = False,
    b_to_lds: bool = True,
    b_preshuffle: bool = False,
    auto_shuffle_b: bool = False,
    c_to_lds: bool = False,
    stream: Optional[torch.cuda.Stream] = None,
    max_size: int = _DEFAULT_MAX_SIZE,
) -> torch.Tensor:
    m, n, k = _validate_hgemm_inputs(a, b, out)
    _normalize_supported_kernel_metadata(
        stage=stages,
        async_copy=async_copy,
        c_to_lds=c_to_lds,
    )
    _validate_hgemm_tiling(
        m,
        n,
        k,
        dtype=_to_kernel_dtype(a.dtype),
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        pack_n=pack_n,
        split_k=split_k,
        stages=stages,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
        b_to_lds=b_to_lds,
    )
    if not supports_hgemm_qr(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        split_k=split_k,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
        pack_n=pack_n,
    ):
        raise ValueError("Current config is not supported by FlyDSL HGEMM QR")

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    if b_preshuffle and not getattr(b, "is_shuffled", False):
        if auto_shuffle_b:
            b = shuffle_weight(b, layout=_get_flydsl_shuffle_layout(pack_n))
        else:
            raise ValueError(
                "`b_preshuffle=True` expects `b` to be pre-shuffled. "
                f"Use `shuffle_weight(b, layout={_get_flydsl_shuffle_layout(pack_n)})` "
                "first or pass `auto_shuffle_b=True`."
            )

    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    out_bytes = int(out.numel()) * int(out.element_size())
    required_blocks = ((m + int(tile_m) - 1) // int(tile_m)) * (n // int(tile_n))
    effective_max_size = max(
        out_bytes,
        required_blocks * int(_QR_MIN_LOGICAL_TILE_BYTES),
    )
    backend = _get_tp_flydsl_hgemm_qr_backend(a.device, max_size=effective_max_size)
    if not backend.should_fuse(out):
        raise ValueError("Current tensor-parallel setup does not support FlyDSL HGEMM QR")
    return backend.hgemm_qr_fusion(
        a,
        b,
        out,
        shuffle_b=False,
        hgemm_kwargs={
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "split_k": split_k,
            "block_m_warps": block_m_warps,
            "block_n_warps": block_n_warps,
            "b_to_lds": b_to_lds,
            "b_preshuffle": b_preshuffle,
        },
        stream=stream,
    )
