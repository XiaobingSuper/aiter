"""FlyDSL fused HGEMM + tensor-parallel all-reduce (high-level API)."""

from __future__ import annotations

import os
from typing import Optional

import torch
from flydsl.expr.typing import Int32, Int64

from aiter.dist.parallel_state import get_tp_group

from ..shuffle import shuffle_weight
from .gemm_kernels import (
    FIXED_STAGE,
    SPLIT_K_COUNTER_MAX_LEN,
    _advance_split_k_signal_state,
    _get_flydsl_shuffle_layout,
    _get_split_k_global_semaphore,
    _get_split_k_signal_state,
    _normalize_launch_stream,
    _normalize_supported_kernel_metadata,
    _to_kernel_dtype,
    _validate_hgemm_inputs,
    _validate_hgemm_tiling,
    flydsl_hgemm,
)
from .hgemm_qr import (
    experimental_hgemm_qr_enabled,
    flydsl_hgemm_qr,
    supports_hgemm_qr,
)
from .kernels.custom_all_reduce import FlyDSLAllreduce, _DEFAULT_MAX_SIZE
from .kernels.splitk_hgemm_ar import compile_hgemm_ar_kernel

__all__ = ["flydsl_hgemm_ar"]

_BACKEND_CACHE: dict[tuple[tuple[int, ...], int, int], "_GEMMARBackend"] = {}
HGEMM_AR_MAX_BLOCKS = 80

_KWARG_MAP = {
    "tile_m": "TILE_M",
    "tile_n": "TILE_N",
    "tile_k": "TILE_K",
    "split_k": "SPLIT_K",
    "block_m_warps": "BLOCK_M_WARPS",
    "block_n_warps": "BLOCK_N_WARPS",
    "b_preshuffle": "B_PRE_SHUFFLE",
    "b_to_lds": "B_TO_LDS",
    "use_atomic_add": "USE_ATOMIC_ADD",
}

_REQUIRED_HGEMM_AR_KWARGS = (
    "TILE_M",
    "TILE_N",
    "TILE_K",
    "SPLIT_K",
    "BLOCK_M_WARPS",
    "BLOCK_N_WARPS",
    "B_PRE_SHUFFLE",
    "B_TO_LDS",
)
_QUICK_REDUCE_QUANTIZATIONS = {"FP", "FP8", "INT6", "INT4"}


def _check_hgemm_ar_block_capacity(
    m: int,
    n: int,
    tile_m: int,
    tile_n: int,
    split_k: int,
) -> None:
    bm = (m + tile_m - 1) // tile_m
    bn = n // tile_n
    required = bm * bn * split_k
    # The HGEMM+AR signal transport still only reserves 80 logical block slots.
    limit = min(HGEMM_AR_MAX_BLOCKS, SPLIT_K_COUNTER_MAX_LEN)
    if required > limit:
        raise ValueError(
            "Fused FlyDSL HGEMM AR block capacity exceeded: "
            f"requires {required} logical blocks, max supported is {limit}"
        )


def _normalize_hgemm_ar_kwargs(hgemm_kwargs: Optional[dict]) -> dict:
    if not hgemm_kwargs:
        raise ValueError(
            "FlyDSL HGEMM AR requires an explicit kernel config; "
            "no default kernel config is defined."
        )
    kwargs = {}
    for key, value in hgemm_kwargs.items():
        if value is None:
            continue
        mapped = _KWARG_MAP.get(key, key)
        kwargs[mapped] = value
    missing = [key for key in _REQUIRED_HGEMM_AR_KWARGS if key not in kwargs]
    if missing:
        raise ValueError(
            "FlyDSL HGEMM AR requires explicit kernel config values; "
            f"missing: {', '.join(missing)}"
        )
    # Latest upstream forces cross-device atomic for split-k kernels.
    kwargs["USE_ATOMIC_ADD"] = bool(kwargs.get("USE_ATOMIC_ADD", False)) or int(
        kwargs["SPLIT_K"]
    ) > 1
    return kwargs


def _is_quick_reduce_enabled() -> bool:
    regime = str(os.environ.get("AITER_QUICK_REDUCE_QUANTIZATION", "")).strip().upper()
    return regime in _QUICK_REDUCE_QUANTIZATIONS


def _get_quick_reduce_comm(out: torch.Tensor):
    if not _is_quick_reduce_enabled():
        return None

    tp_group = get_tp_group()
    device_communicator = getattr(tp_group, "device_communicator", None)
    qr_comm = getattr(device_communicator, "qr_comm", None)
    if qr_comm is None or getattr(qr_comm, "disabled", True):
        return None
    if not qr_comm.should_quick_allreduce(out):
        return None
    return qr_comm


def _run_hgemm_then_quick_reduce(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    pack_n: int,
    split_k: int,
    block_m_warps: int,
    block_n_warps: int,
    stages: int,
    async_copy: bool,
    b_to_lds: bool,
    b_preshuffle: bool,
    auto_shuffle_b: bool,
    c_to_lds: bool,
    stream: Optional[torch.cuda.Stream],
    qr_comm,
) -> torch.Tensor:
    launch_stream = _normalize_launch_stream(a.device, stream)
    with torch.cuda.stream(launch_stream):
        out = flydsl_hgemm(
            a,
            b,
            out=out,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            pack_n=pack_n,
            split_k=split_k,
            block_m_warps=block_m_warps,
            block_n_warps=block_n_warps,
            stages=stages,
            async_copy=async_copy,
            b_to_lds=b_to_lds,
            b_preshuffle=b_preshuffle,
            auto_shuffle_b=auto_shuffle_b,
            c_to_lds=c_to_lds,
            stream=launch_stream,
        )
        # QuickReduce reads the full input tile into registers before writing
        # the reduced result, so aliasing inp/out avoids an extra buffer.
        qr_comm.quick_all_reduce(out, out=out)
    return out


def _run_hgemm_ar(
    world_size: int,
    rank: Int32,
    self_sg: Int64,
    sg_ptrs: Int64,
    tmp_ptrs: Int64,
    out_ptrs: Int64,
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

    kwargs = _normalize_hgemm_ar_kwargs(hgemm_kwargs)
    if a.dtype == torch.float16:
        exe = compile_hgemm_ar_kernel(world_size, "f16", n, k, **kwargs)
    elif a.dtype == torch.bfloat16:
        exe = compile_hgemm_ar_kernel(world_size, "bf16", n, k, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported dtype: {a.dtype}")

    if kwargs["B_PRE_SHUFFLE"] and shuffle_b:
        b = shuffle_weight(b, layout=_get_flydsl_shuffle_layout(1))
    _check_hgemm_ar_block_capacity(
        m,
        n,
        kwargs["TILE_M"],
        kwargs["TILE_N"],
        kwargs["SPLIT_K"],
    )

    exe_compiled = exe.compile(
        rank,
        self_sg,
        sg_ptrs,
        tmp_ptrs,
        out_ptrs,
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
        self_sg,
        sg_ptrs,
        tmp_ptrs,
        out_ptrs,
        c,
        a,
        b,
        m,
        semaphore,
        signal_state,
        launch_stream,
    )
    if kwargs["SPLIT_K"] > 1:
        _advance_split_k_signal_state(launch_stream)


class _GEMMARBackend(FlyDSLAllreduce):
    def hgemm_ar_fusion(
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
            out_bytes = int(c.numel()) * int(c.element_size())
            raise ValueError(
                f"Output tensor with {out_bytes} bytes is not supported by FlyDSL HGEMM AR"
            )

        launch_stream = _normalize_launch_stream(a.device, stream)
        rank = Int32(self.rank)
        self_sg = Int64(self._self_sg)
        sg_ptrs = Int64(int(self._gpu_sg_ptrs_array.data_ptr()))
        tmp_ptrs = Int64(int(self._gpu_tmp_ptrs_array.data_ptr()))
        out_bytes = int(c.numel()) * int(c.element_size())
        is_graph_capture = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        if is_graph_capture:
            graph_out_ptrs = self._get_or_create_graph_out_ptrs(c)
            out_ptrs = Int64(int(graph_out_ptrs.data_ptr()))
        else:
            out_ptrs = Int64(int(self._gpu_output_buffer_ptrs_array.data_ptr()))

        with torch.cuda.stream(launch_stream):
            _run_hgemm_ar(
                self.world_size,
                rank,
                self_sg,
                sg_ptrs,
                tmp_ptrs,
                out_ptrs,
                c,
                a,
                b,
                shuffle_b=shuffle_b,
                hgemm_kwargs=hgemm_kwargs,
                stream=launch_stream,
            )
            if not is_graph_capture:
                c.view(-1).view(torch.uint8)[:out_bytes].copy_(
                    self.output_buffer[:out_bytes]
                )
        return c


def _get_tp_flydsl_hgemm_ar_backend(
    device: torch.device,
    *,
    max_size: int = _DEFAULT_MAX_SIZE,
) -> _GEMMARBackend:
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
    backend = _GEMMARBackend(
        group=tp_group.cpu_group,
        device=device,
        max_size=max_size,
        world_size=tp_group.world_size,
        rank=tp_group.rank_in_group,
        full_nvlink=full_nvlink,
    )
    _BACKEND_CACHE[cache_key] = backend
    return backend


def _get_tp_flydsl_hgemm_ar_capture_context(
    device: Optional[torch.device] = None,
    *,
    max_size: int = _DEFAULT_MAX_SIZE,
):
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return _get_tp_flydsl_hgemm_ar_backend(device, max_size=max_size).capture()


def flydsl_hgemm_ar(
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
    use_atomic_add: bool = False,
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
    if experimental_hgemm_qr_enabled() and supports_hgemm_qr(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        split_k=split_k,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
        pack_n=pack_n,
    ):
        try:
            return flydsl_hgemm_qr(
                a,
                b,
                out=out,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                pack_n=pack_n,
                split_k=split_k,
                block_m_warps=block_m_warps,
                block_n_warps=block_n_warps,
                stages=stages,
                async_copy=async_copy,
                b_to_lds=b_to_lds,
                b_preshuffle=b_preshuffle,
                auto_shuffle_b=auto_shuffle_b,
                c_to_lds=c_to_lds,
                stream=stream,
                max_size=max_size,
            )
        except ValueError:
            pass
    quick_reduce_comm = _get_quick_reduce_comm(out)
    if quick_reduce_comm is not None:
        return _run_hgemm_then_quick_reduce(
            a,
            b,
            out,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            pack_n=pack_n,
            split_k=split_k,
            block_m_warps=block_m_warps,
            block_n_warps=block_n_warps,
            stages=stages,
            async_copy=async_copy,
            b_to_lds=b_to_lds,
            b_preshuffle=b_preshuffle,
            auto_shuffle_b=auto_shuffle_b,
            c_to_lds=c_to_lds,
            stream=stream,
            qr_comm=quick_reduce_comm,
        )
    backend = _get_tp_flydsl_hgemm_ar_backend(a.device, max_size=max_size)
    if not backend.should_fuse(out):
        raise ValueError("Current tensor-parallel setup does not support FlyDSL HGEMM AR")
    return backend.hgemm_ar_fusion(
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
            "use_atomic_add": use_atomic_add,
        },
        stream=stream,
    )
