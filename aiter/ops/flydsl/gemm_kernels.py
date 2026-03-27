# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL HGEMM APIs."""

from __future__ import annotations

from typing import Optional

import torch

from .kernels.preshuffle_splitk_hgemm import (
    compile_hgemm_kernel,
    hgemm_spk_shuffle_b,
)

__all__ = [
    "compile_flydsl_hgemm",
    "flydsl_hgemm",
    "flydsl_hgemm_shuffle_b",
]


def _to_kernel_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"Only fp16/bf16 are supported, got {dtype!r}")


def _validate_hgemm_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor],
) -> tuple[int, int, int]:
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"`flydsl_hgemm` expects 2D inputs, got a.dim={a.dim()} b.dim={b.dim()}"
        )
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise ValueError("`flydsl_hgemm` only supports CUDA/ROCm tensors")
    if a.device != b.device:
        raise ValueError(
            f"`a` and `b` must be on the same device, got {a.device=} {b.device=}"
        )
    if a.dtype != b.dtype:
        raise ValueError(
            f"`a` and `b` must have the same dtype, got {a.dtype=} {b.dtype=}"
        )

    m, k = a.shape
    n, bk = b.shape
    if k != bk:
        raise ValueError(f"Incompatible GEMM shapes: a={tuple(a.shape)} b={tuple(b.shape)}")

    if out is not None:
        if out.shape != (m, n):
            raise ValueError(f"`out` must have shape {(m, n)}, got {tuple(out.shape)}")
        if out.dtype != a.dtype:
            raise ValueError(
                f"`out` dtype must match input dtype, got {out.dtype=} {a.dtype=}"
            )
        if out.device != a.device:
            raise ValueError(f"`out` must be on {a.device}, got {out.device}")
        if not out.is_contiguous():
            raise ValueError("`out` must be contiguous")

    return m, n, k


def _validate_hgemm_tiling(
    m: int,
    n: int,
    k: int,
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    pack_n: int,
    split_k: int,
    block_m_warps: int,
    block_n_warps: int,
) -> None:
    del m

    if pack_n != 1:
        raise ValueError(
            "Latest `hgemm.py` kernel only supports `pack_n=1`; "
            f"got pack_n={pack_n}"
        )

    warp_atom_m = 16
    warp_atom_n = 16

    if tile_m % (block_m_warps * warp_atom_m) != 0:
        raise ValueError(
            f"Invalid tiling: tile_m={tile_m} must be divisible by "
            f"block_m_warps * 16 = {block_m_warps * warp_atom_m}"
        )
    if tile_n % (block_n_warps * warp_atom_n) != 0:
        raise ValueError(
            f"Invalid tiling: tile_n={tile_n} must be divisible by "
            f"block_n_warps * 16 = {block_n_warps * warp_atom_n}"
        )

    block_n = tile_n
    if n < block_n or n % block_n != 0:
        raise ValueError(
            f"Invalid N for this kernel: N={n} must satisfy N >= {block_n} and N % {block_n} == 0"
        )

    if k % split_k != 0:
        raise ValueError(f"Invalid split-K: K={k} must be divisible by split_k={split_k}")

    ks = k // split_k
    if ks < tile_k or ks % tile_k != 0:
        raise ValueError(
            f"Invalid K for this kernel: K/split_k={ks} must satisfy "
            f">= tile_k={tile_k} and % tile_k == 0"
        )

    block_threads = block_m_warps * block_n_warps * 64
    ldg_vec_size = 8
    ldg_reg_a_count = (tile_m * tile_k) // ldg_vec_size // block_threads
    ldg_reg_b_count = (tile_n * tile_k) // ldg_vec_size // block_threads
    if ldg_reg_a_count < 1 or ldg_reg_b_count < 1:
        raise ValueError(
            "Invalid tile combination: requires at least one vectorized global load per thread "
            f"(got ldg_reg_a_count={ldg_reg_a_count}, ldg_reg_b_count={ldg_reg_b_count})"
        )


def compile_flydsl_hgemm(
    dtype: str,
    m: int,
    n: int,
    k: int,
    *,
    tile_k: int = 64,
    block_m_warps: int = 1,
    block_n_warps: int = 4,
    tile_m: int = 128,
    tile_n: int = 128,
    pack_n: int = 1,
    stages: int = 2,
    async_copy: bool = False,
    b_to_lds: bool = False,
    b_preshuffle: bool = True,
    split_k: int = 1,
    c_to_lds: bool = False,
):
    """Compile and cache a FlyDSL HGEMM kernel launcher."""

    if dtype not in {"f16", "bf16"}:
        raise ValueError(f"`dtype` must be 'f16' or 'bf16', got {dtype!r}")

    _validate_hgemm_tiling(
        m,
        n,
        k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        pack_n=pack_n,
        split_k=split_k,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
    )

    return compile_hgemm_kernel(
        dtype,
        m,
        n,
        k,
        TILE_K=tile_k,
        BLOCK_M_WARPS=block_m_warps,
        BLOCK_N_WARPS=block_n_warps,
        TILE_M=tile_m,
        TILE_N=tile_n,
        STAGES=stages,
        ASYNC_COPY=async_copy,
        B_TO_LDS=b_to_lds,
        B_PRE_SHUFFLE=b_preshuffle,
        SPLIT_K=split_k,
        C_TO_LDS=c_to_lds,
    )


def flydsl_hgemm_shuffle_b(
    weight: torch.Tensor,
    *,
    layout: tuple[int, int] = (16, 16),
    pack_n: int = 1,
    k_steps: int = 2,
) -> torch.Tensor:
    """Convert `(N, K)` weight to the preshuffled layout expected by the kernel."""

    if weight.dim() != 2:
        raise ValueError(f"`weight` must be 2D, got {weight.dim()}D")
    if weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Only fp16/bf16 are supported, got {weight.dtype!r}")
    return hgemm_spk_shuffle_b(
        weight.contiguous(),
        layout=layout,
        pack_n=pack_n,
        k_steps=k_steps,
    )


def flydsl_hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 64,
    pack_n: int = 1,
    split_k: int = 1,
    block_m_warps: int = 1,
    block_n_warps: int = 4,
    stages: int = 2,
    async_copy: bool = False,
    b_to_lds: bool = False,
    b_preshuffle: bool = True,
    auto_shuffle_b: bool = False,
    c_to_lds: bool = False,
) -> torch.Tensor:
    """Run FlyDSL HGEMM.

    `a` is `(M, K)`.
    `b` is `(N, K)`, optionally pre-shuffled via `flydsl_hgemm_shuffle_b()`.
    Returns `(M, N)`.
    """

    m, n, k = _validate_hgemm_inputs(a, b, out)
    kernel_dtype = _to_kernel_dtype(a.dtype)

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    if b_preshuffle and not getattr(b, "is_shuffled", False):
        if auto_shuffle_b:
            b = flydsl_hgemm_shuffle_b(b, pack_n=pack_n)
        else:
            raise ValueError(
                "`b_preshuffle=True` expects `b` to be pre-shuffled. "
                "Use `flydsl_hgemm_shuffle_b()` first or pass `auto_shuffle_b=True`."
            )

    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)

    launcher = compile_flydsl_hgemm(
        kernel_dtype,
        m,
        n,
        k,
        tile_k=tile_k,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
        tile_m=tile_m,
        tile_n=tile_n,
        pack_n=pack_n,
        stages=stages,
        async_copy=async_copy,
        b_to_lds=b_to_lds,
        b_preshuffle=b_preshuffle,
        split_k=split_k,
        c_to_lds=c_to_lds,
    )

    if split_k > 1:
        out.zero_()

    launcher(out, a, b)
    return out
