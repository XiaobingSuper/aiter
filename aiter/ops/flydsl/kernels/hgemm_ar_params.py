# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Supported HGEMM+TP-allreduce kernel names for tuning CSVs."""

from __future__ import annotations

import re
from itertools import product
from typing import Any, Dict, Optional

from flydsl.runtime.device import get_rocm_arch

from ..gemm_kernels import FIXED_STAGE, _validate_hgemm_tiling

KERNEL_ASYNC_COPY = get_rocm_arch() != "gfx942"

_KERNEL_NAME_RE = re.compile(
    r"^hgemm_ar_"
    r"(?P<dtype>bf16|f16)_"
    r"(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)"
    r"_S(?P<stages>\d+)TN_"
    r"(?P<async_copy>AS|NA)_W"
    r"(?P<block_m_warps>\d+)x(?P<block_n_warps>\d+)"
    r"(?P<b_preshuffle>_BP)?"
    r"(?P<split_k>_SPK\d+)?"
    r"(?P<b_to_lds>_BS)?"
    r"(?P<use_atomic_add>_CATOM)?$"
)

TP_HGEMM_AR_CONFIG_SELECTIONS = {
    "tile_k": (64, 128),
    "tile_m": (16, 32, 48, 64, 96, 128),
    "tile_n": (64, 128, 256),
    "split_k": (1, 2, 4, 8, 16),
}

TP_HGEMM_AR_CONFIG_VARIANTS = (
    {
        "block_m_warps": 1,
        "block_n_warps": 4,
        "b_to_lds": False,
    },
    {
        "block_m_warps": 2,
        "block_n_warps": 2,
        "b_to_lds": True,
    },
)


def _uses_cross_device_atomic(split_k: int, use_atomic_add: bool) -> bool:
    # Latest upstream forces cross-device atomic for split-k kernels.
    return bool(use_atomic_add) or int(split_k) > 1


def _normalize_registry_config(
    *,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
    block_m_warps: int,
    block_n_warps: int,
    b_to_lds: bool,
    b_preshuffle: bool,
    use_atomic_add: bool,
) -> Optional[Dict[str, Any]]:
    config = {
        "tile_m": int(tile_m),
        "tile_n": int(tile_n),
        "tile_k": int(tile_k),
        "splitK": int(split_k),
        "block_m_warps": int(block_m_warps),
        "block_n_warps": int(block_n_warps),
        "b_to_lds": bool(b_to_lds),
        "bpreshuffle": bool(b_preshuffle),
        "use_atomic_add": _uses_cross_device_atomic(split_k, use_atomic_add),
    }
    if config["bpreshuffle"] and config["b_to_lds"]:
        return None

    try:
        _validate_hgemm_tiling(
            1,
            config["tile_n"],
            config["tile_k"] * config["splitK"],
            dtype=dtype,
            tile_m=config["tile_m"],
            tile_n=config["tile_n"],
            tile_k=config["tile_k"],
            pack_n=1,
            split_k=config["splitK"],
            stages=FIXED_STAGE,
            block_m_warps=config["block_m_warps"],
            block_n_warps=config["block_n_warps"],
            b_to_lds=config["b_to_lds"],
        )
    except ValueError:
        return None

    return config


def flydsl_tp_hgemm_ar_kernel_name(
    dtype: str,
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
    block_m_warps: int,
    block_n_warps: int,
    b_preshuffle: bool,
    b_to_lds: bool,
    use_atomic_add: bool = False,
    stages: int = 2,
    async_copy: bool = True,
) -> str:
    """Stable kernel name for a registered HGEMM+AR config."""
    if b_preshuffle and b_to_lds:
        raise ValueError("HGEMM+AR does not support b_preshuffle=True with b_to_lds=True")
    use_cross_device_atomic = _uses_cross_device_atomic(split_k, use_atomic_add)
    name = (
        f"hgemm_ar_{dtype}_{int(tile_m)}x{int(tile_n)}x{int(tile_k)}"
        f"_S{int(stages)}TN_{'AS' if async_copy else 'NA'}"
        f"_W{int(block_m_warps)}x{int(block_n_warps)}"
    )
    if b_preshuffle:
        name += "_BP"
    if int(split_k) > 1:
        name += f"_SPK{int(split_k)}"
    if b_to_lds:
        name += "_BS"
    if use_cross_device_atomic:
        name += "_CATOM"
    return name


def _register_supported_hgemm_ar_kernels() -> Dict[str, Dict[str, Any]]:
    kernels: Dict[str, Dict[str, Any]] = {}
    for dtype in ("bf16", "f16"):
        for tile_m, tile_n, tile_k, split_k, b_preshuffle, variant, use_atomic_add in product(
            TP_HGEMM_AR_CONFIG_SELECTIONS["tile_m"],
            TP_HGEMM_AR_CONFIG_SELECTIONS["tile_n"],
            TP_HGEMM_AR_CONFIG_SELECTIONS["tile_k"],
            TP_HGEMM_AR_CONFIG_SELECTIONS["split_k"],
            (False, True),
            TP_HGEMM_AR_CONFIG_VARIANTS,
            (False, True),
        ):
            config = _normalize_registry_config(
                dtype=dtype,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                split_k=split_k,
                block_m_warps=variant["block_m_warps"],
                block_n_warps=variant["block_n_warps"],
                b_to_lds=variant["b_to_lds"],
                b_preshuffle=b_preshuffle,
                use_atomic_add=use_atomic_add,
            )
            if config is None:
                continue
            name = flydsl_tp_hgemm_ar_kernel_name(
                dtype,
                tile_m=config["tile_m"],
                tile_n=config["tile_n"],
                tile_k=config["tile_k"],
                split_k=config["splitK"],
                block_m_warps=config["block_m_warps"],
                block_n_warps=config["block_n_warps"],
                b_preshuffle=config["bpreshuffle"],
                b_to_lds=config["b_to_lds"],
                use_atomic_add=config["use_atomic_add"],
                stages=FIXED_STAGE,
                async_copy=KERNEL_ASYNC_COPY,
            )
            kernels[name] = config
    return kernels


_TP_HGEMM_AR_KERNELS = _register_supported_hgemm_ar_kernels()


def _parse_hgemm_ar_kernel_name(name: str) -> Optional[Dict[str, Any]]:
    match = _KERNEL_NAME_RE.fullmatch(str(name).strip())
    if match is None:
        return None

    groups = match.groupdict()
    stages = int(groups["stages"])
    async_copy = groups["async_copy"] == "AS"
    if stages != FIXED_STAGE or async_copy != KERNEL_ASYNC_COPY:
        return None

    split_k_group = groups["split_k"]
    split_k = 1 if split_k_group is None else int(split_k_group.removeprefix("_SPK"))
    return _normalize_registry_config(
        dtype=groups["dtype"],
        tile_m=int(groups["tile_m"]),
        tile_n=int(groups["tile_n"]),
        tile_k=int(groups["tile_k"]),
        split_k=split_k,
        block_m_warps=int(groups["block_m_warps"]),
        block_n_warps=int(groups["block_n_warps"]),
        b_to_lds=groups["b_to_lds"] is not None,
        b_preshuffle=groups["b_preshuffle"] is not None,
        use_atomic_add=groups["use_atomic_add"] is not None,
    )


def get_flydsl_tp_hgemm_ar_kernel_params(name: str) -> Optional[Dict[str, Any]]:
    """Look up a registered `kernelName` for fused HGEMM+AR."""
    config = _TP_HGEMM_AR_KERNELS.get(name)
    if config is None:
        config = _parse_hgemm_ar_kernel_name(name)
    if config is None:
        return None
    return dict(config)
