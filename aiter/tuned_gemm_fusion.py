"""
Dedicated GEMM + TP all-reduce fusion dispatcher.

This module intentionally keeps the original `tuned_gemm.py` untouched and
handles fused tuning config lookup, fused FlyDSL dispatch, and fallback to
`tgemm.mm + get_tp_group().all_reduce`.
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import aiter
import pandas as pd
import torch
from aiter import dtypes, logger
from aiter.dist.parallel_state import get_tp_group
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.flydsl.gemm_kernels import FIXED_C_TO_LDS, FIXED_STAGE, KERNEL_ASYNC_COPY
from aiter.ops.flydsl.kernels.hgemm_ar_params import get_flydsl_tp_hgemm_ar_kernel_params
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.tuned_gemm import tgemm
from torch import Tensor

this_dir = os.path.dirname(os.path.abspath(__file__))

FUSED_TUNE_ENV = "AITER_CONFIG_TP_GEMM_AR_BF16"
FUSED_TUNE_DEFAULT_PATH = AITER_CONFIGS.AITER_CONFIG_TP_GEMM_AR_BF16_FILE
FUSED_UNTUNED_PATH = f"{this_dir}/configs/bf16_untuned_tp_gemm_ar.csv"

fused_tuned_df = pd.DataFrame(
    columns=[
        "world_size",
        "M",
        "N",
        "K",
        "bias",
        "dtype",
        "outdtype",
        "scaleAB",
        "bpreshuffle",
    ]
)


def _get_fused_tune_path() -> str:
    return os.environ.get(FUSED_TUNE_ENV, AITER_CONFIGS.AITER_CONFIG_TP_GEMM_AR_BF16_FILE)


def _config_bool(value, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


@functools.lru_cache(maxsize=1)
def get_GEMM_A16W16_AR_config_():
    tuned_file = _get_fused_tune_path()
    if not os.path.exists(tuned_file):
        return {}
    df = pd.read_csv(tuned_file).drop_duplicates()
    index_cols = [
        "world_size",
        "cu_num",
        "M",
        "N",
        "K",
        "bias",
        "dtype",
        "outdtype",
        "scaleAB",
        "bpreshuffle",
        "libtype",
    ]
    required_in_csv = index_cols + ["kernelName"]
    missing = [col for col in required_in_csv if col not in df.columns]
    if missing:
        logger.warning(
            "Ignoring fused GEMM config file %s because required columns are missing: %s",
            tuned_file,
            missing,
        )
        return {}

    # Keep the full CSV row in the value dict. `to_dict("index")` would drop
    # index columns like `libtype`/`bpreshuffle`, but the dispatcher still
    # needs those fields after lookup.
    configs = {}
    for row in df.to_dict("records"):
        key = tuple(row[col] for col in index_cols)
        configs[key] = row
    return configs


@functools.lru_cache(maxsize=4096)
def get_GEMM_A16W16_AR_config(
    world_size: int,
    M: int,
    N: int,
    K: int,
    bias: bool,
    dtype: str,
    otype: str,
    scaleAB: bool = False,
    bpreshuffle: bool = False,
):
    cfg = get_GEMM_A16W16_AR_config_()
    cu_num = get_cu_num()

    for candidate_cu_num in (cu_num, -1):
        config = cfg.get(
            (
                world_size,
                candidate_cu_num,
                M,
                N,
                K,
                bias,
                str(dtype),
                str(otype),
                scaleAB,
                bpreshuffle,
                "flydsl_ar",
            )
        )
        if config is None:
            continue
        if config.get("libtype") == "flydsl_ar":
            kname = config.get("kernelName")
            if kname is None or pd.isna(kname) or not is_flydsl_available():
                continue
            flydsl_config = get_flydsl_tp_hgemm_ar_kernel_params(str(kname).strip())
            if flydsl_config is None:
                continue
            if _config_bool(config.get("bpreshuffle")) != bool(
                flydsl_config["bpreshuffle"]
            ):
                continue
        return dict(config)
    return None


def gen_gemm_a16w16_tp_allreduce_fake_tensor(
    A: Tensor,
    B: Tensor,
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
    ca_use_new: bool = True,
    ca_fp8_quant: bool = False,
) -> Tensor:
    del bias, scale_a, scale_b, scale_c, ca_use_new, ca_fp8_quant
    return torch.empty(
        *A.shape[:-1],
        B.shape[0],
        dtype=otype or A.dtype,
        device=A.device,
    )


def flydsl_gemm_ar(
    inp: Tensor,
    weights: Tensor,
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
    config: dict = None,
):
    assert (
        scale_a is None and scale_b is None and scale_c is None
    ), "FlyDSL hgemm_ar does not support scaling yet."
    flydsl_config = get_flydsl_tp_hgemm_ar_kernel_params(str(config["kernelName"]).strip())
    if flydsl_config is None:
        raise ValueError(f"Invalid FlyDSL HGEMM+AR kernel name: {config['kernelName']}")
    if _config_bool(config.get("bpreshuffle")) != bool(flydsl_config["bpreshuffle"]):
        raise ValueError(
            "Tuned bpreshuffle column does not match FlyDSL HGEMM+AR kernel config"
        )
    out = aiter.ops.flydsl.flydsl_hgemm_ar(
        inp,
        weights,
        tile_m=flydsl_config["tile_m"],
        tile_n=flydsl_config["tile_n"],
        tile_k=flydsl_config["tile_k"],
        split_k=flydsl_config["splitK"],
        stages=FIXED_STAGE,
        async_copy=KERNEL_ASYNC_COPY,
        block_m_warps=flydsl_config["block_m_warps"],
        block_n_warps=flydsl_config["block_n_warps"],
        b_to_lds=flydsl_config["b_to_lds"],
        b_preshuffle=flydsl_config["bpreshuffle"],
        c_to_lds=FIXED_C_TO_LDS,
        use_atomic_add=flydsl_config.get("use_atomic_add", False),
    )
    if bias is not None:
        out = out + bias
    if otype is not None and out.dtype != otype:
        out = out.to(otype)
    return out


def _can_try_fused(
    inp_view: Tensor,
    bias: Optional[Tensor],
    scale_a: Optional[Tensor],
    scale_b: Optional[Tensor],
    scale_c: Optional[Tensor],
    ca_use_new: bool,
    ca_fp8_quant: bool,
    config: Optional[dict],
) -> bool:
    return bool(
        config is not None
        and config.get("libtype") == "flydsl_ar"
        and is_flydsl_available()
        and bias is None
        and scale_a is None
        and scale_b is None
        and scale_c is None
        and ca_use_new
        and not ca_fp8_quant
        and inp_view.dtype in (dtypes.fp16, dtypes.bf16)
    )


@torch_compile_guard(gen_fake=gen_gemm_a16w16_tp_allreduce_fake_tensor)
def gemm_a16w16_tp_allreduce(
    A: Tensor,
    B: Tensor,
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
    ca_use_new: bool = True,
    ca_fp8_quant: bool = False,
) -> Tensor:
    tp_group = get_tp_group()
    if A.dim() >= 3:
        out = tgemm.mm(
            A,
            B,
            bias=bias,
            otype=otype,
            scale_a=scale_a,
            scale_b=scale_b,
            scale_c=scale_c,
        )
        return tp_group.all_reduce(
            out,
            ca_use_new=ca_use_new,
            ca_fp8_quant=ca_fp8_quant,
        )

    m, k = A.shape
    n = B.shape[0]
    bpreshuffle = hasattr(B, "is_shuffled") and B.is_shuffled is True
    otype = otype if otype is not None else A.dtype
    world_size = tp_group.world_size

    config = get_GEMM_A16W16_AR_config(
        world_size=world_size,
        M=m,
        N=n,
        K=k,
        bias=bias is not None,
        dtype=str(A.dtype),
        otype=str(otype),
        scaleAB=scale_a is not None or scale_b is not None,
        bpreshuffle=bpreshuffle,
    )

    if _can_try_fused(
        A,
        bias,
        scale_a,
        scale_b,
        scale_c,
        ca_use_new,
        ca_fp8_quant,
        config,
    ):
        try:
            out = flydsl_gemm_ar(
                A,
                B,
                bias=bias,
                otype=otype,
                scale_a=scale_a,
                scale_b=scale_b,
                scale_c=scale_c,
                config=config,
            )
            return out
        except Exception as exc:
            logger.info(
                "FlyDSL GEMM+AR fallback for world_size=%d M=%d N=%d K=%d: %s",
                world_size,
                m,
                n,
                k,
                exc,
            )

    out = tgemm.mm(
        A,
        B,
        bias=bias,
        otype=otype,
        scale_a=scale_a,
        scale_b=scale_b,
        scale_c=scale_c,
    )
    out = tp_group.all_reduce(
        out,
        ca_use_new=ca_use_new,
        ca_fp8_quant=ca_fp8_quant,
    )
    return out
