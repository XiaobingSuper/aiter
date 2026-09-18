# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL-side coverage for the a16w4 (bf16 A x MXFP4 W) SiTUv2 MoE kernel.

Exercises the NEW shared a16w-mix a16w4 kernel
(``aiter/ops/flydsl/kernels/moe_2stage_a16wmix``) through the production
``fused_moe`` a16w4 path (the standard 2-stage FlyDSL dispatch:
``get_2stage_cfgs`` -> ``fused_moe_2stages`` -> the a16w4 branches of
``_flydsl_moe_stage{1,2}_impl`` -> ``flydsl_a16w4_gemm{1,2}``, built via
``compile_flydsl_moe_stage{1,2}``), in the SiTUv2 / SEPARATED bf16-activation x
mxfp4-weight configuration, against a bf16 SiTUv2 torch reference with a strict
cos/logits_diff gate.

This is the explicit FlyDSL-side test complementing the routed a16w4 rows of
``op_tests/test_moe_2stage.py``. It goes through ``fused_moe`` (or
``flydsl_a16w4_gemm1/2``), NOT the removed low-level ``compile_mixed_moe_gemm1_a16w4``
API.

Run:
    pytest op_tests/flydsl_tests/test_flydsl_moe_a16wfp4.py -q
"""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

_SKIP = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="CDNA (gfx942/gfx950) + FlyDSL required for a16w4 SiTUv2",
)


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom)


@pytest.mark.skipif(get_gfx() != "gfx950", reason="requires K32 BF16 MFMA")
@pytest.mark.parametrize("model_dim", [1024, 2048])
@pytest.mark.parametrize("w_dtype", ["fp4", "int4", "bf16"])
@pytest.mark.parametrize(
    "tile_n,tile_k,k_wave",
    [
        (64, 128, 1),
        (64, 256, 2),
        (64, 128, 4),
        (128, 256, 1),
        (192, 128, 1),
        (64, 512, 1),
        (64, 256, 4),
        (32, 256, 2),
    ],
)
def test_bm16_stage1_lds_layout(model_dim, w_dtype, tile_n, tile_k, k_wave):
    from aiter.fused_moe import moe_sorting
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import flydsl_a16w4_gemm1
    from aiter.ops.quant import per_1x32_f4_quant, per_1x32_i4_quant
    from aiter.ops.shuffle import (
        pack_int8_to_packed_int4,
        shuffle_scale_for_int4,
        shuffle_weight,
    )
    from aiter.utility.fp4_utils import e8m0_shuffle, e8m0_to_f32, mxfp4_to_f32

    tokens, experts, inter, topk = 35, 7, 384, 2
    torch.manual_seed(0)
    x = torch.randn((tokens, model_dim), device="cuda", dtype=dtypes.bf16) / 10
    w = (
        torch.randn((experts, 2 * inter, model_dim), device="cuda", dtype=dtypes.bf16)
        / 10
    )
    if w_dtype == "fp4":
        q, scale = per_1x32_f4_quant(w, quant_dtype=dtypes.fp4x2, shuffle=False)
        dequant = (
            (
                mxfp4_to_f32(q).view(experts, 2 * inter, model_dim // 32, 32)
                * e8m0_to_f32(scale).view(experts, 2 * inter, model_dim // 32, 1)
            )
            .reshape_as(w)
            .to(dtypes.bf16)
        )
        packed = shuffle_weight(q.view(experts, 2 * inter, model_dim // 2), (16, 16))
        scale = e8m0_shuffle(scale)
    elif w_dtype == "int4":
        q, scale = per_1x32_i4_quant(w)
        dequant = (
            (
                q.float().view(experts, 2 * inter, model_dim // 32, 32)
                * scale.transpose(-1, -2).unsqueeze(-1)
            )
            .reshape_as(w)
            .to(dtypes.bf16)
        )
        packed = pack_int8_to_packed_int4(shuffle_weight(q, (16, 16)))
        scale = shuffle_scale_for_int4(scale, group_size=32).contiguous()
    else:
        dequant, packed = w, shuffle_weight(w, (16, 16))
        scale = torch.empty(0, device="cuda", dtype=torch.uint8)

    # Two shared experts exercise full BM16 blocks, a partial tail, and empty experts.
    ids = (
        torch.arange(topk, device="cuda", dtype=torch.int32)
        .expand(tokens, topk)
        .contiguous()
    )
    weights = torch.ones((tokens, topk), device="cuda")
    si, _, se, nv, _ = moe_sorting(ids, weights, experts, model_dim, dtypes.bf16, 16)
    out = torch.empty((si.numel(), inter), device="cuda", dtype=dtypes.bf16)
    flydsl_a16w4_gemm1(
        a_bf16=x,
        w1_u8=packed,
        w1_scale_u8=scale,
        sorted_expert_ids=se,
        cumsum_tensor=nv,
        m_indices=si,
        inter_sorted_bf16=out,
        n_tokens=tokens,
        NE=experts,
        D_HIDDEN=model_dim,
        D_INTER=inter,
        topk=topk,
        tile_m=16,
        tile_n=tile_n,
        tile_k=tile_k,
        k_wave=k_wave,
        w_dtype=w_dtype,
    )
    pos = torch.arange(int(nv[0]), device="cuda")
    pos = pos[(si[pos] & 0xFFFFFF) < tokens]
    assert pos.numel() == tokens * topk
    projected = torch.einsum("mh,enh->men", x.float(), dequant.float())
    gate, up = projected.chunk(2, dim=-1)
    ref = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(
        out[pos].float(),
        ref[si[pos] & 0xFFFFFF, se[pos // 16].long()],
        rtol=0.02,
        atol=0.005,
    )


@_SKIP
@pytest.mark.parametrize("model_dim,inter_dim", [(3584, 512), (3584, 384)])
@pytest.mark.parametrize("token", [1, 16, 128])
# (1,1) = plain tanh; (4,25) = kimi-k3 production betas, which exercise the
# runtime situ_beta/situ_linear_beta f32 args (beta is NOT a compile key, so this
# adds no extra kernel compiles).
@pytest.mark.parametrize("situ_beta,situ_linear_beta", [(1.0, 1.0), (4.0, 25.0)])
def test_flydsl_a16wfp4_situv2_e2e(
    model_dim, inter_dim, token, situ_beta, situ_linear_beta
):
    """a16w4 SiTUv2 SEPARATED end-to-end through fused_moe vs a bf16 SiTUv2 ref."""
    E, topk = 896, 16
    dtype = dtypes.bf16
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device="cuda")
    score = torch.randn((token, E), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = tq(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = tq(w2, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    w1_scale_e = w1_scale.view(E, inter_dim * 2, model_dim // 32)
    w2_scale_e = w2_scale.view(E, model_dim, inter_dim // 32)

    # bf16 SiTUv2 SEPARATED reference
    o1 = torch_moe_stage1(
        inp.to(dtype),
        w1_qt.view(dtypes.fp4x2),
        w2_qt.view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=w1_scale_e,
        doweight=False,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    ref = torch_moe_stage2(
        o1.view(token, topk, inter_dim),
        w1_qt.view(dtypes.fp4x2),
        w2_qt.view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale_e,
        a2_scale=None,
        doweight=True,
    )

    # caller contract: standard GGUU (separated gate/up) W1 layout, matching main
    w1_gui = shuffle_weight_a16w4(w1_qt, 16, False)
    w2_gui = shuffle_weight_a16w4(w2_qt, 16, False)
    w1_scale_gui = shuffle_scale_a16w4(w1_scale, E, False)
    w2_scale_gui = shuffle_scale_a16w4(w2_scale, E, False)

    out = fused_moe(
        inp,
        w1_gui,
        w2_gui,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale_gui,
        w2_scale=w2_scale_gui,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Situv2,
        doweight_stage1=False,
        gate_mode=GateMode.SEPARATED.value,
        beta=situ_beta,
        linear_beta=situ_linear_beta,
    )

    assert not out.isnan().any().item(), "a16w4 SiTUv2 output contains NaN"
    ld = _cos_diff(ref.float(), out.float())
    assert ld < 1e-2, f"a16w4 SiTUv2 cos/logits_diff too large: {ld:.3e}"
