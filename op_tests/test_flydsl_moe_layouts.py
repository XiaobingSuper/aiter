# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""BM16 MoE: validate BF16 LDS layout, MXFP8 scales, and both GEMM2 layouts.

Run on gfx950:
    python op_tests/test_flydsl_moe_layouts.py
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2
from aiter.ops.flydsl.moe_kernels import (
    _run_moe_reduction,
    flydsl_moe_stage1,
    flydsl_moe_stage2,
)
from aiter.utility.fp4_utils import e8m0_to_f32
from csrc.ck_gemm_moe_2stages_codegen.mxfp4_v2_tune_utils import gen

pytestmark = pytest.mark.skipif(get_gfx() != "gfx950", reason="requires gfx950")


def _unshuffle_scale(scale):
    """Invert the CK 32-row / 8-column scale permutation using tensor axes."""
    rows, cols = scale.shape
    return (
        scale.view(torch.uint8)
        .view(rows // 32, cols // 8, 4, 16, 2, 2)
        .permute(0, 5, 3, 1, 4, 2)
        .reshape(rows, cols)
    )


@pytest.mark.parametrize("v2_layout", [False, True], ids=["legacy", "v2"])
@pytest.mark.parametrize(
    "tokens,inter_dim,tile_n,k_wave,zero_bias",
    [
        (1, 384, 64, 1, False),
        (2, 768, 64, 2, False),
        (1, 384, 64, 4, False),
        (33, 768, 64, 4, False),
        (17, 384, 64, 4, True),
        (17, 384, 128, 1, False),
        (33, 768, 64, 2, False),
        (64, 384, 64, 1, False),
    ],
)
def test_mxfp8_bm16_scales_and_stage2(
    tokens, inter_dim, tile_n, k_wave, zero_bias, v2_layout, monkeypatch
):
    # Check both epilogues even if a local environment forces reduce mode.
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "0")
    monkeypatch.delenv("FLYDSL_RUNTIME_RUN_ONLY", raising=False)
    model_dim, experts, topk, bm = 1024, 7, 5, 16
    with torch.device("cuda"):
        data = gen(
            tokens,
            model_dim,
            inter_dim,
            experts,
            topk,
            bm,
            adtype="fp8",
            b_dtype="fp8",
            activation=ActivationType.Swiglu,
        )
    base = data["base"]
    sti, sei, nv = (
        base[k] for k in ("sorted_ids", "sorted_expert_ids", "num_valid_ids")
    )
    inter, scale = flydsl_moe_stage1(
        a=data["a1_qt"],
        w1=base["w1_qt_shuf"],
        sorted_token_ids=sti,
        sorted_expert_ids=sei,
        num_valid_ids=nv,
        topk=topk,
        tile_m=bm,
        tile_n=tile_n,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="fp8",
        act="swiglu",
        w1_scale=base["w1_scale_shuf"],
        a1_scale=base["a1_scale_sort"],
        gate_mode="interleave",
        use_async_copy=True,
        waves_per_eu=2,
        k_wave=k_wave,
        v2_output_layout=v2_layout,
        # Forces the general CShuffle fallback, which must keep all waves
        # at barriers while limiting read/store lanes to the BM16 tile.
        bias=(
            torch.zeros((experts, 2 * inter_dim), device="cuda") if zero_bias else None
        ),
    )

    positions = torch.arange(sti.numel(), device="cuda")
    token_ids, slots = sti & 0xFFFFFF, (sti >> 24) & 0xFF
    valid = (positions < nv[0]) & (token_ids < tokens) & (slots < topk)
    positions, token_ids, slots = positions[valid], token_ids[valid], slots[valid]
    assert positions.numel() == tokens * topk
    assert positions.max().item() >= bm  # Exercise more than the first scale chunk.

    # Native BM16 uses only the first row group in each 32-row scale chunk.
    # Check decoded values, including their magnitude: per-group cosine alone
    # would fail to detect a wrong E8M0 exponent or the wrong chunk stride.
    scale_rows = (
        positions // bm * 32 + positions % bm if v2_layout and bm == 16 else positions
    )
    dense_scale = _unshuffle_scale(scale)[scale_rows, : inter_dim // 32]
    assert (dense_scale != 127).any(), "test must exercise non-unit activation scales"
    values = (
        inter.view(dtypes.fp8)[positions]
        if v2_layout
        else inter.view(tokens, topk, inter_dim)[token_ids, slots]
    )
    dequant = (
        values.float().view(-1, inter_dim // 32, 32)
        * e8m0_to_f32(dense_scale).unsqueeze(-1)
    ).reshape(-1, inter_dim)
    torch.testing.assert_close(
        dequant, data["ref1"][token_ids, slots].float(), rtol=0.08, atol=0.04
    )

    # Isolate GEMM2 from GEMM1 quantization error by using the actual decoded
    # intermediate as the reference operand.
    ref_inter = torch.empty((tokens, topk, inter_dim), device="cuda")
    ref_inter[token_ids, slots] = dequant
    ref = torch_moe_stage2(
        ref_inter,
        data["w1_qt"],
        data["w2_qt"],
        data["topk_weights"],
        data["topk_ids"],
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=data["w2_scale"],
    )
    for epilog in ("atomic", "reduce"):
        out = torch.zeros((tokens, model_dim), dtype=dtypes.bf16, device="cuda")
        if v2_layout:
            target = (
                torch.empty((tokens, topk, model_dim), dtype=dtypes.bf16, device="cuda")
                if epilog == "reduce"
                else out
            )
            mxfp4_moe_gemm2(
                inter_sorted_quant=inter.view(torch.uint8),
                inter_sorted_shuffled_scale=scale.view(torch.uint8),
                w2_u8=base["w2_qt_shuf"].view(torch.uint8),
                w2_scale_u8=base["w2_scale_shuf"].view(torch.uint8),
                sorted_expert_ids=sei,
                cumsum_tensor=nv,
                sorted_token_ids=sti,
                sorted_weights=base["sorted_weights"],
                out=target,
                M_logical=tokens,
                max_sorted=inter.shape[0],
                NE=experts,
                D_HIDDEN=model_dim,
                D_INTER=inter_dim,
                topk=topk,
                BM=bm,
                BN=128,
                BK=128 if inter_dim == 384 else 256,
                SBM=bm,
                a_dtype="fp8",
                b_dtype="fp8",
                epilog=epilog,
            )
            if epilog == "reduce":
                _run_moe_reduction(target, out, tokens, topk, model_dim)
        else:
            flydsl_moe_stage2(
                inter_states=inter,
                w2=base["w2_qt_shuf"],
                sorted_token_ids=sti,
                sorted_expert_ids=sei,
                num_valid_ids=nv,
                out=out,
                topk=topk,
                tile_m=bm,
                tile_n=128,
                tile_k=128 if inter_dim == 384 else 256,
                a_dtype="fp8",
                b_dtype="fp8",
                out_dtype="bf16",
                mode=epilog,
                w2_scale=base["w2_scale_shuf"],
                a2_scale=scale,
                sorted_weights=base["sorted_weights"],
                use_async_copy=False,
            )
        torch.testing.assert_close(out, ref, rtol=0.03, atol=0.03)


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
