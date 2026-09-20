# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Validate the BF16 MoE GEMM1 LDS layout with FP4, INT4, and BF16 weights.

Run on gfx950:
    python op_tests/test_flydsl_moe_layouts.py
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx


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
