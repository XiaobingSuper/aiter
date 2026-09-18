# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""BM16 MXFP8 MoE: validate the scale payload and both GEMM2 layouts.

Run on gfx950:
    pytest op_tests/flydsl_tests/test_flydsl_moe_mxfp8.py -q
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


@pytest.mark.parametrize(
    "tokens,experts,hidden,topk,pattern",
    [
        (1, 129, 6144, 5, "random"),
        (3, 129, 6144, 5, "shared"),
        (8, 129, 6144, 5, "shared"),
        (8, 129, 6144, 5, "same"),
        (2, 129, 6144, 5, "invalid"),
        (16, 129, 6144, 5, "random"),
        (32, 129, 6144, 5, "shared"),
        (64, 129, 6144, 5, "shared"),
        (64, 129, 6144, 5, "same"),
        (17, 129, 6144, 5, "invalid"),
        (64, 129, 6144, 5, "all_invalid"),
        (103, 129, 6144, 5, "topk"),  # Above the 512-route limit.
        (8, 33, 7168, 8, "topk"),
        (8, 56, 3584, 16, "topk"),
        (64, 128, 3072, 4, "topk"),
        (64, 256, 3072, 8, "topk"),
        (32, 256, 4096, 6, "topk"),
        (8, 896, 3584, 16, "topk"),  # Above the 256-expert limit.
        (8, 7, 1024, 5, "topk"),  # No generated instance: general-sort fallback.
    ],
)
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("output_aux", [False, True])
def test_adaptive_sort_graph_replay(
    tokens, experts, hidden, topk, pattern, accumulate, output_aux, monkeypatch
):
    import importlib

    fm = importlib.import_module("aiter.fused_moe")
    monkeypatch.setattr(fm, "_USE_CK_MOE_SORTING", False)
    monkeypatch.setattr(fm, "_USE_FLYDSL_MOE_SORTING", False)
    monkeypatch.setattr(fm, "_MOE_SORT_BACKEND", "auto")

    bm = 16
    if output_aux and not fm._mxfp4_aux_instance_supported(
        experts, topk, hidden, bm, accumulate
    ):
        pytest.skip("aux outputs require a generated instance")
    ids = torch.empty((tokens, topk), device="cuda", dtype=torch.int32)
    weights = torch.empty((tokens, topk), device="cuda", dtype=torch.float32)
    output = torch.full((tokens, hidden), 42, device="cuda", dtype=dtypes.bf16)

    def fill(seed):
        torch.manual_seed(seed)
        ids.copy_(torch.randint(experts, ids.shape, device="cuda", dtype=torch.int32))
        if pattern == "shared":
            ids[:, -1] = experts - 1
        elif pattern == "same":
            ids.fill_(experts - 1)
        elif pattern == "invalid":
            ids[0].fill_(-1)
            ids[-1, -1] = experts
        elif pattern == "all_invalid":
            ids.fill_(-1)
        elif pattern == "topk":
            ids.copy_(torch.rand((tokens, experts), device="cuda").topk(topk).indices)
        weights.normal_()
        output.fill_(42)

    def run():
        result = fm.moe_sorting(
            ids,
            weights,
            experts,
            hidden,
            dtypes.bf16,
            bm,
            accumulate=accumulate,
            output_aux=output_aux,
            output=output,
        )
        assert len(result) == (7 if output_aux else 5)
        return result[:5]

    fill(3)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        si, sw, se, nv, out = run()
    for seed in (4, 5):
        fill(seed)
        graph.replay()
        torch.cuda.synchronize()
        ids_cpu, weights_cpu = ids.cpu(), weights.cpu()
        expected = {}
        for t in range(tokens):
            for k in range(topk):
                e = int(ids_cpu[t, k])
                if 0 <= e < experts:
                    expected.setdefault(e, []).append(
                        ((k << 24) | t, float(weights_cpu[t, k]))
                    )
        size = int(nv[0])
        assert int(nv[1]) == tokens
        assert size == sum((len(v) + bm - 1) // bm * bm for v in expected.values())
        actual = {}
        si_cpu, sw_cpu, se_cpu = si.cpu(), sw.cpu(), se.cpu()
        for p in range(size):
            packed = int(si_cpu[p])
            if (packed & 0xFFFFFF) < tokens:
                actual.setdefault(int(se_cpu[p // bm]), []).append(
                    (packed, float(sw_cpu[p]))
                )
            else:
                assert (packed & 0xFFFFFF) == tokens
                assert float(sw_cpu[p]) == 0
        assert {e: sorted(v) for e, v in actual.items()} == {
            e: sorted(v) for e, v in expected.items()
        }
        if accumulate:
            assert out.data_ptr() == output.data_ptr()
            assert torch.count_nonzero(out) == 0
        else:
            assert out.numel() == 0
            assert torch.all(output == 42)


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
                g2_bf16_lds=False if k_wave == 2 else None,
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
