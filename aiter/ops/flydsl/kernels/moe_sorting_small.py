# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Route compaction for small MoE batches.

Short route lists use a single wave and first-occurrence expert order, without
atomics or a histogram. Larger batches use one CTA with an LDS histogram and
ascending expert order; route order within an expert may vary. GEMMs consume
explicit expert and packed route IDs. Other CTAs zero the accumulation buffer.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .kernels_common import atomic_add_i32
from .moe_sorting_kernel import _buf_iter, _dpp_intra_wave_prefix_sum, _gld, _gst
from .tensor_shim import _run_compiled


@functools.lru_cache(maxsize=128)
def _compile_small_sort(tokens, topk, experts, bm, zero_bytes):
    routes = tokens * topk
    assert 0 < routes <= 64
    assert zero_bytes % 16 == 0
    grid = 1 + (zero_bytes + 1023) // 1024

    @flyc.kernel(
        name=f"moe_sort_small_m{tokens}_k{topk}_e{experts}_b{bm}_z{zero_bytes}",
        known_block_size=[64, 1, 1],
    )
    def kernel(
        ids: fx.Tensor,
        weights: fx.Tensor,
        si: fx.Tensor,
        sw: fx.Tensor,
        se: fx.Tensor,
        nv: fx.Tensor,
        out: fx.Tensor,
    ):
        lane = gpu.thread_idx.x
        bid = gpu.block_idx.x
        if bid == fx.Int32(0):
            ids_it, weights_it = _buf_iter(ids), _buf_iter(weights)
            si_it, sw_it = _buf_iter(si), _buf_iter(sw)
            se_it, nv_it = _buf_iter(se), _buf_iter(nv)
            eid = _gld(ids_it, (lane < routes).select(lane, fx.Int32(0)))
            valid = (lane < routes) & (eid >= 0) & (eid < experts)
            eid = valid.select(eid, fx.Int32(-1))
            count, rank = fx.Int32(0), fx.Int32(0)
            first = lane
            for j in range_constexpr(routes):
                peer = fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(j * 4), eid))
                match = valid & (peer == eid)
                count = count + match.select(fx.Int32(1), fx.Int32(0))
                rank = rank + (match & (lane > j)).select(fx.Int32(1), fx.Int32(0))
                first = (match & (first > j)).select(fx.Int32(j), first)
            leader = valid & (rank == 0)
            padded = ((count + bm - 1) // bm) * bm
            contribution = leader.select(padded, fx.Int32(0))
            inclusive = _dpp_intra_wave_prefix_sum(contribution, lane, 64)
            total = fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(63 * 4), inclusive))
            own_start = inclusive - contribution
            start = fx.Int32(rocdl.ds_bpermute(T.i32, first * 4, own_start))
            if lane == fx.Int32(0):
                _gst(nv_it, total, fx.Int32(0))
                _gst(nv_it, fx.Int32(tokens), fx.Int32(1))
            if valid:
                packed = (lane // topk) | ((lane % topk) << 24)
                _gst(si_it, packed, start + rank)
                _gst(sw_it, _gld(weights_it, lane), start + rank)
            if leader:
                for b in range_constexpr((routes + bm - 1) // bm):
                    if fx.Int32(b * bm) < padded:
                        _gst(se_it, eid, start // bm + b)
                for p in range_constexpr(bm - 1):
                    pos = count + p
                    if pos < padded:
                        _gst(si_it, fx.Int32((topk << 24) | tokens), start + pos)
                        _gst(sw_it, fx.Float32(0), start + pos)
        else:
            idx = (bid - 1) * 64 + lane
            if idx < zero_bytes // 16:
                fx.ptr_store(
                    fx.Vector.filled(4, 0, fx.Int32), _buf_iter(out) + fx.Int64(idx * 4)
                )

    @flyc.jit
    def launch(
        ids: fx.Tensor,
        weights: fx.Tensor,
        si: fx.Tensor,
        sw: fx.Tensor,
        se: fx.Tensor,
        nv: fx.Tensor,
        out: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(ids, weights, si, sw, se, nv, out).launch(
            grid=(grid, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_cta_sort(tokens, topk, experts, bm, zero_bytes):
    routes = tokens * topk
    assert 0 < routes <= 512 and 0 < experts <= 256
    assert bm in (16, 32) and zero_bytes % 16 == 0
    grid = 1 + (zero_bytes + 4095) // 4096

    @fx.struct
    class Storage:
        counts: fx.Array[fx.Int32, 256]
        offsets: fx.Array[fx.Int32, 256]
        ranks: fx.Array[fx.Int32, 512]
        scan: fx.Array[fx.Int32, 4]

    @flyc.kernel(
        name=f"moe_sort_cta_m{tokens}_k{topk}_e{experts}_b{bm}_z{zero_bytes}_v1",
        known_block_size=[256, 1, 1],
    )
    def kernel(
        ids: fx.Tensor,
        weights: fx.Tensor,
        si: fx.Tensor,
        sw: fx.Tensor,
        se: fx.Tensor,
        nv: fx.Tensor,
        out: fx.Tensor,
    ):
        tid, bid = gpu.thread_idx.x, gpu.block_idx.x
        if bid == fx.Int32(0):
            storage = fx.SharedAllocator().allocate(Storage)
            counts = storage.counts.peek().view(fx.make_layout(256, 1))
            offsets = storage.offsets.peek().view(fx.make_layout(256, 1))
            ranks = storage.ranks.peek().view(fx.make_layout(512, 1))
            scan = storage.scan.peek().view(fx.make_layout(4, 1))
            ids_it, weights_it = _buf_iter(ids), _buf_iter(weights)
            si_it, sw_it = _buf_iter(si), _buf_iter(sw)
            se_it, nv_it = _buf_iter(se), _buf_iter(nv)
            counts[tid] = fx.Int32(0)
            gpu.barrier()
            for r in range_constexpr((routes + 255) // 256):
                route = tid + r * 256
                eid = _gld(ids_it, (route < routes).select(route, fx.Int32(0)))
                if (route < routes) & (eid >= 0) & (eid < experts):
                    ranks[route] = atomic_add_i32(counts, 1, eid, "workgroup")
            gpu.barrier()
            count = counts[tid]
            padded = ((count + bm - 1) // bm) * bm
            lane, wave = tid % 64, tid // 64
            inclusive = _dpp_intra_wave_prefix_sum(padded, lane, 64)
            if lane == fx.Int32(63):
                scan[wave] = inclusive
            gpu.barrier()
            cross, total = fx.Int32(0), fx.Int32(0)
            for w in range_constexpr(4):
                subtotal = scan[fx.Int32(w)]
                cross = cross + (wave > w).select(subtotal, fx.Int32(0))
                total = total + subtotal
            start = inclusive + cross - padded
            offsets[tid] = start
            if tid == fx.Int32(0):
                _gst(nv_it, total, fx.Int32(0))
                _gst(nv_it, fx.Int32(tokens), fx.Int32(1))
            # Expert lanes initialize only padding. Valid routes are scattered
            # separately, using the unique rank returned by the LDS atomic.
            for p in range(count, padded, fx.Int32(1)):
                _gst(si_it, fx.Int32((topk << 24) | tokens), start + p)
                _gst(sw_it, fx.Float32(0), start + p)
            for b in range(fx.Int32(0), padded // bm, fx.Int32(1)):
                _gst(se_it, tid, start // bm + b)
            gpu.barrier()
            for r in range_constexpr((routes + 255) // 256):
                route = tid + r * 256
                eid = _gld(ids_it, (route < routes).select(route, fx.Int32(0)))
                if (route < routes) & (eid >= 0) & (eid < experts):
                    pos = offsets[eid] + ranks[route]
                    packed = (route // topk) | ((route % topk) << 24)
                    _gst(si_it, packed, pos)
                    _gst(sw_it, _gld(weights_it, route), pos)
        else:
            idx = (bid - 1) * 256 + tid
            if idx < zero_bytes // 16:
                fx.ptr_store(
                    fx.Vector.filled(4, 0, fx.Int32), _buf_iter(out) + fx.Int64(idx * 4)
                )

    @flyc.jit
    def launch(
        ids: fx.Tensor,
        weights: fx.Tensor,
        si: fx.Tensor,
        sw: fx.Tensor,
        se: fx.Tensor,
        nv: fx.Tensor,
        out: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(ids, weights, si, sw, se, nv, out).launch(
            grid=(grid, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch


def small_moe_sort(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    dtype,
    block_size,
    *,
    accumulate=True,
    output=None,
):
    tokens, topk = topk_ids.shape
    routes = tokens * topk
    assert 0 < topk < 256 and 0 < routes <= 512
    assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
    assert topk_ids.is_contiguous() and topk_weights.is_contiguous()
    device = topk_ids.device
    # Each nonempty expert adds at most BM-1 padding rows. This bound also
    # covers repeated routes and experts spanning multiple sort blocks.
    blocks = (
        routes
        if routes <= 64
        else (routes + min(routes, num_experts) * (block_size - 1)) // block_size
    )
    capacity = blocks * block_size
    si = torch.empty(capacity, device=device, dtype=torch.int32)
    sw = torch.empty(capacity, device=device, dtype=torch.float32)
    se = torch.empty(blocks, device=device, dtype=torch.int32)
    nv = torch.empty(2, device=device, dtype=torch.int32)
    out = (
        (
            output
            if output is not None
            else torch.empty((tokens, model_dim), device=device, dtype=dtype)
        )
        if accumulate
        else torch.empty((0, 0), device=device, dtype=dtype)
    )
    # For 33-64 routes the CTA histogram also avoids the wave sort's quadratic
    # route comparisons. Preserve the wave path's wider expert/block support.
    use_wave = routes <= 64 and (
        routes <= 32 or num_experts > 256 or block_size not in (16, 32)
    )
    compile_sort = _compile_small_sort if use_wave else _compile_cta_sort
    launch = compile_sort(
        tokens, topk, num_experts, block_size, out.numel() * out.element_size()
    )
    _run_compiled(
        launch,
        topk_ids,
        topk_weights,
        si,
        sw,
        se,
        nv,
        out.reshape(-1).view(torch.int32),
        fx.Stream(torch.cuda.current_stream(device)),
    )
    return si, sw, se, nv, out
