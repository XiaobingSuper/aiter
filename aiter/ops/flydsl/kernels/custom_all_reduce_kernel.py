"""FlyDSL helpers for cross-GPU signal synchronization."""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _mlir_arith
from flydsl._mlir.dialects import llvm, rocdl, scf
from flydsl.expr import arith as ea, gpu
from flydsl.expr.arith import ArithValue
from flydsl.expr.meta import traced_op
from flydsl.expr.typing import T


def extui(self, target_type, *, loc=None):
    return ea.ExtUIOp(target_type, self, loc=loc).result


def extsi(self, target_type, *, loc=None):
    return ea.ExtSIOp(target_type, self, loc=loc).result


def trunci(self, target_type, *, loc=None):
    return ea.TruncIOp(target_type, self, loc=loc).result


ArithValue.extui = extui
ArithValue.extsi = extsi
ArithValue.trunci = trunci


@traced_op
def select_by_index(index_val, values):
    out = values[0]
    for i in range(1, len(values)):
        pred = _mlir_arith.CmpIOp(
            _mlir_arith.CmpIPredicate.eq,
            index_val,
            ea.constant(i, type=index_val.type),
        ).result
        out = _mlir_arith.SelectOp(pred, values[i], out).result
    return out


ea.select_by_index = select_by_index


@traced_op
def load_i32_uncached(addr_i64):
    value = llvm.InlineAsmOp(
        T.i32,
        [addr_i64],
        "global_load_dword $0, $1, off sc1",
        "=v,v",
        has_side_effects=True,
    ).result
    rocdl.s_waitcnt(0)
    rocdl.sched_barrier(0)
    return value


@traced_op
def store_i32_uncached_flush(addr_i64, val_i32):
    llvm.InlineAsmOp(None, [], "buffer_wbl2 sc0 sc1", "", has_side_effects=True)
    llvm.InlineAsmOp(
        None,
        [addr_i64, val_i32],
        "global_store_dword $0, $1, off sc0 sc1",
        "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)
    rocdl.sched_barrier(0)


@traced_op
def store_i32_uncached(addr_i64, val_i32):
    llvm.InlineAsmOp(
        None,
        [addr_i64, val_i32],
        "global_store_dword $0, $1, off sc0 sc1",
        "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)
    rocdl.sched_barrier(0)


@traced_op
def store_i32(addr_i64, val_i32):
    llvm.InlineAsmOp(
        None,
        [addr_i64, val_i32],
        "global_store_dword $0, $1, off",
        "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)
    rocdl.sched_barrier(0)


@traced_op
def load_v4i32(addr_i64):
    value = llvm.InlineAsmOp(
        T.i32x4,
        [addr_i64],
        "flat_load_dwordx4 $0, $1",
        "=v,v",
        has_side_effects=True,
    ).result
    rocdl.s_waitcnt(0)
    rocdl.sched_barrier(0)
    return value


@traced_op
def store_v4i32(addr_i64, v4i32_val):
    llvm.InlineAsmOp(
        None,
        [addr_i64, v4i32_val],
        "global_store_dwordx4 $0, $1, off",
        "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)
    rocdl.sched_barrier(0)


@traced_op
def store_v4i32_nt(addr_i64, v4i32_val):
    llvm.InlineAsmOp(
        None,
        [addr_i64, v4i32_val],
        "flat_store_dwordx4 $0, $1 nt",
        "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)
    rocdl.sched_barrier(0)


@traced_op
def load_device_ptr(array_base_i64, index):
    i64 = T.i64
    if hasattr(index, "type") and str(index.type) == "i32":
        index = _mlir_arith.ExtUIOp(i64, index).result
    elem_addr = array_base_i64 + index * ea.constant(8, type=i64)
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), elem_addr).result
    return llvm.LoadOp(i64, ptr).result


@traced_op
def invalidate_l1():
    llvm.InlineAsmOp(None, [], "buffer_inv sc1", "", has_side_effects=True)


_SG_START_OFF_B = 0
_SG_END_OFF_B = 2560
_SG_FLAG_OFF_B = 5120


def _u(value):
    return value.with_signedness(False)


def _signal_start_sync(
    *,
    lane_i32,
    rank_i32,
    bid_i32,
    self_sg_i64,
    sgs_i64,
    ngpus: int,
):
    i32, i64 = T.i32, T.i64

    flag_addr = (
        self_sg_i64
        + ea.constant(_SG_FLAG_OFF_B, type=i64)
        + bid_i32.extui(i64) * ea.constant(4, type=i64)
    )
    flag = load_i32_uncached(flag_addr) + ea.constant(1, type=i32)

    bid8 = bid_i32 * ea.constant(8, type=i32)
    lin_lane = bid8 + lane_i32
    start_wait_addr = (
        self_sg_i64
        + ea.constant(_SG_START_OFF_B, type=i64)
        + lin_lane.extui(i64) * ea.constant(4, type=i64)
    )
    lin_rank = bid8 + rank_i32
    start_rank_off = (
        ea.constant(_SG_START_OFF_B, type=i64)
        + lin_rank.extui(i64) * ea.constant(4, type=i64)
    )

    is_lane = _u(lane_i32) < ea.constant(ngpus, type=i32)
    if_op = scf.IfOp(is_lane, results_=[], has_else=False)
    with ir.InsertionPoint(if_op.then_block):
        peer_sg = ea.select_by_index(lane_i32, sgs_i64)
        store_i32_uncached_flush(peer_sg + start_rank_off, flag)
        init_cur = load_i32_uncached(start_wait_addr)
        loop = scf.WhileOp([i32], [init_cur])
        before = ir.Block.create_at_start(loop.before, [i32])
        after = ir.Block.create_at_start(loop.after, [i32])
        with ir.InsertionPoint(before):
            cur = before.arguments[0]
            need_wait = _u(cur) < flag
            scf.ConditionOp(need_wait, [cur])
        with ir.InsertionPoint(after):
            scf.YieldOp([load_i32_uncached(start_wait_addr)])
        scf.YieldOp([])

    gpu.barrier()
    is_t0 = lane_i32 == ea.constant(0, type=i32)
    if_t0 = scf.IfOp(is_t0, results_=[], has_else=False)
    with ir.InsertionPoint(if_t0.then_block):
        store_i32(flag_addr, flag)
        scf.YieldOp([])
    return flag_addr


def _signal_end_sync(
    *,
    lane_i32,
    rank_i32,
    bid_i32,
    self_sg_i64,
    sgs_i64,
    ngpus: int,
    need_wbl2: bool = False,
):
    i32, i64 = T.i32, T.i64

    gpu.barrier()
    flag_addr = (
        self_sg_i64
        + ea.constant(_SG_FLAG_OFF_B, type=i64)
        + bid_i32.extui(i64) * ea.constant(4, type=i64)
    )
    flag = load_i32_uncached(flag_addr) + ea.constant(1, type=i32)

    bid8 = bid_i32 * ea.constant(8, type=i32)
    lin_lane = bid8 + lane_i32
    end_wait_addr = (
        self_sg_i64
        + ea.constant(_SG_END_OFF_B, type=i64)
        + lin_lane.extui(i64) * ea.constant(4, type=i64)
    )
    lin_rank = bid8 + rank_i32
    end_rank_off = (
        ea.constant(_SG_END_OFF_B, type=i64)
        + lin_rank.extui(i64) * ea.constant(4, type=i64)
    )

    is_lane = _u(lane_i32) < ea.constant(ngpus, type=i32)
    if_op = scf.IfOp(is_lane, results_=[], has_else=False)
    with ir.InsertionPoint(if_op.then_block):
        peer_sg = ea.select_by_index(lane_i32, sgs_i64)
        if need_wbl2:
            store_i32_uncached_flush(peer_sg + end_rank_off, flag)
        else:
            store_i32_uncached(peer_sg + end_rank_off, flag)
        init_cur = load_i32_uncached(end_wait_addr)
        loop = scf.WhileOp([i32], [init_cur])
        before = ir.Block.create_at_start(loop.before, [i32])
        after = ir.Block.create_at_start(loop.after, [i32])
        with ir.InsertionPoint(before):
            cur = before.arguments[0]
            need_wait = _u(cur) < flag
            scf.ConditionOp(need_wait, [cur])
        with ir.InsertionPoint(after):
            nxt = load_i32_uncached(end_wait_addr)
            invalidate_l1()
            scf.YieldOp([nxt])
        scf.YieldOp([])

    gpu.barrier()
    is_t0 = lane_i32 == ea.constant(0, type=i32)
    if_t0 = scf.IfOp(is_t0, results_=[], has_else=False)
    with ir.InsertionPoint(if_t0.then_block):
        store_i32(flag_addr, flag)
        scf.YieldOp([])
