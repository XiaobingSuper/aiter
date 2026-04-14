# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Low-level FlyDSL compile for single-kernel HGEMM + INT4 QuickReduce."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _mlir_arith
from flydsl._mlir.dialects import fly, gpu as _mlir_gpu, llvm, memref, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import fly_values
from flydsl.expr import arith, gpu, range_constexpr, rocdl, vector
from flydsl.expr.buffer_ops import _unwrap_value
from flydsl.expr.typing import Int32, Int64, T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .custom_all_reduce_kernel import (
    load_device_ptr,
    load_i32_uncached,
    select_by_index,
    store_i32_uncached,
    store_i32_uncached_flush,
)
from .tensor_shim import GTensor, STensor, _to_raw, get_dtype_in_kernel

SPLIT_K_COUNTER_MAX_LEN = 128
SPLIT_K_SIGNAL_STATE_COUNT = 3


def swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)


class WmmaHalfBase(ABC):
    @abstractmethod
    def __init__(self, dtype: str):
        pass

    @abstractmethod
    def __call__(self, a_frag, b_frag, c_frag):
        pass


class WmmaHalf_m16n16k16(WmmaHalfBase):
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    WMMA_A_FRAG_VALUES = 4
    WMMA_B_FRAG_VALUES = 4
    WMMA_C_FRAG_VALUES = 4

    def __init__(self, dtype: str):
        self.dtype = dtype

    def __call__(self, a_frag, b_frag, c_frag):
        if self.dtype == "bf16":
            a_frag_vi16 = vector.bitcast(T.vec(self.WMMA_A_FRAG_VALUES, T.i16), a_frag)
            b_frag_vi16 = vector.bitcast(T.vec(self.WMMA_B_FRAG_VALUES, T.i16), b_frag)
            return rocdl.mfma_f32_16x16x16bf16_1k(
                T.f32x4, [a_frag_vi16, b_frag_vi16, c_frag, 0, 0, 0]
            )
        return rocdl.mfma_f32_16x16x16f16(
            T.vec(self.WMMA_C_FRAG_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0]
        )


class WmmaHalf_m16n16k32(WmmaHalfBase):
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 32
    WMMA_A_FRAG_VALUES = 8
    WMMA_B_FRAG_VALUES = 8
    WMMA_C_FRAG_VALUES = 4

    def __init__(self, dtype: str):
        self.dtype = dtype

    def __call__(self, a_frag, b_frag, c_frag):
        if self.dtype == "bf16":
            return rocdl.mfma_f32_16x16x32_bf16(
                T.vec(self.WMMA_C_FRAG_VALUES, T.f32), a_frag, b_frag, c_frag, 0, 0, 0
            ).res
        return rocdl.mfma_f32_16x16x32_f16(
            T.vec(self.WMMA_C_FRAG_VALUES, T.f32), a_frag, b_frag, c_frag, 0, 0, 0
        ).res


class OnlineScheduler:
    def __init__(self, total_signals: int, init_count: int = 0):
        self.total_signals = total_signals
        self.current_signal_id = init_count
        self.remaining = init_count

    def release(self, count: int):
        count = min(count, self.total_signals - self.current_signal_id)
        self.current_signal_id += count
        self.remaining += count

    def consume(self, count: int):
        count = min(count, self.remaining)
        self.remaining -= count
        return count


@functools.lru_cache(maxsize=1024)
def compile_hgemm_qr_kernel(
    world_size: int,
    dtype: str,
    n: int,
    k: int,
    TILE_M: int = 128,
    TILE_N: int = 128,
    TILE_K: int = 64,
    SPLIT_K: int = 1,
    BLOCK_M_WARPS: int = 1,
    BLOCK_N_WARPS: int = 4,
    B_PRE_SHUFFLE: bool = False,
    B_TO_LDS: bool = False,
):
    IS_SPLIT_K = SPLIT_K > 1
    BLOCK_K = TILE_K
    assert (k % SPLIT_K == 0) and (k // SPLIT_K >= 1)
    ks = k // SPLIT_K
    assert (ks % BLOCK_K == 0) and (ks // BLOCK_K >= 1)
    assert BLOCK_K >= 32
    if B_PRE_SHUFFLE:
        B_TO_LDS = False
        assert B_TO_LDS is False

    GPU_ARCH = get_rocm_arch()
    if GPU_ARCH == "gfx942":
        WMMA_IMPL = WmmaHalf_m16n16k16(dtype)
        DMA_BYTES = 4
        MFMA_PER_WARP_K = 2
        ASYNC_COPY = False
    else:
        WMMA_IMPL = WmmaHalf_m16n16k32(dtype)
        DMA_BYTES = 16
        MFMA_PER_WARP_K = 1
        ASYNC_COPY = True

    WARP_SIZE = 64
    DTYPE_BYTES = 2
    LDG_VEC_SIZE = 8
    STAGES = 2

    WMMA_M = WMMA_IMPL.WMMA_M
    WMMA_N = WMMA_IMPL.WMMA_N
    WMMA_K = WMMA_IMPL.WMMA_K
    WMMA_A_FRAG_VALUES = WMMA_IMPL.WMMA_A_FRAG_VALUES
    WMMA_B_FRAG_VALUES = WMMA_IMPL.WMMA_B_FRAG_VALUES
    WMMA_C_FRAG_VALUES = WMMA_IMPL.WMMA_C_FRAG_VALUES
    WARP_ATOM_M = WMMA_M
    WARP_ATOM_N = WMMA_N
    WARP_ATOM_K = WMMA_K * MFMA_PER_WARP_K
    BLOCK_K_LOOPS = ks // BLOCK_K
    WARP_K_STEPS = BLOCK_K // WARP_ATOM_K
    assert (BLOCK_K % WARP_ATOM_K == 0) and (WARP_K_STEPS >= 1)
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE
    WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WARP_ATOM_M
    WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WARP_ATOM_N
    assert (WARP_M_STEPS >= 1) and (WARP_N_STEPS >= 1)
    assert TILE_M % (BLOCK_M_WARPS * WARP_ATOM_M) == 0
    assert TILE_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0
    WARP_M = WARP_M_STEPS * WARP_ATOM_M
    WARP_N = WARP_N_STEPS * WARP_ATOM_N
    BLOCK_M = BLOCK_M_WARPS * WARP_M
    BLOCK_N = BLOCK_N_WARPS * WARP_N
    assert (n >= BLOCK_N) and (n % BLOCK_N == 0)
    BLOCK_MK_SIZE = BLOCK_M * BLOCK_K
    BLOCK_NK_SIZE = BLOCK_N * BLOCK_K
    BLOCK_MN_SIZE = BLOCK_M * BLOCK_N
    LDG_A_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_B_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_C_X_THREADS = BLOCK_N // LDG_VEC_SIZE
    BLOCK_VECS = LDG_VEC_SIZE * BLOCK_THREADS
    LDG_REG_A_COUNT = BLOCK_MK_SIZE // BLOCK_VECS
    LDG_REG_B_COUNT = BLOCK_NK_SIZE // BLOCK_VECS
    LDG_REG_C_COUNT = BLOCK_MN_SIZE // BLOCK_VECS
    assert (LDG_REG_A_COUNT >= 1) and (LDG_REG_B_COUNT >= 1) and (LDG_REG_C_COUNT >= 1)
    assert BLOCK_MK_SIZE % BLOCK_VECS == 0
    assert BLOCK_NK_SIZE % BLOCK_VECS == 0
    assert BLOCK_MN_SIZE % BLOCK_VECS == 0
    assert world_size in (2, 4, 8)
    assert SPLIT_K == 1
    assert BLOCK_THREADS == 256
    assert LDG_REG_C_COUNT <= 8
    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES

    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem")
    smem_a_offset = allocator._align(allocator.ptr, 16)
    AS_BYTES = STAGES * BLOCK_M * BLOCK_K * DTYPE_BYTES
    AS_BYTES = max(AS_BYTES, BLOCK_M * BLOCK_N * DTYPE_BYTES)
    allocator.ptr = smem_a_offset + AS_BYTES
    SMEM_USE = AS_BYTES
    if B_TO_LDS:
        smem_b_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = smem_b_offset + STAGES * BLOCK_N * BLOCK_K * DTYPE_BYTES
        SMEM_USE += STAGES * BLOCK_N * BLOCK_K * DTYPE_BYTES
    assert SMEM_USE <= 163840
    LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
    LDG_A_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    LDG_REG_A_COUNT_AS = BLOCK_MK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
    LDG_B_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    LDG_REG_B_COUNT_AS = BLOCK_NK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS

    QR_RANK_ATOMS = 8 // world_size
    QR_RANK_TILE_STRIDE = 1152
    QR_SCALE_OFFSET = 1024
    QR_RANK_TRANS_BYTES = QR_RANK_TILE_STRIDE * QR_RANK_ATOMS
    QR_TRANS_TILE_BYTES = QR_RANK_TRANS_BYTES * world_size

    KERNEL_NAME = f"hgemm_qr_int4_ws{world_size}_{dtype}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_S{STAGES}TN"
    KERNEL_NAME += "_NA" if not ASYNC_COPY else "_AS"
    if B_PRE_SHUFFLE:
        KERNEL_NAME += "_BP"
    if B_TO_LDS:
        KERNEL_NAME += "_BS"

    @flyc.kernel
    def hgemm_qr_kernel(
        rank: Int32,
        buffer_ptrs: Int64,
        flags_phase_bytes: Int64,
        data_offset: Int64,
        phase_bytes: Int64,
        C: fx.Tensor,
        A: fx.Tensor,
        B: fx.Tensor,
        m: fx.Int32,
        COUNTER: fx.Tensor,
        signal_state: fx.Int32,
    ):
        dtype_ = get_dtype_in_kernel(dtype)
        _ptr_type = ir.Type.parse("!llvm.ptr<1>")
        _i64_type = T.i64
        c_zero_d = arith.constant(0.0, type=dtype_)
        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))

        A_ = GTensor(A, dtype=dtype_, shape=(-1, k))
        B_ = GTensor(B, dtype=dtype_, shape=(n, k))
        C_ = GTensor(C, dtype=dtype_, shape=(-1, n))
        base_ptr = allocator.get_base()
        smem_a_ptr = SmemPtr(
            base_ptr, smem_a_offset, dtype_, shape=(STAGES * BLOCK_M * BLOCK_K,)
        )
        as_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, BLOCK_M, BLOCK_K))
        if B_TO_LDS:
            smem_b_ptr = SmemPtr(
                base_ptr, smem_b_offset, dtype_, shape=(STAGES * BLOCK_N * BLOCK_K,)
            )
            bs_ = STensor(smem_b_ptr, dtype_, shape=(STAGES, BLOCK_N, BLOCK_K))
        smem_c_ptr = SmemPtr(
            base_ptr, smem_a_offset, dtype_, shape=(BLOCK_M * BLOCK_N,)
        )
        cs_ = STensor(smem_c_ptr, dtype_, shape=(BLOCK_M, BLOCK_N))
        if B_PRE_SHUFFLE:
            SHUFFLED_B_ = GTensor(
                B,
                dtype=dtype_,
                shape=(
                    n // WARP_ATOM_N,
                    k // WARP_ATOM_K,
                    WARP_ATOM_K // LDG_VEC_SIZE,
                    WARP_ATOM_N,
                    LDG_VEC_SIZE,
                ),
            )
        if IS_SPLIT_K:
            COUNTER_ = GTensor(COUNTER, dtype=T.i32, shape=(-1,))

        tid = fx.Int32(fx.thread_idx.x)
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE
        block_m_idx = fx.block_idx.x
        block_n_idx = fx.block_idx.y
        ks_idx = fx.Index(fx.block_idx.z)
        ks_begin = arith.index_cast(T.i32, ks_idx * ks)
        rank_i32 = _unwrap_value(rank)
        buffer_ptrs_i64 = _unwrap_value(buffer_ptrs)
        flags_phase_bytes_i64 = _unwrap_value(flags_phase_bytes)
        data_offset_i64 = _unwrap_value(data_offset)
        phase_bytes_i64 = _unwrap_value(phase_bytes)
        bid_linear = fx.block_idx.x * fx.Int32(n // BLOCK_N) + fx.block_idx.y
        bid_i32 = arith.index_cast(T.i32, fx.Index(bid_linear))
        bid_i64 = arith.index_cast(T.i64, fx.Index(bid_linear))
        buffer_ptrs_arr = [
            load_device_ptr(buffer_ptrs_i64, arith.constant(i, type=T.i32))
            for i in range(8)
        ]
        self_buffer_ptr = select_by_index(rank_i32, buffer_ptrs_arr)
        counter_idx = (
            fx.Int32(signal_state * SPLIT_K_COUNTER_MAX_LEN)
            + fx.block_idx.x * fx.Int32(n // BLOCK_N)
            + fx.block_idx.y
        )

        m_offset = fx.Index(block_m_idx * BLOCK_M)
        n_offset = fx.Index(block_n_idx * BLOCK_N)
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)

        warp_m_idx = wid // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid % BLOCK_N_WARPS * WARP_N
        ldmatrix_a_m_idx = w_tid % WMMA_M
        ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K
        ldmatrix_b_n_idx = w_tid % WMMA_N
        ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
        A_FRAGS_LEN = WARP_K_STEPS * WARP_M_STEPS
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
        c_frags = [acc_init] * C_FRAGS_LEN

        def zero_c():
            cond_ks0 = arith.cmpi(arith.CmpIPredicate.eq, ks_idx, fx.Index(0))
            cond_ks0_if = scf.IfOp(cond_ks0, results_=[], has_else=False)
            with ir.InsertionPoint(cond_ks0_if.then_block):
                zero_vec = vector.broadcast(T.vec(LDG_VEC_SIZE, dtype_), c_zero_d)
                for i in range_constexpr(LDG_REG_C_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    m_local_idx = global_tid // LDG_C_X_THREADS
                    n_local_idx = global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE
                    row_idx = m_offset + fx.Index(m_local_idx)
                    cond_boundary = arith.cmpi(
                        arith.CmpIPredicate.ult, row_idx, fx.Index(m)
                    )
                    cond_boundary_if = scf.IfOp(
                        cond_boundary, results_=[], has_else=False
                    )
                    with ir.InsertionPoint(cond_boundary_if.then_block):
                        C_.vec_store(
                            (row_idx, n_offset + n_local_idx), zero_vec, LDG_VEC_SIZE
                        )
                        scf.YieldOp([])
                scf.YieldOp([])
            rocdl.sched_barrier(0)
            gpu.barrier()

            cond_ks0_if = scf.IfOp(cond_ks0, results_=[], has_else=False)
            with ir.InsertionPoint(cond_ks0_if.then_block):
                is_t0_cond = arith.cmpi(
                    arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0)
                )
                is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
                with ir.InsertionPoint(is_t0_cond_if.then_block):
                    counter_base_ptr = fly.extract_aligned_pointer_as_index(
                        _ptr_type, fly_values(COUNTER)[0]
                    )
                    counter_base_ptr = llvm.PtrToIntOp(
                        _i64_type, counter_base_ptr
                    ).result
                    counter_byte_offset = arith.index_cast(
                        T.i64, fx.Index(counter_idx) * fx.Index(4)
                    )
                    counter_ptr = llvm.AddOp(
                        counter_base_ptr,
                        counter_byte_offset,
                        llvm.IntegerOverflowFlags(0),
                    ).result
                    counter_ptr = llvm.IntToPtrOp(_ptr_type, counter_ptr).result
                    counter_ptr_v = (
                        counter_ptr._value
                        if hasattr(counter_ptr, "_value")
                        else counter_ptr
                    )
                    llvm.InlineAsmOp(
                        None, [], "buffer_wbl2 sc0 sc1", "", has_side_effects=True
                    )
                    llvm.InlineAsmOp(
                        None,
                        [counter_ptr_v, arith.constant(1, type=T.i32)],
                        "global_store_dword $0, $1, off sc0 sc1",
                        "v,v",
                        has_side_effects=True,
                    )
                    rocdl.s_waitcnt(0)
                    scf.YieldOp([])
                scf.YieldOp([])
            rocdl.sched_barrier(0)
            gpu.barrier()

            cond_ks0_if = scf.IfOp(cond_ks0, results_=[], has_else=False)
            with ir.InsertionPoint(cond_ks0_if.then_block):
                clean_cond = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    fx.Index(tid),
                    fx.Index(SPLIT_K_COUNTER_MAX_LEN),
                )
                clean_cond_if = scf.IfOp(clean_cond, results_=[], has_else=False)
                with ir.InsertionPoint(clean_cond_if.then_block):
                    clean_counter_idx = fx.Int32(
                        (
                            (signal_state + SPLIT_K_SIGNAL_STATE_COUNT - 1)
                            % SPLIT_K_SIGNAL_STATE_COUNT
                        )
                        * SPLIT_K_COUNTER_MAX_LEN
                    ) + fx.Index(tid)
                    COUNTER_[fx.Index(clean_counter_idx)] = arith.constant(
                        0, type=T.i32
                    )
                    scf.YieldOp([])
                scf.YieldOp([])
            rocdl.sched_barrier(0)
            gpu.barrier()

        def split_k_barrier():
            init_cur = arith.constant(0, type=T.i32)
            w = scf.WhileOp([T.i32], [init_cur])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                cur = before.arguments[0]
                need_wait = arith.CmpIOp(
                    arith.CmpIPredicate.eq, cur, arith.constant(0, type=T.i32)
                ).result
                scf.ConditionOp(need_wait, [cur])
            with ir.InsertionPoint(after):
                counter_base_ptr = fly.extract_aligned_pointer_as_index(
                    _ptr_type, fly_values(COUNTER)[0]
                )
                counter_base_ptr = llvm.PtrToIntOp(_i64_type, counter_base_ptr).result
                counter_byte_offset = arith.index_cast(
                    T.i64, fx.Index(counter_idx) * fx.Index(4)
                )
                counter_ptr = llvm.AddOp(
                    counter_base_ptr,
                    counter_byte_offset,
                    llvm.IntegerOverflowFlags(0),
                ).result
                counter_ptr = llvm.IntToPtrOp(_ptr_type, counter_ptr).result
                counter_ptr_v = (
                    counter_ptr._value
                    if hasattr(counter_ptr, "_value")
                    else counter_ptr
                )
                data = llvm.InlineAsmOp(
                    T.i32,
                    [counter_ptr_v],
                    "global_load_dword $0, $1, off sc1",
                    "=v,v",
                    has_side_effects=True,
                ).result
                rocdl.s_waitcnt(0)
                scf.YieldOp([data])
            gpu.barrier()

        def ldg_a(k_offset):
            vecs = []
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                row_idx = m_offset + fx.Index(m_local_idx)
                safe_row_idx = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                    row_idx,
                    fx.Index(0),
                )
                col_idx = fx.Index(k_offset + k_local_idx)
                vecs.append(A_.vec_load((safe_row_idx, col_idx), LDG_VEC_SIZE))
            return vecs

        def sts_a(vecs, lds_stage):
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                col_in_bytes = swizzle_xor16(
                    m_local_idx, k_local_idx * DTYPE_BYTES, k_blocks16
                )
                as_.vec_store(
                    (fx.Index(lds_stage), m_local_idx, col_in_bytes // DTYPE_BYTES),
                    vecs[i],
                    LDG_VEC_SIZE,
                )

        def ldg_b(k_offset):
            vecs = []
            for i in range_constexpr(LDG_REG_B_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                n_local_idx = global_tid // LDG_B_X_THREADS
                k_local_idx = global_tid % LDG_B_X_THREADS * LDG_VEC_SIZE
                row_idx = n_offset + fx.Index(n_local_idx)
                safe_row_idx = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(n)),
                    row_idx,
                    fx.Index(0),
                )
                col_idx = fx.Index(k_offset + k_local_idx)
                vecs.append(B_.vec_load((safe_row_idx, col_idx), LDG_VEC_SIZE))
            return vecs

        def sts_b(vecs, lds_stage):
            for i in range_constexpr(LDG_REG_B_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                n_local_idx = global_tid // LDG_B_X_THREADS
                k_local_idx = global_tid % LDG_B_X_THREADS * LDG_VEC_SIZE
                col_in_bytes = swizzle_xor16(
                    n_local_idx, k_local_idx * DTYPE_BYTES, k_blocks16
                )
                bs_.vec_store(
                    (fx.Index(lds_stage), n_local_idx, col_in_bytes // DTYPE_BYTES),
                    vecs[i],
                    LDG_VEC_SIZE,
                )

        def ldg_sts_a_async(k_offset, lds_stage):
            for i in range_constexpr(LDG_REG_A_COUNT_AS):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS_AS
                k_local_idx = global_tid % LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
                col_in_bytes = swizzle_xor16(
                    m_local_idx, k_local_idx * DTYPE_BYTES, k_blocks16
                )
                row_idx = m_offset + fx.Index(m_local_idx)
                safe_row_idx = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                    row_idx,
                    fx.Index(0),
                )
                col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
                global_offset = A_.linear_offset((safe_row_idx, col_idx)) * DTYPE_BYTES
                global_offset = arith.index_cast(T.i32, global_offset)
                lds_offset = (
                    as_.linear_offset((fx.Index(lds_stage), m_local_idx, k_local_idx))
                    * DTYPE_BYTES
                )
                lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                lds_addr = (
                    memref.extract_aligned_pointer_as_index(as_.memptr) + lds_offset
                )
                lds_addr_ = rocdl.readfirstlane(
                    T.i64, arith.index_cast(T.i64, lds_addr)
                )
                lds_ptr = llvm.inttoptr(lds_ptr_type, lds_addr_)
                rocdl.raw_ptr_buffer_load_lds(
                    A_.rsrc,
                    lds_ptr,
                    arith.constant(DMA_BYTES, type=T.i32),
                    global_offset,
                    arith.constant(0, type=T.i32),
                    arith.constant(0, type=T.i32),
                    arith.constant(1, type=T.i32),
                )

        def ldg_sts_b_async(k_offset, lds_stage):
            for i in range_constexpr(LDG_REG_B_COUNT_AS):
                global_tid = BLOCK_THREADS * i + tid
                n_local_idx = global_tid // LDG_B_X_THREADS_AS
                k_local_idx = global_tid % LDG_B_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
                col_in_bytes = swizzle_xor16(
                    n_local_idx, k_local_idx * DTYPE_BYTES, k_blocks16
                )
                row_idx = n_offset + fx.Index(n_local_idx)
                safe_row_idx = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(n)),
                    row_idx,
                    fx.Index(0),
                )
                col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
                global_offset = B_.linear_offset((safe_row_idx, col_idx)) * DTYPE_BYTES
                global_offset = arith.index_cast(T.i32, global_offset)
                lds_offset = (
                    bs_.linear_offset((fx.Index(lds_stage), n_local_idx, k_local_idx))
                    * DTYPE_BYTES
                )
                lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                lds_addr = (
                    memref.extract_aligned_pointer_as_index(bs_.memptr) + lds_offset
                )
                lds_addr_ = rocdl.readfirstlane(
                    T.i64, arith.index_cast(T.i64, lds_addr)
                )
                lds_ptr = llvm.inttoptr(lds_ptr_type, lds_addr_)
                rocdl.raw_ptr_buffer_load_lds(
                    B_.rsrc,
                    lds_ptr,
                    arith.constant(DMA_BYTES, type=T.i32),
                    global_offset,
                    arith.constant(0, type=T.i32),
                    arith.constant(0, type=T.i32),
                    arith.constant(1, type=T.i32),
                )

        def lds_matrix_a(lds_stage):
            s = fx.Index(lds_stage)
            a_frags = [0] * (WARP_K_STEPS * WARP_M_STEPS)
            for ii in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                for kk in range_constexpr(WARP_K_STEPS):
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    row = warp_atom_m_idx + ldmatrix_a_m_idx
                    col_in_bytes = (
                        warp_atom_k_idx + ldmatrix_a_k_vec_idx
                    ) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    a_frags[kk * WARP_M_STEPS + ii] = as_.vec_load(
                        (s, row, col_in_bytes // DTYPE_BYTES),
                        WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K,
                    )
            return a_frags

        def lds_matrix_b(lds_stage):
            s = fx.Index(lds_stage)
            b_frags = [0] * (WARP_K_STEPS * WARP_N_STEPS)
            for ii in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                for kk in range_constexpr(WARP_K_STEPS):
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    row = warp_atom_n_idx + ldmatrix_b_n_idx
                    col_in_bytes = (
                        warp_atom_k_idx + ldmatrix_b_k_vec_idx
                    ) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    b_frags[kk * WARP_N_STEPS + ii] = bs_.vec_load(
                        (s, row, col_in_bytes // DTYPE_BYTES),
                        WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
                    )
            return b_frags

        def ldg_matrix_b(k_offset):
            vecs = []
            b_n_intra_base = ldmatrix_b_n_idx
            b_k_intra_vec = ldmatrix_b_k_vec_idx // LDG_VEC_SIZE
            b_n0_base = n_offset // WARP_ATOM_N + warp_n_idx // WARP_ATOM_N
            b_k0_base = k_offset // WARP_ATOM_K
            for kk in range_constexpr(WARP_K_STEPS):
                b_k0 = b_k0_base + kk
                for ii in range_constexpr(WARP_N_STEPS):
                    b_n0 = b_n0_base + ii
                    if not B_PRE_SHUFFLE:
                        warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                        warp_atom_k_idx = kk * WARP_ATOM_K
                        n_idx = n_offset + warp_atom_n_idx + ldmatrix_b_n_idx
                        k_idx = k_offset + warp_atom_k_idx + ldmatrix_b_k_vec_idx
                        vecs.append(
                            B_.vec_load(
                                (n_idx, k_idx), WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
                            )
                        )
                    else:
                        vecs.append(
                            SHUFFLED_B_.vec_load(
                                (b_n0, b_k0, b_k_intra_vec, b_n_intra_base, 0),
                                LDG_VEC_SIZE,
                            )
                        )
            return vecs

        def block_mma_sync(a_frags, b_frags, c_frags):
            c_frags_new = [cx for cx in c_frags]
            for kk in range_constexpr(WARP_K_STEPS):
                for ii in range_constexpr(WARP_M_STEPS):
                    a_frag = a_frags[kk * WARP_M_STEPS + ii]
                    for jj in range_constexpr(WARP_N_STEPS):
                        b_frag = b_frags[kk * WARP_N_STEPS + jj]
                        if MFMA_PER_WARP_K == 2:
                            a_i64x2 = vector.bitcast(T.i64x2, a_frag)
                            a0_i64 = vector.extract(
                                a_i64x2, static_position=[0], dynamic_position=[]
                            )
                            a1_i64 = vector.extract(
                                a_i64x2, static_position=[1], dynamic_position=[]
                            )
                            a_v0 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [a0_i64])
                            )
                            a_v1 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [a1_i64])
                            )
                            b_i64x2 = vector.bitcast(T.i64x2, b_frag)
                            b0_i64 = vector.extract(
                                b_i64x2, static_position=[0], dynamic_position=[]
                            )
                            b1_i64 = vector.extract(
                                b_i64x2, static_position=[1], dynamic_position=[]
                            )
                            b_v0 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [b0_i64])
                            )
                            b_v1 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [b1_i64])
                            )
                            c_idx = ii * WARP_N_STEPS + jj
                            acc_in = c_frags_new[c_idx]
                            acc_mid = WMMA_IMPL(a_v0, b_v0, acc_in)
                            c_frags_new[c_idx] = WMMA_IMPL(a_v1, b_v1, acc_mid)
                        elif MFMA_PER_WARP_K == 1:
                            c_idx = ii * WARP_N_STEPS + jj
                            c_frags_new[c_idx] = WMMA_IMPL(
                                a_frag, b_frag, c_frags_new[c_idx]
                            )
                        else:
                            raise NotImplementedError(
                                f"MFMA_PER_WARP_K={MFMA_PER_WARP_K} not supported"
                            )
            return c_frags_new

        if IS_SPLIT_K:
            zero_c()

        if B_TO_LDS:
            ldg_sts_a_async(ks_begin, 0)
            ldg_sts_b_async(ks_begin, 0)
            gpu.barrier()

            def hot_loop_scheduler():
                for _ in range_constexpr(WARP_K_STEPS * WARP_M_STEPS):
                    rocdl.sched_dsrd(1)
                for _ in range_constexpr(WARP_K_STEPS * WARP_N_STEPS):
                    rocdl.sched_dsrd(1)
                for _ in range_constexpr(
                    LDG_REG_A_COUNT_AS if ASYNC_COPY else LDG_REG_A_COUNT
                ):
                    rocdl.sched_vmem(1)
                for _ in range_constexpr(
                    LDG_REG_B_COUNT_AS if ASYNC_COPY else LDG_REG_B_COUNT
                ):
                    rocdl.sched_vmem(1)
                for _ in range_constexpr(
                    WARP_K_STEPS * WARP_M_STEPS * WARP_N_STEPS * MFMA_PER_WARP_K
                ):
                    rocdl.sched_mfma(1)
                rocdl.sched_barrier(0)

            UNROLL = 8
            init_state = [ks_begin, arith.constant(0, index=True)] + c_frags
            for bki, state in range(0, BLOCK_K_LOOPS - 1, UNROLL, init=init_state):
                k_offset = state[0]
                current_stage = fx.Index(state[1])
                c_frags = state[2 : 2 + C_FRAGS_LEN]
                for unroll_i in range_constexpr(UNROLL):
                    cond = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        fx.Index(bki + unroll_i),
                        fx.Index(BLOCK_K_LOOPS - 1),
                    )
                    cond_if = scf.IfOp(
                        cond,
                        results_=[T.vec(WMMA_C_FRAG_VALUES, T.f32)] * C_FRAGS_LEN
                        + [T.index, T.i32],
                        has_else=True,
                    )
                    with ir.InsertionPoint(cond_if.then_block):
                        next_stage = 1 - current_stage
                        a_frags = lds_matrix_a(current_stage)
                        b_frags = lds_matrix_b(current_stage)
                        ldg_sts_a_async(k_offset + BLOCK_K, next_stage)
                        ldg_sts_b_async(k_offset + BLOCK_K, next_stage)
                        c_frags_new = block_mma_sync(a_frags, b_frags, c_frags)
                        hot_loop_scheduler()
                        gpu.barrier()
                        k_offset_next = k_offset + fx.Int32(BLOCK_K)
                        current_stage_next = 1 - current_stage
                        scf.YieldOp(
                            c_frags_new + [_to_raw(current_stage_next), k_offset_next]
                        )
                    with ir.InsertionPoint(cond_if.else_block):
                        scf.YieldOp(c_frags + [_to_raw(current_stage), k_offset])
                    c_frags = [cond_if.results[i] for i in range(C_FRAGS_LEN)]
                    current_stage = cond_if.results[C_FRAGS_LEN]
                    k_offset = cond_if.results[C_FRAGS_LEN + 1]
                results = yield [k_offset, current_stage] + c_frags
            current_stage = results[1]
            c_frags = results[2 : 2 + C_FRAGS_LEN]
            a_frags = lds_matrix_a(current_stage)
            b_frags = lds_matrix_b(current_stage)
            c_frags = block_mma_sync(a_frags, b_frags, c_frags)
        else:
            sts_a(ldg_a(ks_begin), 0)
            gpu.barrier()
            a_frags = lds_matrix_a(0)
            b_frags = ldg_matrix_b(ks_begin)
            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                mfma_total = (
                    WARP_K_STEPS * WARP_M_STEPS * WARP_N_STEPS * MFMA_PER_WARP_K
                )
                ldg_reg_a_count_ = LDG_REG_A_COUNT_AS if ASYNC_COPY else LDG_REG_A_COUNT
                ldg_total = ldg_reg_a_count_ + WARP_K_STEPS * WARP_N_STEPS
                mfma_ = OnlineScheduler(mfma_total, mfma_total)
                ldg_ = OnlineScheduler(ldg_total, ldg_total)
                if ASYNC_COPY:
                    avg_mfma_count = (mfma_total + ldg_total - 1) // ldg_total
                    for _ in range_constexpr(ldg_total):
                        rocdl.sched_vmem(ldg_.consume(1))
                        rocdl.sched_mfma(mfma_.consume(avg_mfma_count))
                else:
                    ldg_sts_total = ldg_total + ldg_reg_a_count_
                    avg_mfma_count = (mfma_total + ldg_sts_total - 1) // ldg_sts_total
                    for _ in range_constexpr(ldg_total):
                        rocdl.sched_vmem(ldg_.consume(1))
                        rocdl.sched_mfma(mfma_.consume(avg_mfma_count))
                    for _ in range_constexpr(ldg_reg_a_count_):
                        rocdl.sched_dswr(1)
                        rocdl.sched_mfma(mfma_.consume(avg_mfma_count))
                rocdl.sched_barrier(0)

            init_state = (
                [ks_begin, arith.constant(0, index=True)] + c_frags + a_frags + b_frags
            )
            for _, state in range(1, BLOCK_K_LOOPS, init=init_state):
                k_offset = state[0]
                current_stage = fx.Index(state[1])
                next_stage = 1 - current_stage
                c_frags = state[2 : 2 + C_FRAGS_LEN]
                a_frags = state[2 + C_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN]
                b_frags = state[2 + C_FRAGS_LEN + A_FRAGS_LEN :]
                if ASYNC_COPY:
                    ldg_sts_a_async(k_offset + BLOCK_K, next_stage)
                else:
                    a_regs_next = ldg_a(k_offset + BLOCK_K)
                b_frags_next = ldg_matrix_b(k_offset + BLOCK_K)
                c_frags = block_mma_sync(a_frags, b_frags, c_frags)
                if not ASYNC_COPY:
                    sts_a(a_regs_next, next_stage)
                hot_loop_scheduler()
                gpu.barrier()
                a_frags_next = lds_matrix_a(next_stage)
                k_offset = k_offset + fx.Int32(BLOCK_K)
                rocdl.sched_barrier(0)
                results = (
                    yield [k_offset, next_stage] + c_frags + a_frags_next + b_frags_next
                )
            c_frags = results[2 : 2 + C_FRAGS_LEN]
            a_frags = results[2 + C_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN]
            b_frags = results[2 + C_FRAGS_LEN + A_FRAGS_LEN :]
            c_frags = block_mma_sync(a_frags, b_frags, c_frags)

        stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
        stmatrix_c_n_idx = w_tid % WMMA_N
        gpu.barrier()
        for ii in range_constexpr(WARP_M_STEPS):
            warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
            for jj in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                for kk in range_constexpr(WMMA_C_FRAG_VALUES):
                    lds_m_idx = fx.Index(warp_atom_m_idx + stmatrix_c_m_vec_idx + kk)
                    lds_n_idx = fx.Index(warp_atom_n_idx + stmatrix_c_n_idx)
                    val = vector.extract(
                        c_frags[ii * WARP_N_STEPS + jj],
                        static_position=[kk],
                        dynamic_position=[],
                    )
                    cs_[lds_m_idx, lds_n_idx] = val.truncf(dtype_)

        gpu.barrier()

        vec_dtype_ty = T.vec(LDG_VEC_SIZE, dtype_)
        vec_d2_ty = T.vec(2, dtype_)
        vec_f32x2_ty = T.vec(2, T.f32)
        vec_i16x2_ty = T.vec(2, T.i16)
        vec_i32x1_ty = T.vec(1, T.i32)
        vec_i32x2_ty = T.vec(2, T.i32)
        packed_atom_ty = T.i32x4
        c_zero_i32 = arith.constant(0, type=T.i32)
        c_one_i32 = arith.constant(1, type=T.i32)
        c_four_i64 = arith.constant(4, type=T.i64)
        c_eight_i32 = arith.constant(8, type=T.i32)
        c_sixteen_i32 = arith.constant(16, type=T.i32)
        c_mask000f_i32 = arith.constant(0x000F000F, type=T.i32)
        c_ffff_i32 = arith.constant(0xFFFF, type=T.i32)
        c_abs_mask_i32 = arith.constant(0x7FFF7FFF, type=T.i32)
        c_range_bias_i32 = arith.constant(0x00080008, type=T.i32)
        c_group_width_i32 = arith.constant(8, type=T.i32)
        c_rank_tile_stride_i64 = arith.constant(QR_RANK_TILE_STRIDE, type=T.i64)
        c_scale_offset_i64 = arith.constant(QR_SCALE_OFFSET, type=T.i64)
        c_rank_trans_bytes_i64 = arith.constant(QR_RANK_TRANS_BYTES, type=T.i64)
        c_trans_tile_bytes_i64 = arith.constant(QR_TRANS_TILE_BYTES, type=T.i64)
        c_world_flags_stride_i64 = arith.constant(world_size * 4, type=T.i64)
        tid_div_8 = tid // fx.Int32(8)
        tid_mod_8 = tid % fx.Int32(8)
        tid_i64 = fx.Int64(_mlir_arith.ExtUIOp(T.i64, _unwrap_value(tid)).result)
        tid_div_8_i64 = fx.Int64(
            _mlir_arith.ExtUIOp(T.i64, _unwrap_value(tid_div_8)).result
        )
        rank_i64 = fx.Int64(_mlir_arith.ExtUIOp(T.i64, rank_i32).result)
        zero_packed_atom = vector.bitcast(
            packed_atom_ty, vector.broadcast(vec_dtype_ty, c_zero_d)
        )
        scale_factor_i32 = vector.extract(
            vector.bitcast(
                vec_i32x1_ty,
                vector.broadcast(
                    vec_d2_ty, arith.trunc_f(dtype_, arith.constant(-0.125, type=T.f32))
                ),
            ),
            static_position=[0],
            dynamic_position=[],
        )
        scale_eps_i32 = vector.extract(
            vector.bitcast(
                vec_i32x1_ty,
                vector.broadcast(
                    vec_d2_ty, arith.trunc_f(dtype_, arith.constant(1.0e-7, type=T.f32))
                ),
            ),
            static_position=[0],
            dynamic_position=[],
        )
        range_min_i32 = vector.extract(
            vector.bitcast(
                vec_i32x1_ty,
                vector.broadcast(
                    vec_d2_ty, arith.trunc_f(dtype_, arith.constant(-8.0, type=T.f32))
                ),
            ),
            static_position=[0],
            dynamic_position=[],
        )
        range_max_i32 = vector.extract(
            vector.bitcast(
                vec_i32x1_ty,
                vector.broadcast(
                    vec_d2_ty, arith.trunc_f(dtype_, arith.constant(7.0, type=T.f32))
                ),
            ),
            static_position=[0],
            dynamic_position=[],
        )

        def _load_i32_uncached_relaxed(addr_i64):
            return fx.Int32(
                llvm.InlineAsmOp(
                    T.i32,
                    [_unwrap_value(addr_i64)],
                    "global_load_dword $0, $1, off sc1",
                    "=v,v",
                    has_side_effects=True,
                ).result
            )

        def _store_i32_uncached_relaxed(addr_i64, val_i32):
            llvm.InlineAsmOp(
                None,
                [_unwrap_value(addr_i64), _unwrap_value(val_i32)],
                "global_store_dword $0, $1, off sc0 sc1",
                "v,v",
                has_side_effects=True,
            )

        def _pair_from_i32(packed_i32):
            return vector.bitcast(
                vec_d2_ty, vector.from_elements(vec_i32x1_ty, [packed_i32])
            )

        def _pair_to_i32(pair_val):
            return vector.extract(
                vector.bitcast(vec_i32x1_ty, pair_val),
                static_position=[0],
                dynamic_position=[],
            )

        def _pair_i16_to_i32(pair_val):
            return vector.extract(
                vector.bitcast(vec_i32x1_ty, pair_val),
                static_position=[0],
                dynamic_position=[],
            )

        def _packed_add(a_i32, b_i32):
            return _pair_to_i32(
                _mlir_arith.AddFOp(
                    _unwrap_value(_pair_from_i32(a_i32)),
                    _unwrap_value(_pair_from_i32(b_i32)),
                ).result
            )

        def _packed_mul(a_i32, b_i32):
            return _pair_to_i32(
                _mlir_arith.MulFOp(
                    _unwrap_value(_pair_from_i32(a_i32)),
                    _unwrap_value(_pair_from_i32(b_i32)),
                ).result
            )

        def _packed_max(a_i32, b_i32):
            return _pair_to_i32(
                _mlir_arith.MaximumFOp(
                    _unwrap_value(_pair_from_i32(a_i32)),
                    _unwrap_value(_pair_from_i32(b_i32)),
                ).result
            )

        def _packed_min(a_i32, b_i32):
            return _pair_to_i32(
                _mlir_arith.MinimumFOp(
                    _unwrap_value(_pair_from_i32(a_i32)),
                    _unwrap_value(_pair_from_i32(b_i32)),
                ).result
            )

        def _packed_add_i16(a_i32, b_i32):
            return fx.Int32(
                llvm.InlineAsmOp(
                    T.i32,
                    [_unwrap_value(a_i32), _unwrap_value(b_i32)],
                    "v_pk_add_i16 $0, $1, $2",
                    "=v,v,v",
                    has_side_effects=False,
                ).result
            )

        def _packed_abs_max(a_i32, b_i32):
            a_abs = a_i32 & c_abs_mask_i32
            b_abs = b_i32 & c_abs_mask_i32
            lo = arith.select(
                arith.cmpi(
                    arith.CmpIPredicate.uge,
                    a_abs & c_ffff_i32,
                    b_abs & c_ffff_i32,
                ),
                a_i32 & c_ffff_i32,
                b_i32 & c_ffff_i32,
            )
            hi = arith.select(
                arith.cmpi(
                    arith.CmpIPredicate.uge,
                    (a_abs >> c_sixteen_i32) & c_ffff_i32,
                    (b_abs >> c_sixteen_i32) & c_ffff_i32,
                ),
                (a_i32 >> c_sixteen_i32) & c_ffff_i32,
                (b_i32 >> c_sixteen_i32) & c_ffff_i32,
            )
            return lo | (hi << c_sixteen_i32)

        if dtype == "bf16":

            def _packed_rcp(a_i32):
                a_pair = _pair_from_i32(a_i32)
                outs = []
                for pi in range_constexpr(2):
                    av = arith.extf(
                        T.f32,
                        vector.extract(a_pair, static_position=[pi], dynamic_position=[]),
                    )
                    outs.append(
                        arith.trunc_f(
                            dtype_,
                            llvm.call_intrinsic(
                                T.f32,
                                "llvm.amdgcn.rcp.f32",
                                [av],
                                [],
                                [],
                            ),
                        )
                    )
                return _pair_to_i32(vector.from_elements(vec_d2_ty, outs))

        else:

            def _packed_rcp(a_i32):
                a_pair_f32 = _mlir_arith.ExtFOp(
                    vec_f32x2_ty, _unwrap_value(_pair_from_i32(a_i32))
                ).result
                return _pair_to_i32(
                    _mlir_arith.TruncFOp(
                        vec_d2_ty,
                        llvm.call_intrinsic(
                            vec_f32x2_ty,
                            "llvm.amdgcn.rcp",
                            [a_pair_f32],
                            [],
                            [],
                        ),
                    ).result
                )

        def _packed_add_atom(a_atom, b_atom):
            outs = []
            for li in range_constexpr(4):
                outs.append(
                    _packed_add(
                        vector.extract(a_atom, static_position=[li], dynamic_position=[]),
                        vector.extract(b_atom, static_position=[li], dynamic_position=[]),
                    )
                )
            return vector.from_elements(packed_atom_ty, outs)

        def _shuffle_xor_i32(val_i32, offset: int):
            return fx.Int32(
                _mlir_gpu.ShuffleOp(
                    _unwrap_value(val_i32),
                    arith.constant(offset, type=T.i32),
                    c_group_width_i32,
                    mode="xor",
                ).shuffleResult
            )

        def _group_abs_max_from_lanes(lanes):
            a = _packed_max(lanes[0], lanes[1])
            b = _packed_max(lanes[2], lanes[3])
            wmax = _packed_max(a, b)
            a = _packed_min(lanes[0], lanes[1])
            b = _packed_min(lanes[2], lanes[3])
            wmin = _packed_min(a, b)
            for offset in (1, 2, 4):
                wmax = _packed_max(wmax, _shuffle_xor_i32(wmax, offset))
                wmin = _packed_min(wmin, _shuffle_xor_i32(wmin, offset))
            return _packed_abs_max(wmax, wmin)

        def _group_abs_max(atom_i32x4):
            lanes = [
                vector.extract(atom_i32x4, static_position=[li], dynamic_position=[])
                for li in range_constexpr(4)
            ]
            return _group_abs_max_from_lanes(lanes)

        def _encode_q4(atom_i32x4):
            lanes = [
                vector.extract(atom_i32x4, static_position=[li], dynamic_position=[])
                for li in range_constexpr(4)
            ]
            dec_scale_i32 = _packed_mul(
                _group_abs_max_from_lanes(lanes), scale_factor_i32
            )
            enc_scale_i32 = _packed_rcp(_packed_add(dec_scale_i32, scale_eps_i32))
            q_lanes = []
            for li in range_constexpr(4):
                lane_i32 = _packed_mul(lanes[li], enc_scale_i32)
                lane_i32 = _packed_max(lane_i32, range_min_i32)
                lane_i32 = _packed_min(lane_i32, range_max_i32)
                lane_pair = _pair_from_i32(lane_i32)
                lane_pair_f32 = _mlir_arith.ExtFOp(
                    vec_f32x2_ty, _unwrap_value(lane_pair)
                ).result
                rounded_pair_i32 = _mlir_arith.FPToSIOp(
                    vec_i32x2_ty,
                    llvm.call_intrinsic(
                        vec_f32x2_ty,
                        "llvm.rint.v2f32",
                        [lane_pair_f32],
                        [],
                        [],
                    ),
                ).result
                q_pair_i32 = _pair_i16_to_i32(
                    llvm.call_intrinsic(
                        vec_i16x2_ty,
                        "llvm.amdgcn.cvt.pk.i16",
                        [
                            _unwrap_value(
                                vector.extract(
                                    rounded_pair_i32,
                                    static_position=[0],
                                    dynamic_position=[],
                                )
                            ),
                            _unwrap_value(
                                vector.extract(
                                    rounded_pair_i32,
                                    static_position=[1],
                                    dynamic_position=[],
                                )
                            ),
                        ],
                        [],
                        [],
                    )
                )
                q_lanes.append(_packed_add_i16(q_pair_i32, c_range_bias_i32))
            qw_i32 = q_lanes[0]
            for li in range_constexpr(1, 4):
                qw_i32 = qw_i32 | (q_lanes[li] << arith.constant(li * 4, type=T.i32))
            return qw_i32, dec_scale_i32

        def _decode_q4_words(qw_i32, qs_i32):
            lanes = []
            for li in range_constexpr(4):
                int16_2 = (qw_i32 >> arith.constant(li * 4, type=T.i32)) & c_mask000f_i32
                low_i32 = (int16_2 & c_ffff_i32) - c_eight_i32
                high_i32 = ((int16_2 >> c_sixteen_i32) & c_ffff_i32) - c_eight_i32
                low_d = arith.trunc_f(dtype_, arith.sitofp(T.f32, low_i32))
                high_d = arith.trunc_f(dtype_, arith.sitofp(T.f32, high_i32))
                lanes.append(
                    _packed_mul(
                        _pair_to_i32(vector.from_elements(vec_d2_ty, [low_d, high_d])),
                        qs_i32,
                    )
                )
            return vector.from_elements(packed_atom_ty, lanes)

        def _wait_flag(flag_addr_i64, expect_i32):
            is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tid, c_zero_i32)
            if_t0 = scf.IfOp(is_t0, results_=[], has_else=False)
            with ir.InsertionPoint(if_t0.then_block):
                init_cur = load_i32_uncached(flag_addr_i64)
                loop = scf.WhileOp([T.i32], [init_cur])
                before = ir.Block.create_at_start(loop.before, [T.i32])
                after = ir.Block.create_at_start(loop.after, [T.i32])
                with ir.InsertionPoint(before):
                    cur = before.arguments[0]
                    need_wait = arith.CmpIOp(
                        arith.CmpIPredicate.ult,
                        cur,
                        expect_i32,
                    ).result
                    scf.ConditionOp(need_wait, [cur])
                with ir.InsertionPoint(after):
                    scf.YieldOp([load_i32_uncached(flag_addr_i64)])
                scf.YieldOp([])
            gpu.barrier()

        local_atoms = [zero_packed_atom for _ in range_constexpr(8)]
        for i in range_constexpr(LDG_REG_C_COUNT):
            global_tid = BLOCK_THREADS * i + tid
            m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
            n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
            m_global_idx = m_offset + m_local_idx
            cond_boundary = arith.cmpi(
                arith.CmpIPredicate.ult, m_global_idx, fx.Index(m)
            )
            load_if = scf.IfOp(cond_boundary, results_=[packed_atom_ty], has_else=True)
            with ir.InsertionPoint(load_if.then_block):
                scf.YieldOp(
                    [
                        vector.bitcast(
                            packed_atom_ty,
                            cs_.vec_load((m_local_idx, n_local_idx), LDG_VEC_SIZE),
                        )
                    ]
                )
            with ir.InsertionPoint(load_if.else_block):
                scf.YieldOp([zero_packed_atom])
            local_atoms[i] = load_if.results[0]

        phase0_flags_base = bid_i64 * c_world_flags_stride_i64
        phase1_flags_base = flags_phase_bytes_i64 + phase0_flags_base
        phase0_data_base = data_offset_i64 + bid_i64 * c_trans_tile_bytes_i64
        phase1_data_base = data_offset_i64 + phase_bytes_i64 + bid_i64 * c_trans_tile_bytes_i64
        rank_byte_offset = rank_i64 * c_four_i64
        phase0_self_flag = self_buffer_ptr + phase0_flags_base + rank_byte_offset
        phase1_self_flag = self_buffer_ptr + phase1_flags_base + rank_byte_offset
        phase0_flag = load_i32_uncached(phase0_self_flag) + c_one_i32
        phase1_flag = load_i32_uncached(phase1_self_flag) + c_one_i32

        for r in range_constexpr(world_size):
            peer_buf = select_by_index(arith.constant(r, type=T.i32), buffer_ptrs_arr)
            send_base = peer_buf + phase0_data_base + rank_i64 * c_rank_trans_bytes_i64
            for ka in range_constexpr(QR_RANK_ATOMS):
                atom_base = send_base + arith.constant(ka * QR_RANK_TILE_STRIDE, type=T.i64)
                qw_i32, qs_i32 = _encode_q4(local_atoms[r * QR_RANK_ATOMS + ka])
                _store_i32_uncached_relaxed(atom_base + tid_i64 * c_four_i64, qw_i32)
                is_group_leader = arith.cmpi(arith.CmpIPredicate.eq, tid_mod_8, c_zero_i32)
                leader_if = scf.IfOp(is_group_leader, results_=[], has_else=False)
                with ir.InsertionPoint(leader_if.then_block):
                    scale_addr = atom_base + c_scale_offset_i64 + tid_div_8_i64 * c_four_i64
                    _store_i32_uncached_relaxed(scale_addr, qs_i32)
                    scf.YieldOp([])
        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)
        gpu.barrier()

        signal_if = scf.IfOp(
            arith.cmpi(
                arith.CmpIPredicate.ult,
                tid,
                arith.constant(world_size, type=T.i32),
            ),
            results_=[],
            has_else=False,
        )
        with ir.InsertionPoint(signal_if.then_block):
            peer_buf = select_by_index(tid, buffer_ptrs_arr)
            flag_addr = peer_buf + phase0_flags_base + rank_byte_offset
            store_i32_uncached_flush(flag_addr, phase0_flag)
            scf.YieldOp([])
        gpu.barrier()

        reduced_atoms = [zero_packed_atom for _ in range_constexpr(QR_RANK_ATOMS)]
        for r in range_constexpr(world_size):
            wait_addr = self_buffer_ptr + phase0_flags_base + arith.constant(r * 4, type=T.i64)
            _wait_flag(wait_addr, phase0_flag)
            recv_base = self_buffer_ptr + phase0_data_base + arith.constant(r * QR_RANK_TRANS_BYTES, type=T.i64)
            recv_qw = [c_zero_i32 for _ in range_constexpr(QR_RANK_ATOMS)]
            recv_qs = [c_zero_i32 for _ in range_constexpr(QR_RANK_ATOMS)]
            for ka in range_constexpr(QR_RANK_ATOMS):
                atom_base = recv_base + arith.constant(ka * QR_RANK_TILE_STRIDE, type=T.i64)
                recv_qw[ka] = _load_i32_uncached_relaxed(atom_base + tid_i64 * c_four_i64)
                scale_addr = atom_base + c_scale_offset_i64 + tid_div_8_i64 * c_four_i64
                load_scale_if = scf.IfOp(
                    arith.cmpi(arith.CmpIPredicate.eq, tid_mod_8, c_zero_i32),
                    results_=[T.i32],
                    has_else=True,
                )
                with ir.InsertionPoint(load_scale_if.then_block):
                    scf.YieldOp([_unwrap_value(_load_i32_uncached_relaxed(scale_addr))])
                with ir.InsertionPoint(load_scale_if.else_block):
                    scf.YieldOp([_unwrap_value(c_zero_i32)])
                recv_qs[ka] = load_scale_if.results[0]
            rocdl.s_waitcnt(0)
            rocdl.sched_barrier(0)
            for ka in range_constexpr(QR_RANK_ATOMS):
                qs_i32 = recv_qs[ka]
                for offset in (1, 2, 4):
                    qs_i32 = qs_i32 | _shuffle_xor_i32(qs_i32, offset)
                reduced_atoms[ka] = _packed_add_atom(
                    reduced_atoms[ka],
                    _decode_q4_words(recv_qw[ka], qs_i32),
                )

        for r in range_constexpr(world_size):
            peer_buf = select_by_index(arith.constant(r, type=T.i32), buffer_ptrs_arr)
            send_base = peer_buf + phase1_data_base + rank_i64 * c_rank_trans_bytes_i64
            for ka in range_constexpr(QR_RANK_ATOMS):
                atom_base = send_base + arith.constant(ka * QR_RANK_TILE_STRIDE, type=T.i64)
                qw_i32, qs_i32 = _encode_q4(reduced_atoms[ka])
                _store_i32_uncached_relaxed(atom_base + tid_i64 * c_four_i64, qw_i32)
                is_group_leader = arith.cmpi(arith.CmpIPredicate.eq, tid_mod_8, c_zero_i32)
                leader_if = scf.IfOp(is_group_leader, results_=[], has_else=False)
                with ir.InsertionPoint(leader_if.then_block):
                    scale_addr = atom_base + c_scale_offset_i64 + tid_div_8_i64 * c_four_i64
                    _store_i32_uncached_relaxed(scale_addr, qs_i32)
                    scf.YieldOp([])
        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)
        gpu.barrier()

        signal_if = scf.IfOp(
            arith.cmpi(
                arith.CmpIPredicate.ult,
                tid,
                arith.constant(world_size, type=T.i32),
            ),
            results_=[],
            has_else=False,
        )
        with ir.InsertionPoint(signal_if.then_block):
            peer_buf = select_by_index(tid, buffer_ptrs_arr)
            flag_addr = peer_buf + phase1_flags_base + rank_byte_offset
            store_i32_uncached_flush(flag_addr, phase1_flag)
            scf.YieldOp([])
        gpu.barrier()

        final_atoms = [zero_packed_atom for _ in range_constexpr(8)]
        for r in range_constexpr(world_size):
            wait_addr = self_buffer_ptr + phase1_flags_base + arith.constant(r * 4, type=T.i64)
            _wait_flag(wait_addr, phase1_flag)
            recv_base = self_buffer_ptr + phase1_data_base + arith.constant(r * QR_RANK_TRANS_BYTES, type=T.i64)
            recv_qw = [c_zero_i32 for _ in range_constexpr(QR_RANK_ATOMS)]
            recv_qs = [c_zero_i32 for _ in range_constexpr(QR_RANK_ATOMS)]
            for ka in range_constexpr(QR_RANK_ATOMS):
                atom_base = recv_base + arith.constant(ka * QR_RANK_TILE_STRIDE, type=T.i64)
                recv_qw[ka] = _load_i32_uncached_relaxed(atom_base + tid_i64 * c_four_i64)
                scale_addr = atom_base + c_scale_offset_i64 + tid_div_8_i64 * c_four_i64
                load_scale_if = scf.IfOp(
                    arith.cmpi(arith.CmpIPredicate.eq, tid_mod_8, c_zero_i32),
                    results_=[T.i32],
                    has_else=True,
                )
                with ir.InsertionPoint(load_scale_if.then_block):
                    scf.YieldOp([_unwrap_value(_load_i32_uncached_relaxed(scale_addr))])
                with ir.InsertionPoint(load_scale_if.else_block):
                    scf.YieldOp([_unwrap_value(c_zero_i32)])
                recv_qs[ka] = load_scale_if.results[0]
            rocdl.s_waitcnt(0)
            rocdl.sched_barrier(0)
            for ka in range_constexpr(QR_RANK_ATOMS):
                qs_i32 = recv_qs[ka]
                for offset in (1, 2, 4):
                    qs_i32 = qs_i32 | _shuffle_xor_i32(qs_i32, offset)
                final_atoms[r * QR_RANK_ATOMS + ka] = _decode_q4_words(
                    recv_qw[ka],
                    qs_i32,
                )

        for i in range_constexpr(LDG_REG_C_COUNT):
            global_tid = BLOCK_THREADS * i + tid
            m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
            n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
            m_global_idx = m_offset + m_local_idx
            cond_boundary = arith.cmpi(
                arith.CmpIPredicate.ult, m_global_idx, fx.Index(m)
            )
            cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
            with ir.InsertionPoint(cond_boundary_if.then_block):
                C_.vec_store(
                    (m_global_idx, n_offset + n_local_idx),
                    vector.bitcast(vec_dtype_ty, final_atoms[i]),
                    LDG_VEC_SIZE,
                )
                scf.YieldOp([])
        return

    @flyc.jit
    def launch_hgemm_qr_kernel(
        rank: Int32,
        buffer_ptrs: Int64,
        flags_phase_bytes: Int64,
        data_offset: Int64,
        phase_bytes: Int64,
        C: fx.Tensor,
        A: fx.Tensor,
        B: fx.Tensor,
        m: fx.Int32,
        COUNTER: fx.Tensor,
        signal_state: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bm = (m + BLOCK_M - 1) // BLOCK_M
        bn = n // BLOCK_N
        hgemm_qr_kernel._func.__name__ = KERNEL_NAME
        hgemm_qr_kernel(
            rank,
            buffer_ptrs,
            flags_phase_bytes,
            data_offset,
            phase_bytes,
            C,
            A,
            B,
            m,
            COUNTER,
            signal_state,
        ).launch(
            grid=(bm, bn, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    _compile_hints = {
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_compile_hints):
            return launch_hgemm_qr_kernel(*args, **kwargs)

    _compile_cache = {}

    def _compile(rank, buffer_ptrs, flags_phase_bytes, data_offset, phase_bytes, C, A, B, m, COUNTER, signal_state, stream):
        with CompilationContext.compile_hints(_compile_hints):
            if _compile_cache.get(m, None) is None:
                _compile_cache[m] = flyc.compile(
                    launch_hgemm_qr_kernel, rank, buffer_ptrs, flags_phase_bytes, data_offset, phase_bytes, C, A, B, m, COUNTER, signal_state, stream
                )
            return _compile_cache[m]

    _launch.compile = _compile

    return _launch
