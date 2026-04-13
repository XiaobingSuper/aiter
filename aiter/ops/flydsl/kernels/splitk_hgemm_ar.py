# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Low-level FlyDSL compile for fused HGEMM + tensor-parallel all-reduce."""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, memref, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import fly_values
from flydsl.expr import arith, gpu, range_constexpr, rocdl, vector
from flydsl.expr.buffer_ops import _unwrap_value
from flydsl.expr.typing import Int32, Int64, T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from ..gemm_kernels import (
    SPLIT_K_COUNTER_MAX_LEN,
)
from .custom_all_reduce_kernel import (
    _signal_start_sync,
    load_device_ptr,
    load_v4i32,
    select_by_index,
    store_v4i32,
    store_v4i32_nt,
)
from .hgemm_ar_params import flydsl_tp_hgemm_ar_kernel_name
from .splitk_hgemm import (
    OnlineScheduler,
    WmmaHalf_m16n16k16,
    WmmaHalf_m16n16k32,
    swizzle_xor16,
)
from .tensor_shim import GTensor, STensor, _to_raw, get_dtype_in_kernel


@functools.lru_cache(maxsize=1024)
def compile_hgemm_ar_kernel(
    world_size: int,
    dtype: str,
    n: int,
    k: int,
    TILE_M: int = 128,
    TILE_N: int = 128,
    TILE_K: int = 64,
    SPLIT_K: int = 1,
    BLOCK_M_WARPS: int = 2,
    BLOCK_N_WARPS: int = 2,
    B_PRE_SHUFFLE: bool = False,
    B_TO_LDS: bool = True,
    USE_ATOMIC_ADD: bool = False,
):
    IS_SPLIT_K = SPLIT_K > 1
    USE_CROSS_DEVICE_ATOMIC = bool(USE_ATOMIC_ADD) or IS_SPLIT_K
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

    KERNEL_NAME = flydsl_tp_hgemm_ar_kernel_name(
        dtype,
        tile_m=TILE_M,
        tile_n=TILE_N,
        tile_k=TILE_K,
        split_k=SPLIT_K,
        block_m_warps=BLOCK_M_WARPS,
        block_n_warps=BLOCK_N_WARPS,
        b_preshuffle=B_PRE_SHUFFLE,
        b_to_lds=B_TO_LDS,
        use_atomic_add=USE_CROSS_DEVICE_ATOMIC,
        stages=STAGES,
        async_copy=ASYNC_COPY,
    )

    @flyc.kernel
    def hgemm_ar_kernel(
        rank: Int32,
        self_sg: Int64,
        sg_ptrs: Int64,
        tmp_ptrs: Int64,
        out_ptrs: Int64,
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
            base_ptr,
            smem_a_offset,
            dtype_,
            shape=(STAGES * BLOCK_M * BLOCK_K,),
        )
        as_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, BLOCK_M, BLOCK_K))
        if B_TO_LDS:
            smem_b_ptr = SmemPtr(
                base_ptr,
                smem_b_offset,
                dtype_,
                shape=(STAGES * BLOCK_N * BLOCK_K,),
            )
            bs_ = STensor(smem_b_ptr, dtype_, shape=(STAGES, BLOCK_N, BLOCK_K))
        smem_c_ptr = SmemPtr(
            base_ptr,
            smem_a_offset,
            dtype_,
            shape=(BLOCK_M * BLOCK_N,),
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
        ldmatrix_a_k_vec_idx = (
            w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K
        )
        ldmatrix_b_n_idx = w_tid % WMMA_N
        ldmatrix_b_k_vec_idx = (
            w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
        )
        A_FRAGS_LEN = WARP_K_STEPS * WARP_M_STEPS
        B_FRAGS_LEN = WARP_K_STEPS * WARP_N_STEPS
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
        c_frags = [acc_init] * C_FRAGS_LEN

        bid_linear = (
            (fx.block_idx.x * (n // BLOCK_N) + fx.block_idx.y) * SPLIT_K + fx.block_idx.z
        )
        rank_i32 = _unwrap_value(rank)
        self_sg_i64 = _unwrap_value(self_sg)
        sg_ptrs_i64 = _unwrap_value(sg_ptrs)
        tmp_ptrs_i64 = _unwrap_value(tmp_ptrs)
        out_ptrs_i64 = _unwrap_value(out_ptrs)
        bid_i32 = arith.index_cast(T.i32, fx.Index(bid_linear))
        lane_i32 = arith.index_cast(T.i32, fx.Index(fx.thread_idx.x))
        sgs = [
            load_device_ptr(sg_ptrs_i64, arith.constant(i, type=T.i32))
            for i in range(8)
        ]
        tmp_ptrs_arr = [
            load_device_ptr(tmp_ptrs_i64, arith.constant(i, type=T.i32))
            for i in range(8)
        ]
        out_ptrs_arr = [
            load_device_ptr(out_ptrs_i64, arith.constant(i, type=T.i32))
            for i in range(8)
        ]
        self_tmp_ptr = select_by_index(arith.constant(0, type=T.i32), tmp_ptrs_arr)
        self_out_ptr = select_by_index(rank_i32, out_ptrs_arr)

        def zero_c():
            cond_ks0 = arith.cmpi(arith.CmpIPredicate.eq, ks_idx, fx.Index(0))
            cond_ks0_if = scf.IfOp(cond_ks0, results_=[], has_else=False)
            with ir.InsertionPoint(cond_ks0_if.then_block):
                zero_vec = vector.broadcast(T.vec(LDG_VEC_SIZE, dtype_), c_zero_d)
                vec_i32x4 = vector.bitcast(T.i32x4, zero_vec)
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
                        linear_byte_offset = (
                            C_.linear_offset((row_idx, n_offset + n_local_idx)) * DTYPE_BYTES
                        )
                        byte_offset_i64 = arith.index_cast(T.i64, linear_byte_offset)
                        if USE_CROSS_DEVICE_ATOMIC:
                            store_v4i32_nt(self_out_ptr + byte_offset_i64, vec_i32x4)
                        else:
                            store_v4i32_nt(self_tmp_ptr + byte_offset_i64, vec_i32x4)
                        scf.YieldOp([])
                scf.YieldOp([])
            if IS_SPLIT_K:
                rocdl.sched_barrier(0)
                gpu.barrier()

                cond_ks0_if = scf.IfOp(cond_ks0, results_=[], has_else=False)
                with ir.InsertionPoint(cond_ks0_if.then_block):
                    is_t0_cond = arith.cmpi(
                        arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0)
                    )
                    is_t0_cond_if = scf.IfOp(
                        is_t0_cond, results_=[], has_else=False
                    )
                    with ir.InsertionPoint(is_t0_cond_if.then_block):
                        counter_base_ptr = fly_values(COUNTER)[0]
                        counter_base_ptr = fly.extract_aligned_pointer_as_index(
                            _ptr_type, counter_base_ptr
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
                            None,
                            [],
                            "buffer_wbl2 sc0 sc1",
                            "",
                            has_side_effects=True,
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
                    clean_cond_if = scf.IfOp(
                        clean_cond, results_=[], has_else=False
                    )
                    with ir.InsertionPoint(clean_cond_if.then_block):
                        clean_counter_idx = fx.Int32(
                            (
                                (signal_state + 2) % 3
                            )
                            * SPLIT_K_COUNTER_MAX_LEN
                        ) + fx.Index(tid)
                        COUNTER_[fx.Index(clean_counter_idx)] = arith.constant(
                            0,
                            type=T.i32,
                        )
                        scf.YieldOp([])
                    scf.YieldOp([])
                rocdl.sched_barrier(0)
                gpu.barrier()

        def split_k_barrier():
            init_cur = arith.constant(0, type=T.i32)
            loop = scf.WhileOp([T.i32], [init_cur])
            before = ir.Block.create_at_start(loop.before, [T.i32])
            after = ir.Block.create_at_start(loop.after, [T.i32])
            with ir.InsertionPoint(before):
                cur = before.arguments[0]
                need_wait = arith.CmpIOp(
                    arith.CmpIPredicate.eq,
                    cur,
                    arith.constant(0, type=T.i32),
                ).result
                scf.ConditionOp(need_wait, [cur])
            with ir.InsertionPoint(after):
                counter_base_ptr = fly_values(COUNTER)[0]
                counter_base_ptr = fly.extract_aligned_pointer_as_index(
                    _ptr_type, counter_base_ptr
                )
                counter_base_ptr = llvm.PtrToIntOp(_i64_type, counter_base_ptr).result
                counter_byte_offset = arith.index_cast(
                    T.i64,
                    fx.Index(counter_idx) * fx.Index(4),
                )
                counter_ptr = llvm.AddOp(
                    counter_base_ptr,
                    counter_byte_offset,
                    llvm.IntegerOverflowFlags(0),
                ).result
                counter_ptr = llvm.IntToPtrOp(_ptr_type, counter_ptr).result
                counter_ptr_v = (
                    counter_ptr._value if hasattr(counter_ptr, "_value") else counter_ptr
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
                lds_addr = memref.extract_aligned_pointer_as_index(as_.memptr) + lds_offset
                lds_addr_ = rocdl.readfirstlane(
                    T.i64,
                    arith.index_cast(T.i64, lds_addr),
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
                lds_addr = memref.extract_aligned_pointer_as_index(bs_.memptr) + lds_offset
                lds_addr_ = rocdl.readfirstlane(
                    T.i64,
                    arith.index_cast(T.i64, lds_addr),
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
                                (n_idx, k_idx),
                                WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
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
                                a_i64x2,
                                static_position=[0],
                                dynamic_position=[],
                            )
                            a1_i64 = vector.extract(
                                a_i64x2,
                                static_position=[1],
                                dynamic_position=[],
                            )
                            a_v0 = vector.bitcast(
                                T.f16x4,
                                vector.from_elements(T.vec(1, T.i64), [a0_i64]),
                            )
                            a_v1 = vector.bitcast(
                                T.f16x4,
                                vector.from_elements(T.vec(1, T.i64), [a1_i64]),
                            )
                            b_i64x2 = vector.bitcast(T.i64x2, b_frag)
                            b0_i64 = vector.extract(
                                b_i64x2,
                                static_position=[0],
                                dynamic_position=[],
                            )
                            b1_i64 = vector.extract(
                                b_i64x2,
                                static_position=[1],
                                dynamic_position=[],
                            )
                            b_v0 = vector.bitcast(
                                T.f16x4,
                                vector.from_elements(T.vec(1, T.i64), [b0_i64]),
                            )
                            b_v1 = vector.bitcast(
                                T.f16x4,
                                vector.from_elements(T.vec(1, T.i64), [b1_i64]),
                            )
                            c_idx = ii * WARP_N_STEPS + jj
                            acc_in = c_frags_new[c_idx]
                            acc_mid = WMMA_IMPL(a_v0, b_v0, acc_in)
                            c_frags_new[c_idx] = WMMA_IMPL(a_v1, b_v1, acc_mid)
                        elif MFMA_PER_WARP_K == 1:
                            c_idx = ii * WARP_N_STEPS + jj
                            c_frags_new[c_idx] = WMMA_IMPL(
                                a_frag,
                                b_frag,
                                c_frags_new[c_idx],
                            )
                        else:
                            raise NotImplementedError(
                                f"MFMA_PER_WARP_K={MFMA_PER_WARP_K} not supported"
                            )
            return c_frags_new

        zero_c()

        if B_TO_LDS:
            ldg_sts_a_async(ks_begin, 0)
            ldg_sts_b_async(ks_begin, 0)
            gpu.barrier()

            def hot_loop_scheduler():
                mfma_total = (
                    WARP_K_STEPS * WARP_M_STEPS * WARP_N_STEPS * MFMA_PER_WARP_K
                )
                ldg_reg_a_count_ = LDG_REG_A_COUNT_AS if ASYNC_COPY else LDG_REG_A_COUNT
                ldg_reg_b_count_ = LDG_REG_B_COUNT_AS if ASYNC_COPY else LDG_REG_B_COUNT
                for _ in range_constexpr(WARP_K_STEPS * WARP_M_STEPS):
                    rocdl.sched_dsrd(1)
                for _ in range_constexpr(WARP_K_STEPS * WARP_N_STEPS):
                    rocdl.sched_dsrd(1)
                for _ in range_constexpr(ldg_reg_a_count_):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(2)
                for _ in range_constexpr(ldg_reg_b_count_):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(2)
                remaining = max(mfma_total - (ldg_reg_a_count_ + ldg_reg_b_count_) * 2, 0)
                for _ in range_constexpr(remaining):
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
                ldg_reg_a_count_ = (
                    LDG_REG_A_COUNT_AS if ASYNC_COPY else LDG_REG_A_COUNT
                )
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
                b_frags = state[
                    2
                    + C_FRAGS_LEN
                    + A_FRAGS_LEN : 2
                    + C_FRAGS_LEN
                    + A_FRAGS_LEN
                    + B_FRAGS_LEN
                ]
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
            b_frags = results[
                2
                + C_FRAGS_LEN
                + A_FRAGS_LEN : 2
                + C_FRAGS_LEN
                + A_FRAGS_LEN
                + B_FRAGS_LEN
            ]
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

        if IS_SPLIT_K:
            split_k_barrier()
        else:
            gpu.barrier()

        if USE_CROSS_DEVICE_ATOMIC:
            _signal_start_sync(
                lane_i32=lane_i32,
                rank_i32=rank_i32,
                bid_i32=bid_i32,
                self_sg_i64=self_sg_i64,
                sgs_i64=sgs,
                ngpus=world_size,
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
                    pk_val = cs_.vec_load((m_local_idx, n_local_idx), LDG_VEC_SIZE)
                    linear_bytes_offset = (
                        C_.linear_offset((m_global_idx, n_offset + n_local_idx))
                        * DTYPE_BYTES
                    )
                    vec2_ty = T.vec(2, dtype_)
                    for wi in range_constexpr(world_size):
                        out_memref = select_by_index(
                            arith.constant(wi, type=T.i32), out_ptrs_arr
                        )
                        for vec_idx in range_constexpr(LDG_VEC_SIZE // 2):
                            e0 = vector.extract(
                                pk_val,
                                static_position=[vec_idx * 2],
                                dynamic_position=[],
                            )
                            e1 = vector.extract(
                                pk_val,
                                static_position=[vec_idx * 2 + 1],
                                dynamic_position=[],
                            )
                            pair = vector.from_elements(vec2_ty, [e0, e1])
                            pair_byte_offset = arith.index_cast(
                                T.i64,
                                linear_bytes_offset + fx.Index(vec_idx * 2 * DTYPE_BYTES),
                            )
                            pair_addr_i64 = llvm.AddOp(
                                out_memref,
                                pair_byte_offset,
                                llvm.IntegerOverflowFlags(0),
                            ).result
                            pair_ptr = llvm.IntToPtrOp(_ptr_type, pair_addr_i64).result
                            pair_ptr_v = (
                                pair_ptr._value if hasattr(pair_ptr, "_value") else pair_ptr
                            )
                            pair_v = pair._value if hasattr(pair, "_value") else pair
                            llvm.AtomicRMWOp(
                                llvm.AtomicBinOp.fadd,
                                pair_ptr_v,
                                pair_v,
                                llvm.AtomicOrdering.monotonic,
                                syncscope="agent",
                                alignment=4,
                            )
                    scf.YieldOp([])
        else:
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
                    vec = cs_.vec_load((m_local_idx, n_local_idx), LDG_VEC_SIZE)
                    vec_i32x4 = vector.bitcast(T.i32x4, vec)
                    linear_bytes_offset = (
                        C_.linear_offset((m_global_idx, n_offset + n_local_idx))
                        * DTYPE_BYTES
                    )
                    store_v4i32_nt(
                        self_tmp_ptr + arith.index_cast(T.i64, linear_bytes_offset),
                        vec_i32x4,
                    )
                    scf.YieldOp([])

            _signal_start_sync(
                lane_i32=lane_i32,
                rank_i32=rank_i32,
                bid_i32=bid_i32,
                self_sg_i64=self_sg_i64,
                sgs_i64=sgs,
                ngpus=world_size,
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
                    linear_bytes_offset = (
                        C_.linear_offset((m_global_idx, n_offset + n_local_idx))
                        * DTYPE_BYTES
                    )
                    final_vec = arith.constant_vector(0.0, T.vec(LDG_VEC_SIZE, T.f32))
                    for wi in range_constexpr(world_size):
                        vec_v4i32 = load_v4i32(
                            select_by_index(arith.constant(wi, type=T.i32), tmp_ptrs_arr)
                            + arith.index_cast(T.i64, linear_bytes_offset)
                        )
                        vec = vector.bitcast(T.vec(LDG_VEC_SIZE, dtype_), vec_v4i32)
                        final_vec = final_vec + vec.extf(T.vec(LDG_VEC_SIZE, T.f32))
                    final_vec_i32x4 = vector.bitcast(
                        T.i32x4,
                        final_vec.truncf(T.vec(LDG_VEC_SIZE, dtype_)),
                    )
                    store_v4i32(
                        self_out_ptr + arith.index_cast(T.i64, linear_bytes_offset),
                        final_vec_i32x4,
                    )
                    scf.YieldOp([])
        return

    @flyc.jit
    def launch_hgemm_ar_kernel(
        rank: Int32,
        self_sg: Int64,
        sg_ptrs: Int64,
        tmp_ptrs: Int64,
        out_ptrs: Int64,
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
        hgemm_ar_kernel._func.__name__ = KERNEL_NAME
        hgemm_ar_kernel(
            rank,
            self_sg,
            sg_ptrs,
            tmp_ptrs,
            out_ptrs,
            C,
            A,
            B,
            m,
            COUNTER,
            signal_state,
        ).launch(
            grid=(bm, bn, SPLIT_K),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    _compile_hints = {}

    def _launch(
        rank,
        self_sg,
        sg_ptrs,
        tmp_ptrs,
        out_ptrs,
        C,
        A,
        B,
        m,
        COUNTER,
        signal_state,
        stream=None,
    ):
        with CompilationContext.compile_hints(_compile_hints):
            return launch_hgemm_ar_kernel(
                rank,
                self_sg,
                sg_ptrs,
                tmp_ptrs,
                out_ptrs,
                C,
                A,
                B,
                m,
                COUNTER,
                signal_state,
                stream=stream,
            )

    _compile_cache = {}

    def _compile(
        rank,
        self_sg,
        sg_ptrs,
        tmp_ptrs,
        out_ptrs,
        C,
        A,
        B,
        m,
        COUNTER,
        signal_state,
        stream,
    ):
        with CompilationContext.compile_hints(_compile_hints):
            lookup_key = (rank, m)
            if _compile_cache.get(lookup_key, None) is None:
                _compile_cache[lookup_key] = flyc.compile(
                    launch_hgemm_ar_kernel,
                    rank,
                    self_sg,
                    sg_ptrs,
                    tmp_ptrs,
                    out_ptrs,
                    C,
                    A,
                    B,
                    m,
                    COUNTER,
                    signal_state,
                    stream,
                )
            return _compile_cache[lookup_key]

    _launch.compile = _compile

    return _launch
