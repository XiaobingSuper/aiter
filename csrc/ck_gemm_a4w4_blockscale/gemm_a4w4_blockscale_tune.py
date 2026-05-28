# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import os

import pandas as pd
import torch
from gemm_a4w4_blockscale_common import kernels_list

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_A4W4, get_asm_dir
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import perftest
from aiter.utility import fp4_utils
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

# torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
torch.random.manual_seed(0)
SCALE_GROUP_SIZE = 32
block_shape = (128, 128)
TRITON_A4W4_PRESHUFFLE_KERNEL_NAME = "triton_gemm_afp4wfp4_preshuffle"


def checkClose(a, b, rtol=1e-3, atol=0.01):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    mask = ~isClose
    if isClose.all():
        return True
    else:
        percent = (a[mask]).numel() / a.numel()
        if percent > 0.01:
            return False
        else:
            return True


def run_torch(x, w, x_scales, w_scales, dtype):
    m, k = x.shape
    n, k = w.shape
    # First convert the x and w inputs to f32.
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


@perftest()
def kernel_instance_test(x, weight, x_scale, w_scale, out, kernel_id, splitK=0):
    aiter.gemm_a4w4_blockscale_tune(x, weight, x_scale, w_scale, out, kernel_id, splitK)
    return out


def run_gemm_a4w4_blockscale(x, weight, x_scale, w_scale, out, kernel_id, splitK):
    m, k = x.shape
    n, k = weight.shape
    res = aiter.gemm_a4w4_blockscale_tune(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    )
    return res[:m]


def run_gemm_a4w4_blockscale_asm(
    x,
    weight_shuffle,
    x_scale,
    w_scale,
    out,
    bias,
    kernelName,
    dtype=dtypes.bf16,
    bpreshuffle=True,
    splitK=None,
):
    m, k = x.shape
    # if splitK is not None and splitK > 0:
    #    out_reset = torch.zeros(
    #        out.shape[0], out.shape[1], dtype=dtype, device=torch.cuda.current_device()
    #    )
    #    out = out_reset
    res = aiter.gemm_a4w4_asm(
        x,
        weight_shuffle,
        x_scale,
        w_scale,
        out,
        kernelName,
        bias,
        bpreshuffle=bpreshuffle,
        log2_k_split=splitK,
    )
    return res[:m]


def run_gemm_a4w4_blockscale_triton(
    x,
    weight_shuffle,
    x_scale,
    w_scale,
    out,
    dtype=dtypes.bf16,
):
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle

    m, _ = x.shape
    n = out.shape[1]
    weight_shuffle = weight_shuffle.view(torch.uint8).view(n // 16, -1)
    if m < 32:
        x_scale = x_scale[:m].view(torch.uint8)
    else:
        x_scale = x_scale.view(torch.uint8).view(x_scale.shape[0] // 32, -1)
    w_scale = w_scale.view(torch.uint8).view(w_scale.shape[0] // 32, -1)
    res = gemm_afp4wfp4_preshuffle(
        x.view(torch.uint8),
        weight_shuffle,
        x_scale,
        w_scale,
        dtype=dtype,
        y=out,
    )
    return res[:m]


def generate_data(m, n, k, seed, device="cuda", dtype=dtypes.bf16):
    torch.manual_seed(seed)
    quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
    x_bf16 = torch.randn((m, k), dtype=dtype, device=device)  #
    w_bf16 = torch.randn((n, k), dtype=dtype, device=device)  #
    x, x_scales = quant_func(x_bf16, quant_dtype=dtypes.fp4x2, shuffle=False)
    _, x_scales_shuffle = quant_func(
        x_bf16, quant_dtype=dtypes.fp4x2, shuffle=True
    )
    w, w_scales = quant_func(w_bf16, quant_dtype=dtypes.fp4x2, shuffle=False)
    w_shuffle = shuffle_weight(w)
    w_scales_shuffle = fp4_utils.e8m0_shuffle(w_scales)
    out_ck = torch.empty((m + 255) // 256 * 256, n, dtype=dtype, device=device)
    out_gemm = torch.empty((m + 31) // 32 * 32, n, dtype=dtype, device=device)
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)
    bias_f32 = None
    return {
        "x": x,
        "w": w,
        "x_scales": x_scales,
        "w_scales": w_scales,
        "w_shuffle": w_shuffle,
        "x_scales_shuffle": x_scales_shuffle,
        "w_scales_shuffle": w_scales_shuffle,
        "out_ck": out_ck,
        "out_gemm": out_gemm,
        "bias_f32": bias_f32,
    }


class GemmA4W4BlockScaleTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A4W4}",
        "untune_file": "aiter/configs/a4w4_blockscale_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_A4W4",
    }

    def _clear_op_caches(self):
        from aiter.ops.gemm_op_a4w4 import get_GEMM_config

        get_GEMM_config.cache_clear()
        if hasattr(get_GEMM_config, "gemm_dict"):
            del get_GEMM_config.gemm_dict

    def _setup_specific_arguments(self):
        pass

    def run_config(self, args):
        from aiter.ops.gemm_op_a4w4 import gemm_a4w4, get_gemm_a4w4_backend
        from aiter.test_common import run_perftest, checkAllclose

        untunedf = self.untunedf
        results = []
        for i in range(len(untunedf)):
            row = untunedf.iloc[i]
            M = int(row["M"])
            N = int(row["N"])
            K = int(row["K"])
            shape_str = f"({M}, {N}, {K})"
            allowed_err_ratio, allowed_err_ratio_desc = (
                self._get_run_config_err_ratio_limit(row, args)
            )
            try:
                gd = generate_data(M, N, K, 0)
                x, w = gd["x"], gd["w"]
                x_scales, w_scales = gd["x_scales"], gd["w_scales"]
                w_shuffle = gd["w_shuffle"]
                x_scales_shuffle = gd["x_scales_shuffle"]
                w_scales_shuffle = gd["w_scales_shuffle"]
                x_scales_gemm = (
                    x_scales
                    if get_gemm_a4w4_backend(M, N, K) == "triton" and M < 32
                    else x_scales_shuffle
                )
                out, us = run_perftest(
                    gemm_a4w4,
                    x,
                    w_shuffle,
                    x_scales_gemm,
                    w_scales_shuffle,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                    num_rotate_args=1,
                )
                ref = run_torch(x, w, x_scales, w_scales, dtypes.bf16)
                err_ratio = checkAllclose(
                    out[:M].to(dtypes.bf16), ref, msg=f"run_config {shape_str}"
                )
                status = (
                    "ok"
                    if err_ratio <= allowed_err_ratio
                    else f"mismatch:err_ratio={err_ratio:.6g}(>{allowed_err_ratio_desc})"
                )
                results.append({"shape": shape_str, "e2e_us": us, "status": status})
            except Exception as e:
                results.append(
                    {"shape": shape_str, "e2e_us": -1, "status": f"error:{e}"}
                )
            finally:
                torch.cuda.empty_cache()
        return results

    def calculate(self, results, bpes=(1 / 2, 1 / 2, 2)):
        return super().calculate(results, bpes=bpes)

    def get_asm_kernels(self, file):
        if not os.path.exists(file):
            print(f"ASM kernel list file not exist: {file}")
            return {}
        df = pd.read_csv(file)
        shuffle_df = (
            df[df["bpreshuffle"] == 1]
            .reset_index()
            .sort_values(by=["tile_M", "tile_N", "splitK"])
        )
        kernel_dict = (
            shuffle_df.groupby(["tile_M", "tile_N", "splitK"])["knl_name"]
            .apply(list)
            .to_dict()
        )
        return kernel_dict

    def getKernelName(self, kernelId):
        # kernels_list is a dict keyed by kernel index; do not use len() bounds only.
        if kernelId is None or kernelId < 0 or kernelId not in kernels_list:
            return None
        return kernels_list[kernelId].name

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        useSplitK = args.splitK
        mp_num = args.mp
        shape_grouped = args.shape_grouped
        errRatio = args.errRatio
        from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

        if get_gfx() not in ["gfx950"]:
            print(f"tuning is not supported in this chip {get_gfx()}")
            return []
        gfx = self.get_gfx()
        cu_num = self.get_cu_num()
        task = []
        tasks_in_data = []

        ck_kernels_num = len(kernels_list)
        gemm_ck_keys = [
            "x",
            "w_shuffle",
            "x_scales_shuffle",
            "w_scales_shuffle",
            "out_ck",
        ]
        gemm_asm_keys = [
            "x",
            "w_shuffle",
            "x_scales_shuffle",
            "w_scales_shuffle",
            "out_gemm",
            "bias_f32",
        ]
        ref_keys = ["x", "w", "x_scales", "w_scales"]
        seed = 0
        for shape_idx in range(len(untunedf)):
            row = untunedf.iloc[shape_idx]
            # Native int keys so post_process grouping matches single-shape runs (no np.int64 vs int split).
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])

            total_kernel_nums = 0

            for kernel_idx in range(ck_kernels_num):
                kernel = kernels_list[kernel_idx]
                maxsplitK = (
                    aiter.compute_gemm_SplitK(
                        M,
                        N,
                        K,
                        kernel.MPerBLOCK,
                        kernel.NPerBLOCK,
                        kernel.KPerBLOCK,
                    )
                    if useSplitK
                    else 0
                )
                for splitK in range(maxsplitK + 1):
                    info = ((gfx, cu_num, M, N, K), kernel_idx, splitK, "")
                    task.append(
                        (
                            info,
                            generate_data,
                            (M, N, K, seed),
                            run_gemm_a4w4_blockscale,
                            (
                                gemm_ck_keys,
                                kernel_idx,
                                splitK,
                            ),
                            {
                                "num_warmup": 10,
                                "num_iters": 101,
                                "num_rotate_args": 1,
                            },
                            run_torch,
                            (
                                ref_keys,
                                dtypes.bf16,
                            ),
                            {},
                            None,
                            1e-2,
                            0.01,
                            None,
                            None,
                            ("out_ck",),
                        )
                    )
                    total_kernel_nums = total_kernel_nums + 1
            if N % 32 == 0 and K % 256 == 0:
                gemm_triton_keys = [
                    "x",
                    "w_shuffle",
                    "x_scales" if M < 32 else "x_scales_shuffle",
                    "w_scales_shuffle",
                    "out_gemm",
                ]
                triton_kernel_id = ck_kernels_num
                info = (
                    (gfx, cu_num, M, N, K),
                    triton_kernel_id,
                    0,
                    TRITON_A4W4_PRESHUFFLE_KERNEL_NAME,
                )
                task.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed),
                        run_gemm_a4w4_blockscale_triton,
                        (
                            gemm_triton_keys,
                            dtypes.bf16,
                        ),
                        {
                            "num_warmup": 10,
                            "num_iters": 101,
                            "num_rotate_args": 1,
                        },
                        run_torch,
                        (
                            ref_keys,
                            dtypes.bf16,
                        ),
                        {},
                        None,
                        1e-2,
                        0.01,
                        None,
                        None,
                        ("out_gemm",),
                    )
                )
                total_kernel_nums = total_kernel_nums + 1
            ### asm kernels
            asm_kernels_id = ck_kernels_num + 1
            asm_kernel_list_csv = f"{get_asm_dir()}/f4gemm/f4gemm_bf16_per1x32Fp4.csv"
            asm_kernels = self.get_asm_kernels(asm_kernel_list_csv)
            asm_tiles = [key for key in asm_kernels.keys()]
            for key in asm_tiles:
                tile_m, tile_n, splitk = key
                maxsplitK = (
                    aiter.compute_gemm_SplitK(M, N, K, tile_m, tile_n, 256)
                    if useSplitK
                    else 0
                )
                kernelName = asm_kernels.get((tile_m, tile_n, splitk), [])
                if len(kernelName) == 0:
                    print(f"no kernel name for ({tile_m}, {tile_n})!!!!")
                    continue
                if splitk == 0:
                    maxsplitK = 0
                for splitK in range(maxsplitK + 1):
                    kernel_name = kernelName[0]
                    info = ((gfx, cu_num, M, N, K), asm_kernels_id, splitK, kernel_name)
                    task.append(
                        (
                            info,
                            generate_data,
                            (M, N, K, seed),
                            run_gemm_a4w4_blockscale_asm,
                            (
                                gemm_asm_keys,
                                kernel_name,
                                dtypes.bf16,
                                True,
                                splitK,
                            ),
                            {
                                "num_warmup": 10,
                                "num_iters": 101,
                                "num_rotate_args": 1,
                            },
                            run_torch,
                            (
                                ref_keys,
                                dtypes.bf16,
                            ),
                            {},
                            None,
                            1e-2,
                            0.01,
                            None,
                            None,
                            ("out_gemm",),
                        )
                    )
                    asm_kernels_id = asm_kernels_id + 1

                    total_kernel_nums = total_kernel_nums + 1
            tasks_in_data.append((total_kernel_nums, ()))

        ret = []
        if task:
            ret = mp_tuner(
                task,
                tasks_in_data,
                mp_num,
                False,
                shape_grouped,
                errRatio,
                timeout=args.timeout,
                verbose=args.verbose,
            )
        return ret


if __name__ == "__main__":
    # key = [
    #    "cu_num",
    #    "M",
    #    "N",
    #    "K",
    # ]
    # resultList = key + [
    #    "kernelId",
    #    "splitK",
    #    "us",
    #    "kernelName",
    #    "errRatio",
    #    "tflops",
    #    "bw",
    # ]
    ## use default key and resultList
    tuner = GemmA4W4BlockScaleTuner(
        "GemmA4W4BlockScaleTuner", description="gen API for CK gemm a4w4 kernel"
    )

    args = tuner.parse_args()

    tuner.run(args, False)
