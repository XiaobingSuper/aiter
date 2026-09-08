# SPDX-License-Identifier: MIT
"""Runtime GEMM/MoE shape collector for ATOM + aiter.

Enable by pointing ATOM_SHAPE_DUMP at an output file and making this package
importable, e.g.

    ATOM_SHAPE_DUMP=/tmp/shapes.jsonl \
    PYTHONPATH=/path/to/aiter/tools/model_shapes:$PYTHONPATH \
    python -c "import collector; collector.install()" ...

or simply `import collector; collector.install()` at the top of the launcher.
Patches are installed lazily on module import, so install() may run before ATOM
is imported.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading

_records: dict[tuple, dict] = {}
_lock = threading.Lock()
_installed = False


def _emit(key: tuple, payload: dict) -> None:
    with _lock:
        rec = _records.get(key)
        if rec is None:
            payload["count"] = 1
            _records[key] = payload
        else:
            rec["count"] += 1


def _dtype(t) -> str:
    return str(t)


def _enum(v) -> str:
    return getattr(v, "name", None) and f"{type(v).__name__}.{v.name}" or str(v)


# ---------------------------------------------------------------- linear hook


def _linear_kernel(quant_type, params_dtype) -> str:
    """Mirror LinearBase.forward dispatch so emit only has to route."""
    from aiter import QuantType, dtypes

    q = quant_type.value
    if q == QuantType.No.value:
        return "bf16"
    if q == QuantType.per_Tensor.value:
        return "bf16_scaled"
    if q == QuantType.per_Token.value:
        return "a8w8" if params_dtype == dtypes.i8 else "a8w8_bpreshuffle"
    if q == QuantType.per_1x128.value:
        from atom import envs

        return (
            "a8w8_blockscale_bpreshuffle"
            if envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
            else "a8w8_blockscale"
        )
    if q == QuantType.per_1x32.value:
        return "a4w4_blockscale"
    return f"unknown:{quant_type}"


def _wrap_linear_forward(orig):
    def forward(self, x, *args, **kwargs):
        try:
            M = x.numel() // x.shape[-1]
            _emit(
                (
                    "gemm",
                    M,
                    self.output_size,
                    self.input_size,
                    self.quant_type.value,
                    str(self.params_dtype),
                ),
                {
                    "op": "gemm",
                    "M": M,
                    "N": self.output_size,
                    "K": self.input_size,
                    "kernel": _linear_kernel(self.quant_type, self.params_dtype),
                    "params_dtype": _dtype(self.params_dtype),
                    "bias": getattr(self, "bias", None) is not None,
                    "layer": getattr(self, "prefix", type(self).__name__),
                },
            )
        except Exception:  # never break the model because of collection
            pass
        return orig(self, x, *args, **kwargs)

    return forward


# ------------------------------------------------------------------ moe hook


_MOE_POS_ARGS = ("expert_mask", "activation", "quant_type", "doweight_stage1")


def _wrap_fused_moe(orig):
    def fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, *args, **kwargs):
        try:
            from aiter import QuantType
            from aiter.fused_moe import get_inter_dim

            call = dict(zip(_MOE_POS_ARGS, args)) | kwargs

            E, model_dim, inter_dim = get_inter_dim(tuple(w1.shape), tuple(w2.shape))
            token = hidden_states.numel() // hidden_states.shape[-1]
            topk = topk_ids.shape[-1]
            quant_type = call.get("quant_type", QuantType.No)
            act = call.get("activation")
            use_g1u1 = int(w1.shape[1] == inter_dim * 2)
            quantized = quant_type.value != QuantType.No.value
            _emit(
                (
                    "moe",
                    token,
                    model_dim,
                    inter_dim,
                    E,
                    topk,
                    quant_type.value,
                    str(w1.dtype),
                ),
                {
                    "op": "moe",
                    "token": token,
                    "model_dim": model_dim,
                    "inter_dim": inter_dim,
                    "expert": E,
                    "topk": topk,
                    "act_type": _enum(act) if act is not None else "ActivationType.Silu",
                    "dtype": _dtype(hidden_states.dtype),
                    "q_dtype_a": _dtype(w1.dtype if quantized else hidden_states.dtype),
                    "q_dtype_w": _dtype(w1.dtype),
                    "q_type": _enum(quant_type),
                    "use_g1u1": use_g1u1,
                    "doweight_stage1": int(bool(call.get("doweight_stage1", False))),
                    "gate_mode": call.get("gate_mode") or "",
                    "shared": call.get("shared_w1") is not None,
                },
            )
        except Exception:
            pass
        return orig(hidden_states, w1, w2, topk_weight, topk_ids, *args, **kwargs)

    return fused_moe


# --------------------------------------------------------- patch machinery


def _patch_linear(mod) -> None:
    cls = getattr(mod, "LinearBase", None)
    if cls is not None and not getattr(cls.forward, "_shape_hooked", False):
        cls.forward = _wrap_linear_forward(cls.forward)
        cls.forward._shape_hooked = True


def _rebind_fused_moe() -> None:
    """Replace every already-imported reference to aiter's fused_moe."""
    import aiter.fused_moe as fm

    orig = getattr(fm, "_shape_orig_fused_moe", None) or fm.fused_moe
    fm._shape_orig_fused_moe = orig
    wrapped = _wrap_fused_moe(orig)
    wrapped._shape_hooked = True
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            if getattr(mod, "fused_moe", None) is orig:
                mod.fused_moe = wrapped
        except Exception:
            continue
    fm.fused_moe = wrapped


class _LazyPatcher:
    """Apply patches as soon as their target modules show up in sys.modules."""

    TARGETS = {
        "atom.model_ops.linear": _patch_linear,
        "atom.model_ops.moe": lambda mod: _rebind_fused_moe(),
        "atom.model_ops.fused_moe.modular_kernel": lambda mod: _rebind_fused_moe(),
    }

    @classmethod
    def poll(cls):
        for name, fn in cls.TARGETS.items():
            mod = sys.modules.get(name)
            if mod is not None:
                try:
                    fn(mod)
                except Exception:
                    pass


def _install_import_hook() -> None:
    import builtins

    orig_import = builtins.__import__

    def hooked(name, *args, **kwargs):
        mod = orig_import(name, *args, **kwargs)
        _LazyPatcher.poll()
        return mod

    builtins.__import__ = hooked


def dump(path: str | None = None) -> str:
    path = path or os.environ.get("ATOM_SHAPE_DUMP", "atom_shapes.jsonl")
    with _lock:
        records = list(_records.values())
    target = _rank_path(path)
    tmp = f"{target}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, target)
    return target


def _rank_path(path: str) -> str:
    """Each TP/EP rank writes its own file; emit_untuned merges them."""
    rank = (
        os.environ.get("RANK")
        or os.environ.get("LOCAL_RANK")
        or os.environ.get("ATOM_RANK")
    )
    if rank is None:
        return path
    base, _, ext = path.rpartition(".")
    return f"{base}.rank{rank}.{ext}" if base else f"{path}.rank{rank}"


def install(path: str | None = None) -> None:
    global _installed
    if _installed:
        return
    _installed = True
    if path:
        os.environ["ATOM_SHAPE_DUMP"] = path
    _LazyPatcher.poll()
    _install_import_hook()
    atexit.register(dump)


if os.environ.get("ATOM_SHAPE_DUMP"):
    install()
