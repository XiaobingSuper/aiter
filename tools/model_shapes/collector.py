# SPDX-License-Identifier: MIT
"""Collect the GEMM/MoE weight shapes an ATOM model instantiates.

Enable by pointing ATOM_SHAPE_DUMP at an output file and putting this directory
on PYTHONPATH; sitecustomize.py then installs it into every process, including
the spawned TP/EP ranks.

Shapes are read when layers are *constructed*, never from forward. ATOM compiles
the model as a single graph and its backend asserts it is called exactly once,
so anything injected into a forward path causes a dynamo graph break and the
engine dies with "VllmBackend can only be called once". Construction happens
well before compilation, costs nothing per step, and already carries everything
that depends on the parallel config: LinearBase.input_size/output_size are
per-partition and FusedMoE.local_num_experts is post-EP.

The M/token dimension is not observable this way; emit_untuned.py sweeps it per
scenario, which is what tuning needs anyway.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import sys
import threading
import time

_DEBUG = bool(os.environ.get("ATOM_SHAPE_DEBUG"))
_FLUSH_INTERVAL = 2.0

_records: dict[tuple, dict] = {}
_lock = threading.Lock()
_installed = False
_last_flush = 0.0
_warned: set[str] = set()


def _warn(where: str, exc: BaseException) -> None:
    """Report a swallowed failure once per site; silent unless ATOM_SHAPE_DEBUG."""
    if not _DEBUG or where in _warned:
        return
    _warned.add(where)
    print(f"[model_shapes] {where} failed: {exc!r}", file=sys.stderr, flush=True)


def _emit(key: tuple, payload: dict) -> None:
    global _last_flush
    with _lock:
        if key in _records:
            return
        _records[key] = payload
        due = time.monotonic() - _last_flush > _FLUSH_INTERVAL
        if due:
            _last_flush = time.monotonic()
    if due:
        # keep the dump usable even if the engine dies later in startup
        dump()


def _dtype(t) -> str:
    return str(t)


def _enum(v) -> str:
    name = getattr(v, "name", None)
    return f"{type(v).__name__}.{name}" if name else str(v)


# --------------------------------------------------------------- linear layers


def _linear_kernel(quant_type, params_dtype) -> str:
    """Mirror LinearBase.forward's dispatch so emit only has to route."""
    from aiter import QuantType, dtypes

    q = quant_type.value
    if q == QuantType.No.value:
        return "bf16"
    if q == QuantType.per_Tensor.value:
        return "bf16_scaled"
    if q == QuantType.per_Token.value:
        return "a8w8" if params_dtype == dtypes.i8 else "a8w8_bpreshuffle"
    if q == QuantType.per_1x128.value:
        try:
            from atom import envs

            preshuffle = envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
        except Exception:  # losing the whole shape would be worse
            preshuffle = False
        return "a8w8_blockscale_bpreshuffle" if preshuffle else "a8w8_blockscale"
    if q == QuantType.per_1x32.value:
        return "a4w4_blockscale"
    return f"unknown:{quant_type}"


def _record_linear(layer, name: str) -> None:
    N, K = layer.output_size, layer.input_size
    _emit(
        ("gemm", N, K, layer.quant_type.value, str(layer.params_dtype)),
        {
            "op": "gemm",
            "N": N,
            "K": K,
            "kernel": _linear_kernel(layer.quant_type, layer.params_dtype),
            "params_dtype": _dtype(layer.params_dtype),
            "bias": getattr(layer, "bias", None) is not None,
            "layer": name,
        },
    )


# ------------------------------------------------------------------ moe layers


def _moe_quant(layer):
    """QuantType lives on the quant method, which is built during __init__."""
    from aiter import QuantType

    method = getattr(layer, "quant_method", None)
    return getattr(method, "quant_type", QuantType.No)


def _record_moe(layer, name: str) -> None:
    quant_type = _moe_quant(layer)
    model_dim = layer.hidden_size
    inter_dim = layer.intermediate_size_per_partition
    expert = layer.local_num_experts
    dtype = layer.params_dtype
    quantized = quant_type.value != 0
    _emit(
        ("moe", model_dim, inter_dim, expert, layer.top_k, quant_type.value),
        {
            "op": "moe",
            "model_dim": model_dim,
            "inter_dim": inter_dim,
            "expert": expert,
            "topk": layer.top_k,
            "act_type": _enum(getattr(layer, "activation", "")) or "ActivationType.Silu",
            "dtype": _dtype(dtype),
            "q_dtype_a": _dtype(dtype),
            "q_dtype_w": _dtype(dtype),
            "q_type": _enum(quant_type),
            # not knowable at construction; the tuner's defaults
            "use_g1u1": 1,
            "doweight_stage1": 0,
            "gate_mode": "",
            "layer": name,
            "quantized": quantized,
        },
    )


# ----------------------------------------------------------- patch machinery


def walk_model(model) -> int:
    """Record every linear/MoE layer of a fully built model.

    Identification is by attributes rather than class, so a model that wraps or
    subclasses the stock layers is still covered.
    """
    seen = 0
    for name, mod in model.named_modules():
        try:
            if hasattr(mod, "quant_type") and hasattr(mod, "input_size"):
                _record_linear(mod, name)
                seen += 1
            elif hasattr(mod, "local_num_experts") and hasattr(mod, "top_k"):
                _record_moe(mod, name)
                seen += 1
        except Exception as exc:
            _warn(f"record {name}", exc)
    return seen


def _patch_runner(mod) -> None:
    """Walk right before warmup: weights are loaded, online quantization has
    already rewritten quant_type, and torch.compile has not run yet. Reading
    shapes at layer construction would miss online quant; reading them from a
    forward would break ATOM's single-graph compilation."""
    cls = getattr(mod, "ModelRunner", None)
    if cls is None or getattr(cls.warmup_model, "_shape_hooked", False):
        return
    orig = cls.warmup_model

    def warmup_model(self, *args, **kwargs):
        try:
            count = walk_model(self.model)
            path = dump()
            if _DEBUG:
                print(
                    f"[model_shapes] walked {count} layers -> {path}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:
            _warn("model walk", exc)
        return orig(self, *args, **kwargs)

    warmup_model._shape_hooked = True
    cls.warmup_model = warmup_model


class _LazyPatcher:
    """Apply patches as soon as their target modules show up in sys.modules."""

    TARGETS = {
        "atom.model_engine.model_runner": _patch_runner,
    }

    _polling = False

    @classmethod
    def poll(cls):
        if cls._polling:  # the hook runs on every import, including nested ones
            return
        cls._polling = True
        try:
            cls._poll()
        finally:
            cls._polling = False

    @classmethod
    def _poll(cls):
        for name, fn in cls.TARGETS.items():
            mod = sys.modules.get(name)
            if mod is not None:
                try:
                    fn(mod)
                except Exception as exc:
                    _warn(f"patch {name}", exc)


def _install_import_hook() -> None:
    import builtins

    orig_import = builtins.__import__

    def hooked(name, *args, **kwargs):
        mod = orig_import(name, *args, **kwargs)
        _LazyPatcher.poll()
        return mod

    builtins.__import__ = hooked


# ------------------------------------------------------------------- output


def _rank_path(path: str) -> str:
    """Each TP/EP rank writes its own file; emit_untuned merges them."""
    # ATOM does not export RANK to its spawned workers, so fall back to the pid:
    # without a suffix every rank overwrites the same file, which silently loses
    # shapes whenever the ranks are not symmetric (DP/PP, uneven EP).
    rank = (
        os.environ.get("RANK")
        or os.environ.get("LOCAL_RANK")
        or os.environ.get("ATOM_RANK")
        or str(os.getpid())
    )
    base, _, ext = path.rpartition(".")
    return f"{base}.rank{rank}.{ext}" if base else f"{path}.rank{rank}"


def dump(path: str | None = None) -> str:
    path = path or os.environ.get("ATOM_SHAPE_DUMP", "atom_shapes.jsonl")
    with _lock:
        records = list(_records.values())
    if not records:
        return ""
    target = _rank_path(path)
    tmp = f"{target}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        os.replace(tmp, target)
    except OSError as exc:  # atexit and signal handlers must not raise
        print(f"[model_shapes] could not write {target}: {exc}", file=sys.stderr)
        return ""
    return target


def _install_signal_handlers() -> None:
    """atexit covers a clean exit and SIGINT; SIGTERM would otherwise lose it."""
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            previous = signal.getsignal(sig)
        except (ValueError, OSError):
            continue

        def handler(signum, frame, _prev=previous):
            try:
                dump()
            except Exception as exc:
                _warn("signal dump", exc)
            if callable(_prev) and _prev not in (signal.SIG_IGN, signal.SIG_DFL):
                return _prev(signum, frame)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):  # not the main thread
            continue


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
    _install_signal_handlers()
    if _DEBUG:
        print(f"[model_shapes] collector installed in pid {os.getpid()}", file=sys.stderr)


if os.environ.get("ATOM_SHAPE_DUMP"):
    install()
