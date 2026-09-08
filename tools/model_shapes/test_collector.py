# SPDX-License-Identifier: MIT
"""Standalone checks for the shape collector — no GPU, no aiter, no ATOM.

    python3 tools/model_shapes/test_collector.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))

_FAKE_AITER = """
import enum, sys, types
aiter = types.ModuleType("aiter")
class QuantType(enum.Enum):
    No = 0
    per_Tensor = 1
    per_Token = 2
    per_1x128 = 3
    per_1x32 = 4
aiter.QuantType = QuantType
aiter.dtypes = types.SimpleNamespace(i8="int8", bf16="torch.bfloat16")
sys.modules["aiter"] = aiter
"""


def _install_fakes():
    exec(compile(_FAKE_AITER, "<fake_aiter>", "exec"), {})
    sys.path.insert(0, HERE)
    import collector

    return collector


class FakeLinear:
    def __init__(self, n, k, qt, pdt):
        self.output_size, self.input_size = n, k
        self.quant_type, self.params_dtype = qt, pdt
        self.bias = None


class FakeMoE:
    def __init__(self, hidden, inter, experts, topk, quant_type):
        self.hidden_size = hidden
        self.intermediate_size_per_partition = inter
        self.local_num_experts, self.top_k = experts, topk
        self.params_dtype = "torch.bfloat16"
        self.activation = "silu"
        self.quant_method = types.SimpleNamespace(quant_type=quant_type)


class FakeModel:
    def __init__(self, layers):
        self._layers = layers

    def named_modules(self):
        return iter(self._layers)


def _fake_runner(collector, model):
    """ModelRunner appears only after install(), as in the real launcher."""
    mod = types.ModuleType("atom.model_engine.model_runner")

    class ModelRunner:
        def __init__(self, m):
            self.model = m
            self.warmed = False

        def warmup_model(self):
            self.warmed = True
            return "warm"

    mod.ModelRunner = ModelRunner
    sys.modules["atom.model_engine.model_runner"] = mod
    collector._LazyPatcher.poll()
    return ModelRunner(model)


def test_walk_records_shapes(tmp: str) -> None:
    collector = _install_fakes()
    from aiter import QuantType

    os.environ["ATOM_SHAPE_DUMP"] = os.path.join(tmp, "out.jsonl")
    collector.install()

    # per-partition sizes: ATOM already divided by TP/EP
    qkv = FakeLinear(7168, 2048, QuantType.No, "torch.bfloat16")
    model = FakeModel(
        [
            ("layers.0.qkv_proj", qkv),
            ("layers.0.o_proj", FakeLinear(1536, 7168, QuantType.per_1x128, "fp8")),
            ("layers.0.mlp.experts", FakeMoE(7168, 2048, 48, 8, QuantType.per_1x32)),
        ]
    )
    runner = _fake_runner(collector, model)

    # online quantization rewrites quant_type after construction; the walk must
    # see the rewritten value, which is why shapes are not read in __init__
    qkv.quant_type, qkv.params_dtype = QuantType.per_Token, "torch.float8_e4m3fn"

    assert runner.warmup_model() == "warm", "the original warmup must still run"
    assert runner.warmed

    records = [json.loads(x) for x in open(collector.dump())]
    gemms = {(r["N"], r["K"]): r for r in records if r["op"] == "gemm"}
    assert len(gemms) == 2, records
    assert gemms[(7168, 2048)]["kernel"] == "a8w8_bpreshuffle", (
        "post-online-quant kernel expected, got " + gemms[(7168, 2048)]["kernel"]
    )
    assert gemms[(1536, 7168)]["kernel"].startswith("a8w8_blockscale"), gemms
    assert "M" not in gemms[(7168, 2048)], "M must come from the emit sweep"

    moe = next(r for r in records if r["op"] == "moe")
    assert (moe["model_dim"], moe["inter_dim"]) == (7168, 2048), moe
    assert (moe["expert"], moe["topk"]) == (48, 8), moe
    assert moe["q_type"] == "QuantType.per_1x32", moe
    print("ok: pre-warmup walk records gemm and moe shapes after online quant")


def test_no_forward_patching() -> None:
    """ATOM compiles the model as one graph; touching forward breaks it."""
    src = open(os.path.join(HERE, "collector.py")).read()
    assert "cls.forward" not in src and ".forward =" not in src, (
        "collector must not patch any forward method: ATOM's backend asserts it "
        "is called exactly once, and a graph break makes it fire twice"
    )
    print("ok: no forward path is patched")


def test_import_hook_does_not_recurse() -> None:
    """Regression: patching from inside the import hook used to re-enter
    __import__ and recurse until the stack blew."""
    script = _FAKE_AITER + (
        """
sys.path.insert(0, %r)
import collector
collector.install("/dev/null")
import json, csv, decimal, gzip, fractions   # each one re-enters the hook
print("no recursion")
"""
        % HERE
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert "no recursion" in out.stdout, out.stderr[-2000:]
    assert "RecursionError" not in out.stderr, out.stderr[-2000:]
    print("ok: import hook does not recurse")


def test_sigterm_dumps(tmp: str) -> None:
    """atexit covers clean exits and SIGINT; SIGTERM needs its own handler."""
    out = os.path.join(tmp, "sig.jsonl")
    script = _FAKE_AITER + (
        """
import os, signal
sys.path.insert(0, %r)
import collector
collector.install(%r)
from aiter import QuantType
class L:
    output_size, input_size = 512, 256
    quant_type, params_dtype, bias = QuantType.No, "torch.bfloat16", None
class M:
    def named_modules(s): return iter([("l0", L())])
collector.walk_model(M())
os.kill(os.getpid(), signal.SIGTERM)
"""
        % (HERE, out)
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == -15, (proc.returncode, proc.stderr[-2000:])
    import glob

    written = glob.glob(os.path.join(tmp, "sig*.jsonl"))  # pid-suffixed per rank
    assert len(written) == 1, written
    with open(written[0]) as f:
        rec = json.loads(f.readline())
    assert (rec["N"], rec["K"]) == (512, 256), rec
    print("ok: SIGTERM still writes the dump")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_no_forward_patching()
        test_import_hook_does_not_recurse()
        test_sigterm_dumps(tmp)
        test_walk_records_shapes(tmp)
    print("all collector checks passed")
