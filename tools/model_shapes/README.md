# model_shapes

Collect the GEMM/MoE shapes a model instantiates under a given parallel config,
and turn them into aiter untuned CSVs.

Parallelism needs no modelling: `LinearBase.input_size`/`output_size` are already
per-partition and `FusedMoE.local_num_experts` is already post-EP.

## How it captures shapes

The collector walks the built model once, immediately before
`ModelRunner.warmup_model`. That point is chosen deliberately:

- **Not from a forward.** ATOM compiles the model as a single graph and its
  backend asserts it is called exactly once. Anything injected into a forward
  path causes a dynamo graph break and the engine dies with
  `AssertionError: VllmBackend can only be called once`. `--enforce-eager` does
  not help — it disables CUDA graphs, not `torch.compile`.
- **Not at layer construction.** Online quantization (`--online_quant_config`)
  rewrites `quant_type` *after* the layers are built, so construction-time
  reads report bf16 for layers that actually run fp8.
- Before warmup, weights are loaded and quantization is settled, but nothing has
  been compiled yet.

The M/token dimension is not observable this way, and does not need to be:
`emit_untuned.py` sweeps it per scenario, which is what tuning wants anyway.

## 1. collect

Start from `ATOM/recipes/<Model>.md` — it holds the launch command the model is
actually served with, env vars included. Those env vars select kernel paths and
quantization, so collecting without them records shapes the deployment never
issues.

```bash
ATOM_SHAPE_DUMP=/tmp/shapes.jsonl \
PYTHONPATH=$AITER/tools/model_shapes:$PYTHONPATH \
<recipe env vars> \
python -m atom.entrypoints.openai_server --model <path> \
    <recipe flags> --load_dummy=xavier --enforce-eager
```

`sitecustomize.py` installs the collector in every process that inherits
`PYTHONPATH`, which is how the spawned TP/EP ranks are reached. With
`ATOM_SHAPE_DUMP` unset it does nothing, so the path can stay exported.

- `--load_dummy=xavier` skips reading the checkpoint (finite values, so online
  quantization can still compute scales). Shapes come from the config.
- No requests are needed. Each rank writes `shapes.rank<N>.jsonl` before warmup;
  once the log shows `walked <n> layers` for every rank, the run is done and the
  server can be stopped.
- Set `ATOM_SHAPE_DEBUG=1` to see the walk count and any swallowed failure.

Stopping the server: `pkill -f openai_server` **kills the shell running it**,
because `-f` matches that shell's own command line. Use `pkill -f
"[o]penai_server"`.

## 2. emit

```bash
python3 tools/model_shapes/emit_untuned.py '/tmp/shapes.rank*.jsonl' \
    --out aiter/configs --scenario throughput
```

Scenarios (`decode`, `prefill`, `throughput`) supply the M sweep. Shapes already
covered by the matching tuned CSV are dropped; `--append` merges into the
existing untuned CSVs instead of replacing them. Standard library only, so this
step runs outside the ROCm container.

An `[skip] unrouted kernel` warning means a quant path is missing from
`GEMM_ROUTES` and those shapes would go untuned — fix the route, don't ignore it.

## 3. tune

```bash
.github/scripts/op_tune.sh tune ck_gemm_a8w8_bpreshuffle "--shape_grouped"
```

`--shape_grouped` keeps every kernel candidate for one shape on a single GPU, so
candidates are compared without cross-GPU timing variance. The MoE tuner does
not support it (a MoE shape spans two stages).

## Tests

```bash
python3 tools/model_shapes/test_collector.py
```

No GPU, no aiter, no ATOM — fakes stand in for all three. Covers the two
failures that cost the most time to find: the forward-patching graph break and
an import hook that recursed on itself.

## Known limitation

`use_g1u1`, `doweight_stage1` and `gate_mode` are not knowable before the first
`fused_moe` call, so MoE rows are emitted with the tuner's defaults (1, 0,
none). Override them by hand if the model needs something else.
