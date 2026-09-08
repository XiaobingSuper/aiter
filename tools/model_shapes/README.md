# model_shapes

Collect the GEMM/MoE shapes a model actually issues under a given parallel
config, and turn them into aiter untuned CSVs.

Parallelism needs no modelling: `LinearBase.input_size`/`output_size` are already
per-partition, and `fused_moe`'s `w1`/`w2` already hold the local expert count.

## 1. collect

Start from `ATOM/recipes/<Model>.md` — it holds the launch command the model is
actually served with, env vars included. Those env vars select kernel paths, so
collecting without them yields shapes the deployment never issues. Notably
`ATOM_USE_TRITON_MOE=1` sends MoE through triton, in which case zero MoE records
is the correct outcome.

```bash
ATOM_SHAPE_DUMP=/tmp/shapes.jsonl \
PYTHONPATH=$AITER/tools/model_shapes:$PYTHONPATH \
python -m atom.entrypoints.openai_server --model <path> \
    --tp 8 --ep 8 --load_dummy --enforce-eager
```

`sitecustomize.py` installs the hooks in every process that inherits
`PYTHONPATH`, which is what reaches the spawned TP/EP ranks. With
`ATOM_SHAPE_DUMP` unset it does nothing, so the path can stay exported.

- `--load_dummy` skips weight loading; shapes come from the config, not the
  values.
- `--enforce-eager` is required — captured graphs hide the calls.
- Send one long prefill and a few decode requests, then stop the server with
  SIGINT so `atexit` runs. Each rank writes `shapes.rank<N>.jsonl`.

## 2. emit

```bash
python3 tools/model_shapes/emit_untuned.py '/tmp/shapes.rank*.jsonl' \
    --out aiter/configs --scenario throughput
```

Scenarios add an M sweep on top of the observed shapes (`observed`, `decode`,
`prefill`, `throughput`). Shapes already covered by the matching tuned CSV are
dropped; `--append` merges into the existing untuned CSVs instead of replacing
them. Standard library only, so this step runs outside the ROCm container.

An `[skip] unrouted kernel` warning means a quant path is missing from
`GEMM_ROUTES` and those shapes would go untuned — fix the route, don't ignore it.

## 3. tune

```bash
.github/scripts/op_tune.sh tune ck_gemm_a4w4_blockscale "--shape_grouped"
```

`--shape_grouped` keeps every kernel candidate for one shape on a single GPU, so
candidates are compared without cross-GPU timing variance. The MoE tuner does
not support it (a MoE shape spans two stages).
