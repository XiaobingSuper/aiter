---
name: model-shape-tuning
description: End-to-end flow for tuning a new model on aiter — collect the GEMM/MoE shapes ATOM actually issues under a given TP/EP config with dummy weights, emit them into the untuned CSVs, and run the tuners. Use whenever asked to "tune model X", "collect shapes for X", "add configs for a new model", or when a model's shapes are not yet in aiter/configs.
argument-hint: [model name or path] [--tp N --ep N]
---

# Tuning a new model on aiter

Three steps: **collect → emit → tune**. Tooling lives in
`tools/model_shapes/`. Never hand-write shapes into the untuned CSVs — parallel
splitting, MoE local expert counts and quant dtypes are all easy to get wrong by
hand, and a wrong shape costs a full tuning run.

## Step 0 — start from the recipe

`ATOM/recipes/<Model>.md` carries the launch command the model is actually
served with: parallel config, quant flags and the env vars. **Read it first and
reuse that command verbatim** — do not invent a launch line, and do not drop its
env vars. They are not cosmetic; they select kernel paths, and collecting under
a different launch line yields shapes the deployment never issues.

The one that bites hardest: `ATOM_USE_TRITON_MOE=1` (required by e.g.
DeepSeek-V4-Pro) routes MoE through triton, so `aiter.fused_moe` is never called.
For such a model, collecting **zero MoE records is the correct result** — do not
"fix" it, and do not tune `untuned_fmoe.csv`. Others like `ATOM_MOE_GU_ITLV` and
`AITER_BF16_FP8_MOE_BOUND` likewise change what gets issued.

If no recipe exists for the model, ask the user for the launch command — shapes
differ per TP/EP and tuning the wrong config is wasted GPU time.

## Step 1 — collect

Take the recipe's command and add the collector env vars plus `--load_dummy`.
No requests are needed — shapes are read from the built model before warmup.

```bash
ATOM_SHAPE_DUMP=/tmp/<model>_shapes.jsonl \
PYTHONPATH=$AITER/tools/model_shapes:$PYTHONPATH \
<recipe env vars> \
python -m atom.entrypoints.openai_server --model <path> \
    <recipe flags> --load_dummy=xavier --enforce-eager
```

Some recipes set `PYTHONPATH` themselves — append to it, never overwrite, or the
hooks never load in the spawned ranks.

- `--load_dummy=xavier` skips reading the checkpoint; shapes come from the
  config. Use `=xavier`, not the bare flag: online quantization needs finite
  values to compute scales.
- Wait for `walked <n> layers` from every rank (`ATOM_SHAPE_DEBUG=1` prints it),
  then stop the server. Waiting for the server to be ready is not required.
- Stop it with `pkill -f "[o]penai_server"` — plain `pkill -f openai_server`
  matches the shell running it and kills that instead.

The collector walks the built model once, just before `ModelRunner.warmup_model`.
Do not move this earlier or later, and never into a forward: ATOM compiles the
model as one graph whose backend asserts a single call, so a forward hook kills
the engine with "VllmBackend can only be called once", while reading at layer
construction misses the quant_type that online quantization rewrites afterwards.

Nothing about parallelism needs modelling: `input_size`/`output_size` are
per-partition and `local_num_experts` is post-EP.

## Step 2 — emit

```bash
python3 tools/model_shapes/emit_untuned.py '/tmp/<model>_shapes.rank*.jsonl' \
    --out aiter/configs --scenario throughput --append
```

Scenarios (`decode`, `prefill`, `throughput`) supply the M sweep — M is not
collected, so pick the one matching how the model will be served. Shapes already in
the matching tuned CSV are dropped, so only genuinely new work is queued.

Report the per-file counts it prints. **Investigate any `[skip] unrouted
kernel`** — it means a quant path exists that `GEMM_ROUTES` does not cover, and
those shapes would silently go untuned.

## Step 3 — tune

```bash
.github/scripts/op_tune.sh tune <op_list> "--shape_grouped"
```

- **Always pass `--shape_grouped` for GEMM tuners.** It puts every kernel
  candidate for one shape on a single GPU, which is what makes the timings
  comparable across candidates; without it, candidates for the same shape are
  spread over GPUs and cross-GPU variance decides the winner.
- The MoE tuner (`gemm_moe_tune.py`) hardcodes `shape_grouped=False` because a
  MoE shape spans two stages. Do not try to force it on.
- Tuning is long. Run it in the background and report progress rather than
  blocking, and tune only the ops that step 2 actually added shapes to.

## Step 4 — land the configs

Tuned rows go to `aiter/configs/model_configs/`, not the canonical CSVs, and
they must not collide with shapes another model already contributed. Follow the
`aiter-config-shape` skill for that — a duplicate shape breaks `main` for
everyone after two PRs land.

## Checklist

- [ ] launch line taken from `ATOM/recipes/<Model>.md`, env vars included
- [ ] collected with `--load_dummy=xavier`; every rank logged `walked <n> layers`
- [ ] every rank's dump fed to `emit_untuned.py`
- [ ] no unrouted-kernel warnings left unexplained
- [ ] empty MoE records reconciled against the recipe's MoE backend
- [ ] `python3 tools/model_shapes/test_collector.py` passes if the tool was touched
- [ ] GEMM tuners ran with `--shape_grouped`
- [ ] configs landed per `aiter-config-shape`
