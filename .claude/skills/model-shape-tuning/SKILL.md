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

## What the user must supply

Model path and the parallel config to tune for (`--tp`, `--ep`, and whether DP
attention is on). Everything else is derived. If the parallel config is not
given, ask — shapes differ per TP/EP and tuning the wrong one is wasted GPU time.

## Step 1 — collect

```bash
ATOM_SHAPE_DUMP=/tmp/<model>_shapes.jsonl \
PYTHONPATH=$AITER/tools/model_shapes:$PYTHONPATH \
python -m atom.entrypoints.api_server --model <path> \
    --tp 8 --ep 8 --load_dummy --enforce-eager
```

- `--load_dummy` skips weight loading. Shapes come from the config, not the
  values, so dummy weights are always correct here and save minutes per run.
- `--enforce-eager` is required: captured graphs hide the calls from the hooks.
- Drive both regimes before shutting down — one long prefill request and a few
  decode steps — or the M coverage will be one-sided.
- Shut down with SIGINT so `atexit` writes the dump. Each rank writes
  `<name>.rank<N>.jsonl`.

The collector patches `LinearBase.forward` (every dense GEMM in ATOM dispatches
there) and `aiter.fused_moe`. Nothing needs to be modelled about parallelism:
`input_size`/`output_size` are already per-partition and `w1`/`w2` already hold
the local expert count.

If the run prints nothing and the dump is empty, the hooks did not attach —
check that `collector` was imported before ATOM and that eager mode is on.

## Step 2 — emit

```bash
python3 tools/model_shapes/emit_untuned.py '/tmp/<model>_shapes.rank*.jsonl' \
    --out aiter/configs --scenario throughput --append
```

Scenarios (`observed`, `decode`, `prefill`, `throughput`) add an M sweep on top
of the observed shapes; pick by how the model will be served. Shapes already in
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

- [ ] parallel config confirmed with the user
- [ ] collected with `--load_dummy --enforce-eager`, both prefill and decode driven
- [ ] every rank's dump fed to `emit_untuned.py`
- [ ] no unrouted-kernel warnings left unexplained
- [ ] GEMM tuners ran with `--shape_grouped`
- [ ] configs landed per `aiter-config-shape`
