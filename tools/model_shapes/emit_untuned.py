# SPDX-License-Identifier: MIT
"""Turn collected shape records into aiter untuned CSVs.

    python3 tools/model_shapes/emit_untuned.py '/tmp/shapes.jsonl*' \
        --scenario throughput --out aiter/configs

Shapes already present in the corresponding tuned CSV are dropped, so a rerun
after a model change only tunes what is actually new. Standard library only, so
it runs outside the ROCm container.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

# The collector reads weight shapes at construction, so M is not observed:
# every scenario supplies its own sweep. Pick by how the model will be served.
SCENARIOS: dict[str, list[int]] = {
    "decode": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "throughput": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    "prefill": [512, 1024, 2048, 4096, 8192, 16384],
}

# kernel tag (set by collector) -> (untuned file, tuned file, extra columns).
# A None value means "take it from the record's params_dtype".
GEMM_ROUTES: dict[str, tuple[str, str, dict]] = {
    "bf16": (
        "bf16_untuned_gemm.csv",
        "bf16_tuned_gemm.csv",
        {
            "bias": False,
            "dtype": "torch.bfloat16",
            "outdtype": "torch.bfloat16",
            "scaleAB": False,
            "bpreshuffle": False,
        },
    ),
    "bf16_scaled": (
        "bf16_untuned_gemm.csv",
        "bf16_tuned_gemm.csv",
        {
            "bias": False,
            "dtype": "torch.bfloat16",
            "outdtype": "torch.bfloat16",
            "scaleAB": True,
            "bpreshuffle": False,
        },
    ),
    "a8w8": ("a8w8_untuned_gemm.csv", "a8w8_tuned_gemm.csv", {"q_dtype_w": None}),
    "a8w8_bpreshuffle": (
        "a8w8_bpreshuffle_untuned_gemm.csv",
        "a8w8_bpreshuffle_tuned_gemm.csv",
        {"q_dtype_w": None},
    ),
    "a8w8_blockscale": (
        "a8w8_blockscale_untuned_gemm.csv",
        "a8w8_blockscale_tuned_gemm.csv",
        {},
    ),
    "a8w8_blockscale_bpreshuffle": (
        "a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
        "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
        {},
    ),
    "a4w4_blockscale": (
        "a4w4_blockscale_untuned_gemm.csv",
        "a4w4_blockscale_tuned_gemm.csv",
        {},
    ),
}

MOE_COLS = [
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
]

Row = dict[str, object]


def load(patterns: list[str]) -> list[Row]:
    records: list[Row] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)) or ([pattern] if Path(pattern).exists() else []):
            with open(path) as f:
                records += [json.loads(line) for line in f if line.strip()]
    if not records:
        raise SystemExit("no shape records found")
    return records


def dedup(rows: list[Row]) -> list[Row]:
    seen, out = set(), []
    for row in rows:
        key = tuple(str(v) for v in row.values())
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def with_m(rows: list[Row], m_col: str, sweep: list[int]) -> list[Row]:
    """Cross every distinct weight shape with the scenario's M sweep.

    m_col goes first so the emitted column order matches aiter's untuned CSVs.
    """
    return dedup([{m_col: m, **row} for row in rows for m in sweep])


def drop_tuned(rows: list[Row], tuned: Path) -> tuple[list[Row], int]:
    if not rows or not tuned.exists():
        return rows, 0
    with open(tuned, newline="") as f:
        reader = csv.DictReader(f)
        keys = [c for c in rows[0] if c in (reader.fieldnames or [])]
        if not keys:
            return rows, 0
        done = {tuple(str(r[k]).strip() for k in keys) for r in reader}
    keep = [r for r in rows if tuple(str(r[k]) for k in keys) not in done]
    return keep, len(rows) - len(keep)


def write(rows: list[Row], path: Path, append: bool) -> None:
    if append and path.exists():
        with open(path, newline="") as f:
            rows = dedup([{k: v.strip() for k, v in r.items()} for r in csv.DictReader(f)] + rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def emit_gemm(records: list[Row], out: Path, sweep: list[int], append: bool) -> None:
    by_kernel: dict[str, list[Row]] = {}
    for rec in records:
        if rec.get("op") == "gemm":
            by_kernel.setdefault(rec["kernel"], []).append(rec)

    for kernel, group in sorted(by_kernel.items()):
        route = GEMM_ROUTES.get(kernel)
        if route is None:
            print(f"[skip] unrouted kernel {kernel}: {len(group)} shapes")
            continue
        untuned_name, tuned_name, extra = route
        rows = [
            {
                "N": rec["N"],
                "K": rec["K"],
                **{
                    col: (rec["params_dtype"] if val is None else val)
                    for col, val in extra.items()
                },
            }
            for rec in group
        ]
        rows = with_m(dedup(rows), "M", sweep)
        rows, skipped = drop_tuned(rows, out / tuned_name)
        write(rows, out / untuned_name, append)
        print(f"{untuned_name}: +{len(rows)} shapes ({skipped} already tuned)")


def emit_moe(records: list[Row], out: Path, sweep: list[int], append: bool) -> None:
    groups: dict[bool, list[Row]] = {}
    for rec in records:
        if rec.get("op") == "moe":
            groups.setdefault(bool(rec.get("gate_mode")), []).append(rec)

    for grouped, group in sorted(groups.items()):
        cols = MOE_COLS + (["gate_mode"] if grouped else [])
        rows = dedup([{c: rec[c] for c in cols} for rec in group])
        rows = with_m(rows, "token", sweep)
        name = "untuned_grouped_fmoe.csv" if grouped else "untuned_fmoe.csv"
        tuned = "tuned_grouped_fmoe.csv" if grouped else "tuned_fmoe.csv"
        rows, skipped = drop_tuned(rows, out / tuned)
        if grouped:
            rows = [{"cu_num": "", **r} for r in rows]
        write(rows, out / name, append)
        print(f"{name}: +{len(rows)} shapes ({skipped} already tuned)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+", help="jsonl files from collector.py")
    ap.add_argument("--out", default=Path("aiter/configs"), type=Path)
    ap.add_argument("--scenario", default="throughput", choices=sorted(SCENARIOS))
    ap.add_argument(
        "--append",
        action="store_true",
        help="merge into the existing untuned CSVs instead of replacing them",
    )
    args = ap.parse_args()

    records = load(args.records)
    sweep = SCENARIOS[args.scenario]
    emit_gemm(records, args.out, sweep, args.append)
    emit_moe(records, args.out, sweep, args.append)


if __name__ == "__main__":
    main()
