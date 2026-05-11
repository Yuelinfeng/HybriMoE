#!/usr/bin/env bash
set -euo pipefail

# Run the lightweight prefetch-utility validation harness on AutoDL.
# This script intentionally avoids CUDA/model dependencies; it validates the
# metric semantics and phenomenon shape before full kTransformers integration.

ROOT="${HYBRIMOE_ROOT:-$(pwd)}"
OUT_ROOT="${PREFETCH_UTILITY_OUT:-$ROOT/results/prefetch_utility_validation_autodl/$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEEDS="${SEEDS:-7 13 29}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"

echo "[prefetch-utility] root: $ROOT"
echo "[prefetch-utility] output: $OUT_ROOT"
echo "[prefetch-utility] python: $($PYTHON_BIN --version 2>&1)"

for seed in $SEEDS; do
  run_dir="$OUT_ROOT/seed_$seed"
  echo "[prefetch-utility] running seed=$seed -> $run_dir"
  "$PYTHON_BIN" experiments/prefetch_utility_validation.py \
    --seed "$seed" \
    --output-dir "$run_dir" \
    "$@"
done

"$PYTHON_BIN" - "$OUT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
rows = []
for csv_path in sorted(out_root.glob("seed_*/summary.csv")):
    seed = csv_path.parent.name.replace("seed_", "")
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["seed"] = seed
            rows.append(row)

summary_csv = out_root / "all_runs_summary.csv"
if rows:
    fieldnames = ["seed"] + [name for name in rows[0].keys() if name != "seed"]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

flagged = [
    row for row in rows
    if row.get("policy") == "history_prefetch" and row.get("phenomenon_flag", "").lower() == "true"
]
payload = {
    "total_rows": len(rows),
    "history_prefetch_flagged_cells": len(flagged),
    "summary_csv": str(summary_csv),
}
(out_root / "autodl_run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "[prefetch-utility] done"
