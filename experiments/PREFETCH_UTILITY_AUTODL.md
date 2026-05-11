# AutoDL Prefetch Utility Validation

This note is for running the phenomenon-validation harness on AutoDL. The
harness is intentionally lightweight: it does not load model weights or build
kTransformers. Use it to check metric semantics before adding real router-trace
instrumentation.

## Quick Run

From the HybriMoE repo root on AutoDL:

```bash
chmod +x experiments/run_prefetch_utility_autodl.sh
./experiments/run_prefetch_utility_autodl.sh
```

The default output path is:

```text
results/prefetch_utility_validation_autodl/<timestamp>/
```

Key files:

```text
all_runs_summary.csv
autodl_run_summary.json
seed_7/summary.csv
seed_7/summary.json
seed_7/README.md
```

## Common AutoDL Layout

If the repo is under `/root/autodl-tmp/HybriMoE`, run:

```bash
cd /root/autodl-tmp/HybriMoE
PREFETCH_UTILITY_OUT=/root/autodl-tmp/prefetch_utility_runs/$(date +%Y%m%d_%H%M%S) \
  ./experiments/run_prefetch_utility_autodl.sh
```

To run more seeds:

```bash
SEEDS="7 13 29 43 59" ./experiments/run_prefetch_utility_autodl.sh
```

To stress the cache/prefetch setting:

```bash
./experiments/run_prefetch_utility_autodl.sh \
  --cache-size 24 \
  --prefetch-width 20 \
  --history-window 36 \
  --transfer-latency 4
```

## What Counts As Evidence

For the current phenomenon claim, inspect the `history_prefetch` rows:

- high `cache_hit_ratio`
- low `timely_useful_candidate_ratio`
- high `redundant_prefetch_ratio`
- high `issued_late_prefetch_ratio` or `issued_unused_prefetch_ratio`
- higher `bytes_moved` or `stall_time` than a no-prefetch baseline in some cells

This harness can support a controlled phenomenon claim. It cannot prove a full
production-system claim until the same metrics are collected from real MoE
router traces and hardware timing.
