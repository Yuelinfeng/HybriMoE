# Prefetch Utility Phenomenon Validation

This run tests whether cache hit ratio can stay high while prefetch action utility is low.

## Config

- cache_size: 24
- prefetch_width: 20
- history_window: 36
- transfer_latency: 4
- timely_ttl: 10

## Claim Check

- phenomenon cells flagged: 4
- flag rule: cache_hit_ratio >= 0.70, timely_useful_candidate_ratio <= 0.30, and either redundant candidate ratio >= 0.45, issued late ratio >= 0.20, or issued unused ratio >= 0.35

## History-Prefetch Summary

| workload | cache_hit_ratio | timely_useful_candidate_ratio | redundant_candidate_ratio | issued_late_ratio | issued_unused_ratio | stall_time | eviction_damage_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| stable_homogeneous | 0.891 | 0.003 | 0.892 | 0.247 | 0.678 | 801.0 | 0.221 |
| shifted_homogeneous | 0.847 | 0.007 | 0.831 | 0.161 | 0.713 | 1127.0 | 0.310 |
| stable_mixed | 0.686 | 0.031 | 0.690 | 0.210 | 0.492 | 2299.0 | 0.532 |
| shifted_mixed | 0.660 | 0.030 | 0.677 | 0.246 | 0.475 | 2455.0 | 0.524 |

## Interpretation Boundary

This is a controlled trace-level harness, not full model inference. It can validate metric semantics and a plausible causal pattern before expensive kTransformers instrumentation. A production claim still needs real router traces, model execution, and hardware transfer timing.
