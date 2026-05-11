# Router Trace Prefetch Utility Analysis

This report uses real router assignments captured from a model run, then replays cache/prefetch policies over that expert-access trace.

## Trace

- trace_path: C:\Users\Leave\Downloads\router_trace.jsonl
- model_type: deepseek_v2
- events_used: 260
- events_skipped: 0
- num_experts: 64
- max_assignments_per_event: 48

## Config

- cache_size: 48
- prefetch_width: 20
- history_window: 36
- transfer_latency: 4

## Summary

| policy | cache_hit_ratio | timely_useful_candidate_ratio | redundant_ratio | issued_late_ratio | issued_unused_ratio | stall_time | bytes_moved | eviction_damage_ratio | phenomenon_flag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| no_prefetch | 0.7802 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2332.0 | 583 | 0.0000 | False |
| history_prefetch | 0.7892 | 0.0124 | 0.9170 | 0.4307 | 0.1022 | 2153.0 | 637 | 0.6026 | True |
| utility_gate | 0.7802 | 0.0000 | 0.8562 | 0.0000 | 0.0000 | 2332.0 | 583 | 0.0000 | True |

## Delta: history_prefetch - no_prefetch

- cache_hit_delta: 0.0090
- stall_time_delta: -179.0
- bytes_moved_delta: 54
- demand_bytes_delta: -83

## Gate Check

- utility_gate_prefetch_issued: 0
- history_prefetch_issued: 137
- If the gate issues zero prefetches, it is only a conservative-abstention baseline, not yet a validated utility scheduler.

## Interpretation Boundary

This analysis uses real router assignments, but cache movement is still replayed by the analysis script. A full system claim needs the same counters attached to the live expert cache/load path.
