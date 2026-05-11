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

- cache_size: 40
- prefetch_width: 20
- history_window: 36
- transfer_latency: 4

## Summary

| policy | cache_hit_ratio | timely_useful_candidate_ratio | redundant_ratio | issued_late_ratio | issued_unused_ratio | stall_time | bytes_moved | eviction_damage_ratio | phenomenon_flag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| no_prefetch | 0.6912 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3276.0 | 819 | 0.0000 | False |
| history_prefetch | 0.6889 | 0.0162 | 0.8378 | 0.4752 | 0.2270 | 3135.0 | 973 | 0.5405 | False |
| utility_gate | 0.6912 | 0.0000 | 0.7467 | 0.0000 | 0.0000 | 3276.0 | 819 | 0.0000 | False |

## Delta: history_prefetch - no_prefetch

- cache_hit_delta: -0.0023
- stall_time_delta: -141.0
- bytes_moved_delta: 154
- demand_bytes_delta: -128

## Gate Check

- utility_gate_prefetch_issued: 0
- history_prefetch_issued: 282
- If the gate issues zero prefetches, it is only a conservative-abstention baseline, not yet a validated utility scheduler.

## Interpretation Boundary

This analysis uses real router assignments, but cache movement is still replayed by the analysis script. A full system claim needs the same counters attached to the live expert cache/load path.
