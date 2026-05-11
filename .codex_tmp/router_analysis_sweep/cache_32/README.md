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

- cache_size: 32
- prefetch_width: 20
- history_window: 36
- transfer_latency: 4

## Summary

| policy | cache_hit_ratio | timely_useful_candidate_ratio | redundant_ratio | issued_late_ratio | issued_unused_ratio | stall_time | bytes_moved | eviction_damage_ratio | phenomenon_flag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| no_prefetch | 0.5796 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4460.0 | 1115 | 0.0000 | False |
| history_prefetch | 0.5841 | 0.0226 | 0.7347 | 0.4496 | 0.2939 | 4167.0 | 1354 | 0.5339 | False |
| utility_gate | 0.5796 | 0.0000 | 0.6210 | 0.0000 | 0.0000 | 4460.0 | 1115 | 0.0000 | False |

## Delta: history_prefetch - no_prefetch

- cache_hit_delta: 0.0045
- stall_time_delta: -293.0
- bytes_moved_delta: 239
- demand_bytes_delta: -217

## Gate Check

- utility_gate_prefetch_issued: 0
- history_prefetch_issued: 456
- If the gate issues zero prefetches, it is only a conservative-abstention baseline, not yet a validated utility scheduler.

## Interpretation Boundary

This analysis uses real router assignments, but cache movement is still replayed by the analysis script. A full system claim needs the same counters attached to the live expert cache/load path.
