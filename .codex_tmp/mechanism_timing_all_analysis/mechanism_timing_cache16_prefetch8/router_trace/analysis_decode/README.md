# Router Trace Prefetch Utility Analysis

This report uses real router assignments captured from a model run, then replays cache/prefetch policies over that expert-access trace.

## Trace

- trace_path: /root/autodl-tmp/hybrimoe_live_trace/mechanism_timing_cache16_prefetch8/router_trace/router_trace.jsonl
- model_type: deepseek_v2
- events_used: 572
- events_skipped: 26
- num_experts: 64
- max_assignments_per_event: 6

## Config

- cache_size: 16
- prefetch_width: 20
- history_window: 36
- transfer_latency: 4

## Summary

| policy | cache_hit_ratio | timely_useful_candidate_ratio | redundant_ratio | issued_late_ratio | issued_unused_ratio | stall_time | bytes_moved | eviction_damage_ratio | phenomenon_flag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| no_prefetch | 0.2142 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10788.0 | 2697 | 0.0000 | False |
| history_prefetch | 0.2343 | 0.0269 | 0.4259 | 0.3527 | 0.5098 | 9441.0 | 4074 | 0.5761 | False |
| utility_gate | 0.2142 | 0.0000 | 0.3958 | 0.0000 | 0.0000 | 10788.0 | 2697 | 0.0000 | False |

## Delta: history_prefetch - no_prefetch

- cache_hit_delta: 0.0201
- stall_time_delta: -1347.0
- bytes_moved_delta: 1377
- demand_bytes_delta: -857

## Gate Check

- utility_gate_prefetch_issued: 0
- history_prefetch_issued: 2234
- If the gate issues zero prefetches, it is only a conservative-abstention baseline, not yet a validated utility scheduler.

## Interpretation Boundary

This analysis uses real router assignments, but cache movement is still replayed by the analysis script. A full system claim needs the same counters attached to the live expert cache/load path.
