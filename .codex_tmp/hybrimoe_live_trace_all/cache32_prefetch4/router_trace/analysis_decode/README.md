# Router Trace Prefetch Utility Analysis

This report uses real router assignments captured from a model run, then replays cache/prefetch policies over that expert-access trace.

## Trace

- trace_path: /root/autodl-tmp/hybrimoe_live_trace/cache32_prefetch4/router_trace/router_trace.jsonl
- model_type: deepseek_v2
- events_used: 234
- events_skipped: 26
- num_experts: 64
- max_assignments_per_event: 6

## Config

- cache_size: 32
- prefetch_width: 20
- history_window: 36
- transfer_latency: 4

## Summary

| policy | cache_hit_ratio | timely_useful_candidate_ratio | redundant_ratio | issued_late_ratio | issued_unused_ratio | stall_time | bytes_moved | eviction_damage_ratio | phenomenon_flag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| no_prefetch | 0.4879 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2876.0 | 719 | 0.0000 | False |
| history_prefetch | 0.5021 | 0.0253 | 0.7638 | 0.3846 | 0.3154 | 2603.0 | 939 | 0.5750 | False |
| utility_gate | 0.4879 | 0.0000 | 0.7677 | 0.0000 | 0.0000 | 2876.0 | 719 | 0.0000 | False |

## Delta: history_prefetch - no_prefetch

- cache_hit_delta: 0.0142
- stall_time_delta: -273.0
- bytes_moved_delta: 220
- demand_bytes_delta: -170

## Gate Check

- utility_gate_prefetch_issued: 0
- history_prefetch_issued: 390
- If the gate issues zero prefetches, it is only a conservative-abstention baseline, not yet a validated utility scheduler.

## Interpretation Boundary

This analysis uses real router assignments, but cache movement is still replayed by the analysis script. A full system claim needs the same counters attached to the live expert cache/load path.
