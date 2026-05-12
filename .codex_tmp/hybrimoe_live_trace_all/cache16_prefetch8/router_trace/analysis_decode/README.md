# Router Trace Prefetch Utility Analysis

This report uses real router assignments captured from a model run, then replays cache/prefetch policies over that expert-access trace.

## Trace

- trace_path: /root/autodl-tmp/hybrimoe_live_trace/cache16_prefetch8/router_trace/router_trace.jsonl
- model_type: deepseek_v2
- events_used: 234
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
| no_prefetch | 0.2137 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4416.0 | 1104 | 0.0000 | False |
| history_prefetch | 0.2550 | 0.0300 | 0.4458 | 0.3637 | 0.4778 | 3711.0 | 1604 | 0.5556 | False |
| utility_gate | 0.2137 | 0.0000 | 0.4188 | 0.0000 | 0.0000 | 4416.0 | 1104 | 0.0000 | False |

## Delta: history_prefetch - no_prefetch

- cache_hit_delta: 0.0413
- stall_time_delta: -705.0
- bytes_moved_delta: 500
- demand_bytes_delta: -377

## Gate Check

- utility_gate_prefetch_issued: 0
- history_prefetch_issued: 877
- If the gate issues zero prefetches, it is only a conservative-abstention baseline, not yet a validated utility scheduler.

## Interpretation Boundary

This analysis uses real router assignments, but cache movement is still replayed by the analysis script. A full system claim needs the same counters attached to the live expert cache/load path.
