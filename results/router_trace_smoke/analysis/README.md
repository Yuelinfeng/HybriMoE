# Router Trace Prefetch Utility Analysis

This report uses real router assignments captured from a model run, then replays cache/prefetch policies over that expert-access trace.

## Trace

- trace_path: results\router_trace_smoke\router_trace.jsonl
- model_type: smoke
- events_used: 6
- events_skipped: 0
- num_experts: 4
- max_assignments_per_event: 4

## Config

- cache_size: 24
- prefetch_width: 20
- history_window: 36
- transfer_latency: 4

## Summary

| policy | cache_hit_ratio | timely_useful_candidate_ratio | redundant_ratio | issued_late_ratio | issued_unused_ratio | stall_time | bytes_moved | eviction_damage_ratio | phenomenon_flag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| no_prefetch | 0.8333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 16.0 | 4 | 0.0000 | False |
| history_prefetch | 0.8333 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 16.0 | 4 | 0.0000 | True |
| utility_gate | 0.8333 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 16.0 | 4 | 0.0000 | True |

## Delta: history_prefetch - no_prefetch

- cache_hit_delta: 0.0000
- stall_time_delta: 0.0
- bytes_moved_delta: 0
- demand_bytes_delta: 0

## Gate Check

- utility_gate_prefetch_issued: 0
- history_prefetch_issued: 0
- If the gate issues zero prefetches, it is only a conservative-abstention baseline, not yet a validated utility scheduler.

## Interpretation Boundary

This analysis uses real router assignments, but cache movement is still replayed by the analysis script. A full system claim needs the same counters attached to the live expert cache/load path.
