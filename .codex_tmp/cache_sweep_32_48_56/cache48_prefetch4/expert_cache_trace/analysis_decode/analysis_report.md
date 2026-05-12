# Expert Cache Trace Analysis

This report uses live HybriMoE expert placement and prefetch events captured inside `KExpertsMarlin`.

## Core Metrics

- resident_hit_expert_ratio: 0.855413
- resident_hit_assignment_ratio: 0.855413
- cpu_assignment_ratio: 0.166667
- gpu_assignment_ratio: 0.833333
- prefetch_redundant_candidate_ratio: 0.945556
- prefetch_candidate_used_by_next_layer_ratio: 0.268889
- prefetch_issued_used_by_next_layer_ratio: 0.142857
- prefetch_candidate_assignment_coverage_ratio: 0.179259
- loaded_experts: 1510
- evicted_experts: 262
- evictions_per_loaded_expert: 0.173510

## Interpretation Boundary

These are live placement and prefetch-decision metrics, not low-level PCIe timing metrics. They validate whether cached/prefetched experts are selected by the next layer and whether HybriMoE falls back to CPU/GPU execution, but they do not by themselves prove end-to-end latency benefit.
