# Expert Cache Trace Analysis

This report uses live HybriMoE expert placement and prefetch events captured inside `KExpertsMarlin`.

## Core Metrics

- resident_hit_expert_ratio: 0.935897
- resident_hit_assignment_ratio: 0.935897
- cpu_assignment_ratio: 0.166667
- gpu_assignment_ratio: 0.833333
- prefetch_redundant_candidate_ratio: 0.978478
- prefetch_candidate_used_by_next_layer_ratio: 0.227391
- prefetch_issued_used_by_next_layer_ratio: 0.171717
- prefetch_candidate_assignment_coverage_ratio: 0.303188
- loaded_experts: 1663
- evicted_experts: 207
- evictions_per_loaded_expert: 0.124474

## Interpretation Boundary

These are live placement and prefetch-decision metrics, not low-level PCIe timing metrics. They validate whether cached/prefetched experts are selected by the next layer and whether HybriMoE falls back to CPU/GPU execution, but they do not by themselves prove end-to-end latency benefit.
