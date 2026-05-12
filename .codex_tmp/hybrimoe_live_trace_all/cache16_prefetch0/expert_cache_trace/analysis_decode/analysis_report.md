# Expert Cache Trace Analysis

This report uses live HybriMoE expert placement and prefetch events captured inside `KExpertsMarlin`.

## Core Metrics

- resident_hit_expert_ratio: 0.502849
- resident_hit_assignment_ratio: 0.502849
- cpu_assignment_ratio: 0.168803
- gpu_assignment_ratio: 0.831197
- prefetch_redundant_candidate_ratio: 0.000000
- prefetch_candidate_used_by_next_layer_ratio: 0.000000
- prefetch_issued_used_by_next_layer_ratio: 0.000000
- prefetch_candidate_assignment_coverage_ratio: 0.000000
- loaded_experts: 1298
- evicted_experts: 882
- evictions_per_loaded_expert: 0.679507

## Interpretation Boundary

These are live placement and prefetch-decision metrics, not low-level PCIe timing metrics. They validate whether cached/prefetched experts are selected by the next layer and whether HybriMoE falls back to CPU/GPU execution, but they do not by themselves prove end-to-end latency benefit.
