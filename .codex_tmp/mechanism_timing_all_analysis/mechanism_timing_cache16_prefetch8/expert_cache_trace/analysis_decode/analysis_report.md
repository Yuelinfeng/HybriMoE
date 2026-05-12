# Expert Cache Trace Analysis

This report uses live HybriMoE expert placement and prefetch events captured inside `KExpertsMarlin`.

## Core Metrics

- resident_hit_expert_ratio: 0.515152
- resident_hit_assignment_ratio: 0.515152
- cpu_assignment_ratio: 0.172494
- gpu_assignment_ratio: 0.827506
- prefetch_redundant_candidate_ratio: 0.834773
- prefetch_candidate_used_by_next_layer_ratio: 0.225000
- prefetch_issued_used_by_next_layer_ratio: 0.116919
- prefetch_candidate_assignment_coverage_ratio: 0.300000
- loaded_experts: 2631
- evicted_experts: 2215
- evictions_per_loaded_expert: 0.841885

## Interpretation Boundary

These are live placement and prefetch-decision metrics, not low-level PCIe timing metrics. They validate whether cached/prefetched experts are selected by the next layer and whether HybriMoE falls back to CPU/GPU execution, but they do not by themselves prove end-to-end latency benefit.
