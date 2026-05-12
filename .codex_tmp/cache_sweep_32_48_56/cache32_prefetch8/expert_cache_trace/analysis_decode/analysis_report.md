# Expert Cache Trace Analysis

This report uses live HybriMoE expert placement and prefetch events captured inside `KExpertsMarlin`.

## Core Metrics

- resident_hit_expert_ratio: 0.767806
- resident_hit_assignment_ratio: 0.767806
- cpu_assignment_ratio: 0.166667
- gpu_assignment_ratio: 0.833333
- prefetch_redundant_candidate_ratio: 0.915556
- prefetch_candidate_used_by_next_layer_ratio: 0.249444
- prefetch_issued_used_by_next_layer_ratio: 0.131579
- prefetch_candidate_assignment_coverage_ratio: 0.332593
- loaded_experts: 1442
- evicted_experts: 610
- evictions_per_loaded_expert: 0.423024

## Interpretation Boundary

These are live placement and prefetch-decision metrics, not low-level PCIe timing metrics. They validate whether cached/prefetched experts are selected by the next layer and whether HybriMoE falls back to CPU/GPU execution, but they do not by themselves prove end-to-end latency benefit.
