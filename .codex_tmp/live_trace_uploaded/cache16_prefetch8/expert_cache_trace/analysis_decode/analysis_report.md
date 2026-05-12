# Expert Cache Trace Analysis

This report uses live HybriMoE expert placement and prefetch events captured inside `KExpertsMarlin`.

## Core Metrics

- resident_hit_expert_ratio: 0.521368
- resident_hit_assignment_ratio: 0.521368
- cpu_assignment_ratio: 0.169516
- gpu_assignment_ratio: 0.830484
- prefetch_redundant_candidate_ratio: 0.805000
- prefetch_candidate_used_by_next_layer_ratio: 0.233333
- prefetch_issued_used_by_next_layer_ratio: 0.128205
- prefetch_candidate_assignment_coverage_ratio: 0.311111
- loaded_experts: 1629
- evicted_experts: 1213
- evictions_per_loaded_expert: 0.744629

## Interpretation Boundary

These are live placement and prefetch-decision metrics, not low-level PCIe timing metrics. They validate whether cached/prefetched experts are selected by the next layer and whether HybriMoE falls back to CPU/GPU execution, but they do not by themselves prove end-to-end latency benefit.
