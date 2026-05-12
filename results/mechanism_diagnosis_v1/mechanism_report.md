# Mechanism Diagnosis Report

This report separates route drift, redundant candidates, unused issued prefetches, eviction damage, and timing wait evidence.

| run | hit | redundant | issued used | issued coverage | route drift | eviction damage | prefetch wait late | expert wait late |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cache16_prefetch0 | 42.98% | 0.00% | 0.00% | 0.00% | 73.54% | 28.01% | n/a | n/a |
| cache16_prefetch4 | 42.68% | 85.22% | 14.29% | 1.41% | 73.99% | 30.20% | n/a | n/a |
| cache16_prefetch8 | 44.26% | 80.50% | 12.82% | 3.33% | 73.86% | 32.74% | n/a | n/a |
| cache32_prefetch4 | 67.65% | 92.44% | 13.24% | 0.67% | 72.50% | 15.67% | n/a | n/a |
| cache32_prefetch8 | 68.28% | 91.56% | 13.16% | 1.48% | 72.62% | 15.98% | n/a | n/a |
| cache48_prefetch4 | 82.33% | 94.56% | 14.29% | 0.52% | 73.01% | 13.93% | n/a | n/a |
| cache48_prefetch8 | 82.57% | 94.89% | 18.48% | 1.26% | 74.65% | 17.17% | n/a | n/a |
| cache56_prefetch4 | 92.38% | 97.15% | 23.38% | 0.44% | 78.43% | 16.67% | n/a | n/a |
| cache56_prefetch8 | 92.74% | 97.85% | 17.17% | 0.49% | 78.56% | 17.80% | n/a | n/a |

Interpretation:

- `redundant` measures candidates already resident at issue time.
- `issued used` measures issued load experts selected by the next matching placement.
- `issued coverage` measures how many next-layer assignments are covered by issued prefetches.
- `route drift` is `1 - same-layer consecutive decode Jaccard`.
- `eviction damage` counts resident misses whose expert had been evicted since its last load.
- wait ratios require traces generated after timing instrumentation; older traces show `n/a`.
