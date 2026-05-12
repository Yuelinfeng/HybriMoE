# Mechanism Diagnosis Report

This report separates route drift, redundant candidates, unused issued prefetches, eviction damage, and timing wait evidence.

| run | hit | redundant | issued used | issued coverage | route drift | eviction damage | prefetch wait late | expert wait late |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cache16_prefetch0 | 50.28% | 0.00% | 0.00% | 0.00% | 73.54% | 46.28% | n/a | n/a |
| cache16_prefetch4 | 49.86% | 85.22% | 14.29% | 1.41% | 73.99% | 49.72% | n/a | n/a |
| cache16_prefetch4_20260511_180811 | 49.86% | 85.22% | 14.29% | 1.41% | 73.99% | 49.72% | n/a | n/a |
| cache16_prefetch8 | 52.14% | 80.50% | 12.82% | 3.33% | 73.86% | 54.91% | n/a | n/a |
| mechanism_timing_cache16_prefetch8 | 51.52% | 83.48% | 11.69% | 2.58% | 79.00% | 60.76% | 100.00% | 0.49% |
| cache32_prefetch0 | 73.65% | 0.00% | 0.00% | 0.00% | 74.20% | 31.35% | n/a | n/a |
| cache32_prefetch4 | 75.85% | 92.44% | 13.24% | 0.67% | 72.50% | 30.97% | n/a | n/a |
| cache32_prefetch8 | 76.78% | 91.56% | 13.16% | 1.48% | 72.62% | 32.21% | n/a | n/a |
| cache48_prefetch0 | 84.19% | 0.00% | 0.00% | 0.00% | 72.88% | 23.87% | n/a | n/a |
| cache48_prefetch4 | 85.54% | 94.56% | 14.29% | 0.52% | 73.01% | 25.12% | n/a | n/a |
| cache48_prefetch8 | 85.90% | 94.89% | 18.48% | 1.26% | 74.65% | 31.31% | n/a | n/a |
| mechanism_timing_cache48_prefetch8 | 87.54% | 96.12% | 13.92% | 0.72% | 79.44% | 30.04% | 100.00% | 1.08% |
| cache56_prefetch0 | 90.74% | 0.00% | 0.00% | 0.00% | 74.18% | 22.31% | n/a | n/a |
| cache56_prefetch4 | 93.04% | 97.15% | 23.38% | 0.44% | 78.43% | 21.16% | n/a | n/a |
| cache56_prefetch8 | 93.59% | 97.85% | 17.17% | 0.49% | 78.56% | 23.91% | n/a | n/a |
| mechanism_timing_cache56_prefetch8 | 93.59% | 97.85% | 17.17% | 0.49% | 78.56% | 23.91% | 100.00% | 0.07% |

Interpretation:

- `redundant` measures candidates already resident at issue time.
- `issued used` measures issued load experts selected by the next matching placement.
- `issued coverage` measures how many next-layer assignments are covered by issued prefetches.
- `route drift` is `1 - same-layer consecutive decode Jaccard`.
- `eviction damage` counts resident misses whose expert had been evicted since its last load.
- wait ratios require traces generated after timing instrumentation; older traces show `n/a`.
- `load_source_summary.csv` separates future traces by `demand`, `prefetch`, `init`, and `prefill_buffer` sources.
