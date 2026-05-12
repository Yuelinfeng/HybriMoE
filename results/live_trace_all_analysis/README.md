# HybriMoE Live Trace Overall Analysis

## Key Metrics

| cache | pref | decode events | live hit | issued used | issued coverage | redundant | loads | evicts | router timely candidate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0 | 234 | 0.503 | 0.000 | 0.0000 | 0.000 | 1298 | 882 | 0.0289 |
| 16 | 4 | 234 | 0.499 | 0.143 | 0.0141 | 0.852 | 1437 | 1021 | 0.0287 |
| 16 | 8 | 234 | 0.521 | 0.128 | 0.0333 | 0.805 | 1629 | 1213 | 0.0300 |
| 32 | 4 | 234 | 0.759 | 0.132 | 0.0067 | 0.924 | 1369 | 537 | 0.0253 |
| 32 | 8 | 234 | 0.768 | 0.132 | 0.0148 | 0.916 | 1442 | 610 | 0.0263 |
| 48 | 4 | 234 | 0.855 | 0.143 | 0.0052 | 0.946 | 1510 | 262 | 0.0045 |
| 48 | 8 | 234 | 0.859 | 0.185 | 0.0126 | 0.949 | 1550 | 302 | 0.0056 |
| 56 | 4 | 702 | 0.930 | 0.234 | 0.0044 | 0.971 | 1649 | 193 | 0.0014 |
| 56 | 8 | 598 | 0.936 | 0.172 | 0.0049 | 0.978 | 1663 | 207 | 0.0013 |

## Takeaways

- Cache hit rises from about 0.50 at cache=16 to 0.86 at cache=48 and 0.93 at cache=56.
- Issued prefetch assignment coverage remains below 1.5% for cache=32/48 and below 0.5% for cache=56.
- Candidate redundancy reaches 94%-98% once cache is large, so high residency mostly masks low incremental prefetch value.
- Cache16 includes prefetch=0 and can support extra load/evict pressure analysis; cache32/48/56 need prefetch=0 controls for strict churn deltas.
- cache56 runs have longer decode traces than cache16/32/48, so compare ratios rather than absolute load/evict counts across cache sizes.
