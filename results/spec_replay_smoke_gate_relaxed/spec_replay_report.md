# Speculative Prefetch Replay Report

This is an offline replay. Future router assignments are used as oracle/noisy-draft lookahead.
It validates admission-policy direction before connecting a real draft model.

| policy | hit | timely utility | redundant | unused issued | late | eviction damage | prefetch loads | total MB | stall steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always | 95.66% | 95.24% | 95.52% | 1.25% | 3.43% | 50.21% | 4145 | 4619.0 | 1896.0 |
| no_prefetch | 64.51% | 0.00% | 0.00% | 0.00% | 0.00% | 18.01% | 0 | 3876.0 | 15504.0 |
| utility_gate | 71.84% | 70.15% | 67.72% | 0.00% | 25.89% | 22.80% | 3643 | 6718.0 | 12300.0 |

Interpretation boundary:

- `always` is a speculative-lookahead upper-pressure baseline, not a proposed policy.
- `utility_gate` is the minimal admission prototype.
- This replay does not include draft-model compute overhead or token acceptance cost.
- A positive result here only justifies wiring the gate into live prefetch scheduling.