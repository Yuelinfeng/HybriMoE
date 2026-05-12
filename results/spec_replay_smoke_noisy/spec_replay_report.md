# Speculative Prefetch Replay Report

This is an offline replay. Future router assignments are used as oracle/noisy-draft lookahead.
It validates admission-policy direction before connecting a real draft model.

| policy | hit | timely utility | redundant | unused issued | late | eviction damage | prefetch loads | total MB | stall steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always | 95.15% | 93.53% | 75.47% | 56.34% | 3.26% | 56.60% | 12543 | 13073.0 | 2120.0 |
| budgeted | 95.15% | 93.53% | 75.47% | 56.34% | 3.26% | 56.60% | 12543 | 13073.0 | 2120.0 |
| no_prefetch | 64.51% | 0.00% | 0.00% | 0.00% | 0.00% | 18.01% | 0 | 3876.0 | 15504.0 |
| utility_gate | 64.52% | 33.49% | 60.29% | 0.00% | 13.06% | 17.91% | 1426 | 5300.0 | 15496.0 |

Interpretation boundary:

- `always` is a speculative-lookahead upper-pressure baseline, not a proposed policy.
- `utility_gate` is the minimal admission prototype.
- This replay does not include draft-model compute overhead or token acceptance cost.
- A positive result here only justifies wiring the gate into live prefetch scheduling.