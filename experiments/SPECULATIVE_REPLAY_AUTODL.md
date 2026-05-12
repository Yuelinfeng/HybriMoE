# Speculative Prefetch Replay

This workflow implements the conservative stage-3 path:

1. use existing router traces as oracle or noisy-draft lookahead;
2. replay several prefetch policies offline;
3. validate whether utility admission reduces redundant, unused, late, and
   eviction-damaging prefetch before wiring a real draft model.

It does not load the model and does not run token-level speculative decoding.

## Run on Existing Prompt-Stream Traces

```bash
cd /root/autodl-tmp/benchmark/src/HybriMoE
chmod +x experiments/run_speculative_replay_autodl.sh

SPEC_REPLAY_INPUT_ROOT=/root/autodl-tmp/hybrimoe_prompt_stream_runs \
SPEC_REPLAY_OUTPUT_DIR=/root/autodl-tmp/hybrimoe_prompt_stream_runs/spec_replay_analysis \
SPEC_REPLAY_DRAFT_MODE=oracle \
SPEC_REPLAY_DRAFT_FIDELITY=1.0 \
SPEC_REPLAY_LOOKAHEAD_EVENTS=32 \
SPEC_REPLAY_TRANSFER_LATENCY=4 \
SPEC_REPLAY_GATE_THRESHOLDS='-0.5,0,0.25,0.5' \
bash experiments/run_speculative_replay_autodl.sh
```

Outputs:

- `spec_replay_summary.csv`: per run and per policy metrics.
- `spec_replay_policy_summary.csv`: policy-level aggregate.
- `spec_replay_report.md`: compact comparison table.
- `spec_replay_analysis.tar`: packaged analysis directory.

## Run on a Single Existing Live-Trace Directory

```bash
cd /root/autodl-tmp/benchmark/src/HybriMoE

python experiments/replay_speculative_prefetch.py \
  --run-dir /root/autodl-tmp/hybrimoe_live_trace/mechanism_timing_cache56_prefetch8 \
  --output-dir /root/autodl-tmp/hybrimoe_live_trace/mechanism_timing_cache56_prefetch8/spec_replay_analysis \
  --stage decode \
  --draft-mode oracle \
  --lookahead-events 32 \
  --transfer-latency 4
```

## Main Policies

- `no_prefetch`: demand-only cache baseline.
- `always`: prefetch every draft/oracle candidate until the per-step budget is
  exhausted.
- `cutoff`: only prefetch candidates whose target layer is not beyond
  `--cutoff-layer`.
- `budgeted`: rank draft/oracle candidates by confidence and keep the top
  budget.
- `utility_gate`: admit a candidate only when expected saved stall exceeds
  transfer, eviction, contention, and lateness costs.

The minimal gate is:

```text
confidence * assignment_count * miss_penalty
  > transfer_cost
    + eviction_cost * cache_pressure
    + contention_cost * pending_pressure
    + lateness_cost * lateness
    + gate_threshold
```

## Noisy Draft Stress Test

Use this to emulate an imperfect draft model before implementing a real draft
runtime.

```bash
python experiments/replay_speculative_prefetch.py \
  --input-root /root/autodl-tmp/hybrimoe_prompt_stream_runs \
  --output-dir /root/autodl-tmp/hybrimoe_prompt_stream_runs/spec_replay_noisy_f08 \
  --stage decode \
  --draft-mode noisy_oracle \
  --draft-fidelity 0.8 \
  --accept-prob 0.75 \
  --lookahead-events 32 \
  --transfer-latency 4 \
  --gate-thresholds '-0.5,0,0.25,0.5'
```

## Success Criteria for Stage 3

The utility gate is promising if, relative to `always` or `budgeted`, it
reduces:

- `prefetch_redundant_candidate_ratio`;
- `prefetch_unused_issued_ratio`;
- `prefetch_late_assignment_ratio`;
- `prefetch_evicted_unused_ratio`;
- `estimated_total_bytes_mb`;
- `eviction_damage_assignment_ratio`;

while preserving most of:

- `prefetch_timely_useful_assignment_ratio`;
- `cache_hit_assignment_ratio`;
- `estimated_demand_stall_steps`.
