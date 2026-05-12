# Prompt Stream Universality Experiment

This workflow expands the stage-1 and stage-2 evidence from a few simple prompts
to dataset-backed prompt-stream workloads with explicit categories, phases,
cache sizes, and prefetch sizes.

## 1. Build MMLU Prompt Suite

Install the dataset loader if needed:

```bash
python -m pip install datasets
```

```bash
cd /root/autodl-tmp/benchmark/src/HybriMoE

python experiments/build_dataset_prompt_suite.py \
  --preset mmlu \
  --dataset-name cais/mmlu \
  --dataset-config all \
  --split test \
  --output-dir /root/autodl-tmp/hybrimoe_mmlu_prompt_suite_v1 \
  --max-prompts 1000 \
  --stream-length 100 \
  --stream-seeds 0,1,2 \
  --workloads stable_mixed,shifted_mixed \
  --max-new-tokens 128
```

Outputs:

- `prompts.jsonl`: MMLU questions formatted as multiple-choice prompts.
- `streams/*.jsonl`: stable and shifted workload streams.
- `experiment_matrix.csv`: stream/cache/prefetch run matrix.
- `manifest.json`: suite metadata.

## 2. Smoke Test One Short Stream

Use this before launching the full sweep.

```bash
cd /root/autodl-tmp/benchmark/src/HybriMoE
chmod +x experiments/run_prompt_stream_autodl.sh

PROMPT_SUITE_DIR=/root/autodl-tmp/hybrimoe_mmlu_prompt_suite_v1 \
STREAM_GLOB='/root/autodl-tmp/hybrimoe_mmlu_prompt_suite_v1/streams/shifted_mixed_seed0.jsonl' \
CACHE_SIZES='56' \
PREFETCH_SIZES='8' \
PROMPT_LIMIT=10 \
MAX_NEW_TOKENS=64 \
FIXED_DECODE_TOKENS=1 \
DO_SAMPLE=1 \
TEMPERATURE=0.6 \
TOP_P=0.9 \
TOP_K=50 \
OVERRIDE_STREAM_MAX_NEW_TOKENS=1 \
HYBRIMOE_SAFE_SAMPLING=1 \
GGUF_PATH=/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF \
bash experiments/run_prompt_stream_autodl.sh
```

The script keeps one model process alive for the full stream, so expert cache
state carries across prompts.

## 3. Full Stage-1/2 Sweep

```bash
cd /root/autodl-tmp/benchmark/src/HybriMoE

PROMPT_SUITE_DIR=/root/autodl-tmp/hybrimoe_mmlu_prompt_suite_v1 \
CACHE_SIZES='16 32 48 56' \
PREFETCH_SIZES='0 4 8' \
PROMPT_LIMIT=0 \
MAX_NEW_TOKENS=128 \
FIXED_DECODE_TOKENS=1 \
DO_SAMPLE=1 \
TEMPERATURE=0.6 \
TOP_P=0.9 \
TOP_K=50 \
OVERRIDE_STREAM_MAX_NEW_TOKENS=1 \
HYBRIMOE_SAFE_SAMPLING=1 \
GGUF_PATH=/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF \
bash experiments/run_prompt_stream_autodl.sh
```

`FIXED_DECODE_TOKENS=1` suppresses EOS until the final decode step, so each
prompt contributes exactly `MAX_NEW_TOKENS` decode tokens unless the process
fails. `DO_SAMPLE=1` with low temperature/top-p gives more realistic output
than greedy decode, while `HYBRIMOE_SAFE_SAMPLING=1` falls back safely if a
quantized run produces invalid sampling probabilities.

`OVERRIDE_STREAM_MAX_NEW_TOKENS=1` makes the runner's `MAX_NEW_TOKENS` override
any older `max_new_tokens` values stored in a previously generated prompt suite.

For the strictest deterministic measurement, use:

```bash
DO_SAMPLE=0
```

For better qualitative outputs, use:

```bash
DO_SAMPLE=1 TEMPERATURE=0.6 TOP_P=0.9 TOP_K=50
```

To reduce cost, narrow the stream glob first:

```bash
STREAM_GLOB='/root/autodl-tmp/hybrimoe_mmlu_prompt_suite_v1/streams/shifted_mixed_seed*.jsonl'
```

## 4. Aggregate Existing Runs

```bash
cd /root/autodl-tmp/benchmark/src/HybriMoE

python experiments/analyze_prompt_stream_mechanisms.py \
  --input-root /root/autodl-tmp/hybrimoe_prompt_stream_runs \
  --output-dir /root/autodl-tmp/hybrimoe_prompt_stream_runs/aggregate_analysis \
  --stage decode
```

Outputs:

- `group_summary.csv`: per run/workload/phase/category metrics.
- `run_summary.csv`: per stream/cache/prefetch run metrics.
- `category_summary.csv`: cross-run category and phase summary.
- `workload_summary.csv`: workload-level summary.
- `universality_report.md`: high-hit/low-utility prevalence report.

## 5. Package Results

```bash
cd /root/autodl-tmp
tar -cf hybrimoe_prompt_stream_runs.tar hybrimoe_prompt_stream_runs
```

## 6. Stage-3 Speculative Replay

The prompt-stream runner now also runs an offline speculative-prefetch replay
by default:

```bash
RUN_SPEC_REPLAY=1
```

The replay uses router traces as oracle/noisy-draft lookahead and compares:

- `no_prefetch`
- `always`
- `cutoff`
- `budgeted`
- `utility_gate`

To run it later without rerunning the model:

```bash
cd /root/autodl-tmp/benchmark/src/HybriMoE
chmod +x experiments/run_speculative_replay_autodl.sh

SPEC_REPLAY_INPUT_ROOT=/root/autodl-tmp/hybrimoe_prompt_stream_runs \
SPEC_REPLAY_OUTPUT_DIR=/root/autodl-tmp/hybrimoe_prompt_stream_runs/spec_replay_analysis \
SPEC_REPLAY_DRAFT_MODE=oracle \
SPEC_REPLAY_LOOKAHEAD_EVENTS=32 \
SPEC_REPLAY_TRANSFER_LATENCY=4 \
SPEC_REPLAY_GATE_THRESHOLDS='-0.5,0,0.25,0.5' \
bash experiments/run_speculative_replay_autodl.sh
```

See `experiments/SPECULATIVE_REPLAY_AUTODL.md` for noisy-draft and gate
threshold sweeps.

## Interpretation Boundary

This workflow is for stage-1 and stage-2 evidence:

- stage 1: high cache hit and low prefetch utility across many prompt streams.
- stage 2: mechanism diagnosis by redundancy, route drift, unused issued
  prefetch, eviction damage, and timing waits.
- stage 3: offline speculative replay validates whether a utility gate is worth
  wiring into the live scheduler. It is not yet a real draft-model SD runtime.
