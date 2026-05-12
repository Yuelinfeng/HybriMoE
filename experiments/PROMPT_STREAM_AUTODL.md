# Prompt Stream Universality Experiment

This workflow expands the stage-1 and stage-2 evidence from a few simple prompts
to prompt-stream workloads with explicit categories, phases, cache sizes, and
prefetch sizes.

## 1. Build Prompt Suite

```bash
cd /root/autodl-tmp/benchmark/src/HybriMoE

python experiments/build_prompt_suite.py \
  --output-dir /root/autodl-tmp/hybrimoe_prompt_suite_v1 \
  --prompts-per-category 100 \
  --stream-length 100 \
  --stream-seeds 0,1,2 \
  --max-new-tokens 128
```

Outputs:

- `prompts.jsonl`: 10 categories, default 100 prompts per category.
- `streams/*.jsonl`: stable and shifted workload streams.
- `experiment_matrix.csv`: stream/cache/prefetch run matrix.
- `manifest.json`: suite metadata.

## 2. Smoke Test One Short Stream

Use this before launching the full sweep.

```bash
cd /root/autodl-tmp/benchmark/src/HybriMoE
chmod +x experiments/run_prompt_stream_autodl.sh

PROMPT_SUITE_DIR=/root/autodl-tmp/hybrimoe_prompt_suite_v1 \
STREAM_GLOB='/root/autodl-tmp/hybrimoe_prompt_suite_v1/streams/shifted_mixed_seed0.jsonl' \
CACHE_SIZES='56' \
PREFETCH_SIZES='8' \
PROMPT_LIMIT=10 \
MAX_NEW_TOKENS=64 \
DO_SAMPLE=0 \
GGUF_PATH=/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF \
bash experiments/run_prompt_stream_autodl.sh
```

The script keeps one model process alive for the full stream, so expert cache
state carries across prompts.

## 3. Full Stage-1/2 Sweep

```bash
cd /root/autodl-tmp/benchmark/src/HybriMoE

PROMPT_SUITE_DIR=/root/autodl-tmp/hybrimoe_prompt_suite_v1 \
CACHE_SIZES='16 32 48 56' \
PREFETCH_SIZES='0 4 8' \
PROMPT_LIMIT=0 \
MAX_NEW_TOKENS=128 \
DO_SAMPLE=0 \
GGUF_PATH=/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF \
bash experiments/run_prompt_stream_autodl.sh
```

`DO_SAMPLE=0` is intentional for measurement runs. It uses deterministic greedy
decode and avoids CUDA-side crashes from invalid sampling probabilities. The
trace claim depends on router/cache behavior under a prompt stream, not on
sampling diversity.

To reduce cost, narrow the stream glob first:

```bash
STREAM_GLOB='/root/autodl-tmp/hybrimoe_prompt_suite_v1/streams/shifted_mixed_seed*.jsonl'
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

## Interpretation Boundary

This workflow is for stage-1 and stage-2 evidence:

- stage 1: high cache hit and low prefetch utility across many prompt streams.
- stage 2: mechanism diagnosis by redundancy, route drift, unused issued
  prefetch, eviction damage, and timing waits.

It does not yet validate the stage-3 admission gate.
