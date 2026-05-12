#!/usr/bin/env bash
set -euo pipefail

# Run prompt-stream HybriMoE experiments for stage-1/2 universality evidence.
# Each run keeps one model process alive for the full prompt stream, so expert
# cache state carries across prompts in the same workload stream.

ROOT="${HYBRIMOE_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
GGUF_PATH="${GGUF_PATH:-/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF}"
OPTIMIZE_CONFIG_PATH="${OPTIMIZE_CONFIG_PATH:-ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml}"

PROMPT_SUITE_DIR="${PROMPT_SUITE_DIR:-/root/autodl-tmp/hybrimoe_prompt_suite_v1}"
PROMPTS_PER_CATEGORY="${PROMPTS_PER_CATEGORY:-100}"
STREAM_LENGTH="${STREAM_LENGTH:-100}"
STREAM_SEEDS="${STREAM_SEEDS:-0,1,2}"
STREAM_GLOB="${STREAM_GLOB:-${PROMPT_SUITE_DIR}/streams/*.jsonl}"
PROMPT_LIMIT="${PROMPT_LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DO_SAMPLE="${DO_SAMPLE:-0}"

CACHE_SIZES="${CACHE_SIZES:-16 32 48 56}"
PREFETCH_SIZES="${PREFETCH_SIZES:-0 4 8}"
RUN_PREFIX="${RUN_PREFIX:-prompt_stream_v1}"
OUT_BASE="${HYBRIMOE_TRACE_OUT_BASE:-/root/autodl-tmp/hybrimoe_prompt_stream_runs}"
PACKAGE_EACH_RUN="${PACKAGE_EACH_RUN:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ROUTER_STAGE="${ROUTER_STAGE:-decode}"
EXPERT_STAGE="${EXPERT_STAGE:-decode}"
ROUTER_PREFETCH_WIDTH="${ROUTER_PREFETCH_WIDTH:-20}"
ROUTER_HISTORY_WINDOW="${ROUTER_HISTORY_WINDOW:-36}"
ROUTER_TRANSFER_LATENCY="${ROUTER_TRANSFER_LATENCY:-4}"
ROUTER_GATE_THRESHOLD="${ROUTER_GATE_THRESHOLD:-0.25}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

cd "$ROOT"
mkdir -p "$OUT_BASE" "$HF_HOME" "$TRANSFORMERS_CACHE"

if [[ ! -d "$GGUF_PATH" ]]; then
  echo "[prompt-stream] GGUF_PATH does not exist: $GGUF_PATH" >&2
  exit 2
fi

if [[ ! -f "$OPTIMIZE_CONFIG_PATH" ]]; then
  echo "[prompt-stream] optimize config does not exist: $OPTIMIZE_CONFIG_PATH" >&2
  exit 2
fi

if [[ ! -d "${PROMPT_SUITE_DIR}/streams" ]]; then
  echo "[prompt-stream] building prompt suite at $PROMPT_SUITE_DIR"
  "$PYTHON_BIN" experiments/build_prompt_suite.py \
    --output-dir "$PROMPT_SUITE_DIR" \
    --prompts-per-category "$PROMPTS_PER_CATEGORY" \
    --stream-length "$STREAM_LENGTH" \
    --stream-seeds "$STREAM_SEEDS" \
    --max-new-tokens "$MAX_NEW_TOKENS"
fi

shopt -s nullglob
STREAM_FILES=( $STREAM_GLOB )
shopt -u nullglob
if [[ "${#STREAM_FILES[@]}" -eq 0 ]]; then
  echo "[prompt-stream] no stream files matched: $STREAM_GLOB" >&2
  exit 3
fi

echo "[prompt-stream] root=$ROOT"
echo "[prompt-stream] output_base=$OUT_BASE"
echo "[prompt-stream] streams=${#STREAM_FILES[@]}"
echo "[prompt-stream] cache_sizes=$CACHE_SIZES"
echo "[prompt-stream] prefetch_sizes=$PREFETCH_SIZES"
echo "[prompt-stream] prompt_limit=$PROMPT_LIMIT max_new_tokens=$MAX_NEW_TOKENS"
echo "[prompt-stream] do_sample=$DO_SAMPLE"
echo "[prompt-stream] python: $($PYTHON_BIN --version 2>&1)"

run_one() {
  local stream_file="$1"
  local cache_size="$2"
  local prefetch_size="$3"
  local stream_id
  stream_id="$(basename "$stream_file" .jsonl)"
  local run_tag="${RUN_PREFIX}_${stream_id}_cache${cache_size}_prefetch${prefetch_size}"
  local out_root="${OUT_BASE}/${run_tag}"
  local router_dir="${out_root}/router_trace"
  local expert_dir="${out_root}/expert_cache_trace"
  local router_trace="${router_dir}/router_trace.jsonl"
  local expert_trace="${expert_dir}/expert_cache_trace.jsonl"
  local generation_summary="${out_root}/generation_summary.jsonl"

  if [[ "$SKIP_EXISTING" == "1" && -s "${out_root}/mechanism_analysis/mechanism_summary.csv" ]]; then
    echo "[prompt-stream] skip existing $run_tag"
    return
  fi

  mkdir -p "$router_dir" "$expert_dir"
  rm -f "$router_trace" "$expert_trace" "$generation_summary"

  export HYBRIMOE_ROUTER_TRACE=1
  export HYBRIMOE_ROUTER_TRACE_PATH="$router_trace"
  export HYBRIMOE_ROUTER_TRACE_DETAIL="${HYBRIMOE_ROUTER_TRACE_DETAIL:-summary}"
  export HYBRIMOE_EXPERT_TRACE=1
  export HYBRIMOE_EXPERT_TRACE_PATH="$expert_trace"
  export HYBRIMOE_DO_SAMPLE="$DO_SAMPLE"
  export HYBRIMOE_SAFE_SAMPLING="${HYBRIMOE_SAFE_SAMPLING:-1}"

  cat > "${out_root}/run_config.json" <<EOF
{
  "run_tag": "$run_tag",
  "root": "$ROOT",
  "python": "$PYTHON_BIN",
  "model_path": "$MODEL_PATH",
  "gguf_path": "$GGUF_PATH",
  "optimize_config_path": "$OPTIMIZE_CONFIG_PATH",
  "cache_size": $cache_size,
  "prefetch_size": $prefetch_size,
  "prompt_stream": "$stream_file",
  "prompt_limit": $PROMPT_LIMIT,
  "max_new_tokens": $MAX_NEW_TOKENS,
  "do_sample": $DO_SAMPLE,
  "router_trace": "$router_trace",
  "expert_trace": "$expert_trace",
  "generation_summary": "$generation_summary",
  "router_stage": "$ROUTER_STAGE",
  "expert_stage": "$EXPERT_STAGE"
}
EOF

  echo "[prompt-stream] running $run_tag"
  "$PYTHON_BIN" -m ktransformers.local_chat \
    --model_path "$MODEL_PATH" \
    --gguf_path "$GGUF_PATH" \
    --cache_size "$cache_size" \
    --prefetch_size "$prefetch_size" \
    --optimize_config_path "$OPTIMIZE_CONFIG_PATH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --do_sample "$DO_SAMPLE" \
    --prompt_stream "$stream_file" \
    --prompt_limit "$PROMPT_LIMIT" \
    --stream_output "$generation_summary"

  if [[ ! -s "$router_trace" ]]; then
    echo "[prompt-stream] router trace is empty or missing: $router_trace" >&2
    exit 4
  fi
  if [[ ! -s "$expert_trace" ]]; then
    echo "[prompt-stream] expert trace is empty or missing: $expert_trace" >&2
    exit 4
  fi

  "$PYTHON_BIN" experiments/analyze_router_trace_prefetch.py \
    --trace "$router_trace" \
    --stage "$ROUTER_STAGE" \
    --cache-size "$cache_size" \
    --prefetch-width "$ROUTER_PREFETCH_WIDTH" \
    --history-window "$ROUTER_HISTORY_WINDOW" \
    --transfer-latency "$ROUTER_TRANSFER_LATENCY" \
    --gate-threshold "$ROUTER_GATE_THRESHOLD" \
    --output-dir "${router_dir}/analysis_${ROUTER_STAGE}"

  "$PYTHON_BIN" experiments/analyze_expert_cache_trace.py \
    --trace "$expert_trace" \
    --stage "$EXPERT_STAGE" \
    --output-dir "${expert_dir}/analysis_${EXPERT_STAGE}"

  "$PYTHON_BIN" experiments/analyze_mechanism_trace.py \
    --run-dir "$out_root" \
    --stage "$EXPERT_STAGE" \
    --output-dir "${out_root}/mechanism_analysis"

  if [[ "$PACKAGE_EACH_RUN" == "1" ]]; then
    tar -C "$(dirname "$out_root")" -cf "${out_root}.tar" "$(basename "$out_root")"
  fi
}

for stream_file in "${STREAM_FILES[@]}"; do
  for cache_size in $CACHE_SIZES; do
    for prefetch_size in $PREFETCH_SIZES; do
      run_one "$stream_file" "$cache_size" "$prefetch_size"
    done
  done
done

"$PYTHON_BIN" experiments/analyze_prompt_stream_mechanisms.py \
  --input-root "$OUT_BASE" \
  --output-dir "${OUT_BASE}/aggregate_analysis" \
  --stage "$EXPERT_STAGE" || true

tar -C "$OUT_BASE" -cf "${OUT_BASE}/aggregate_analysis.tar" aggregate_analysis 2>/dev/null || true

echo "[prompt-stream] done"
echo "[prompt-stream] output_base=$OUT_BASE"
