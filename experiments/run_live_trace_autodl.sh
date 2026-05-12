#!/usr/bin/env bash
set -euo pipefail

# Run one HybriMoE/kTransformers inference with both router trace and live
# expert cache/load trace enabled, then analyze and package the outputs.

ROOT="${HYBRIMOE_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
GGUF_PATH="${GGUF_PATH:-/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF}"
OPTIMIZE_CONFIG_PATH="${OPTIMIZE_CONFIG_PATH:-ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml}"
CACHE_SIZE="${CACHE_SIZE:-16}"
PREFETCH_SIZE="${PREFETCH_SIZE:-4}"

RUN_TAG="${RUN_TAG:-cache${CACHE_SIZE}_prefetch${PREFETCH_SIZE}_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${HYBRIMOE_TRACE_OUT:-/root/autodl-tmp/hybrimoe_live_trace/${RUN_TAG}}"
ROUTER_DIR="${OUT_ROOT}/router_trace"
EXPERT_DIR="${OUT_ROOT}/expert_cache_trace"
ROUTER_TRACE="${ROUTER_DIR}/router_trace.jsonl"
EXPERT_TRACE="${EXPERT_DIR}/expert_cache_trace.jsonl"

ROUTER_STAGE="${ROUTER_STAGE:-decode}"
EXPERT_STAGE="${EXPERT_STAGE:-decode}"
ROUTER_PREFETCH_WIDTH="${ROUTER_PREFETCH_WIDTH:-20}"
ROUTER_HISTORY_WINDOW="${ROUTER_HISTORY_WINDOW:-36}"
ROUTER_TRANSFER_LATENCY="${ROUTER_TRANSFER_LATENCY:-4}"
ROUTER_GATE_THRESHOLD="${ROUTER_GATE_THRESHOLD:-0.25}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

cd "$ROOT"
mkdir -p "$ROUTER_DIR" "$EXPERT_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE"

if [[ ! -d "$GGUF_PATH" ]]; then
  echo "[hybrimoe-live-trace] GGUF_PATH does not exist: $GGUF_PATH" >&2
  echo "[hybrimoe-live-trace] Set GGUF_PATH to the real DeepSeek GGUF directory and rerun." >&2
  exit 2
fi

if [[ ! -f "$OPTIMIZE_CONFIG_PATH" ]]; then
  echo "[hybrimoe-live-trace] optimize config does not exist: $OPTIMIZE_CONFIG_PATH" >&2
  exit 2
fi

export HYBRIMOE_ROUTER_TRACE=1
export HYBRIMOE_ROUTER_TRACE_PATH="$ROUTER_TRACE"
export HYBRIMOE_ROUTER_TRACE_DETAIL="${HYBRIMOE_ROUTER_TRACE_DETAIL:-summary}"

export HYBRIMOE_EXPERT_TRACE=1
export HYBRIMOE_EXPERT_TRACE_PATH="$EXPERT_TRACE"

cat > "${OUT_ROOT}/run_config.json" <<EOF
{
  "root": "$ROOT",
  "python": "$PYTHON_BIN",
  "model_path": "$MODEL_PATH",
  "gguf_path": "$GGUF_PATH",
  "optimize_config_path": "$OPTIMIZE_CONFIG_PATH",
  "cache_size": $CACHE_SIZE,
  "prefetch_size": $PREFETCH_SIZE,
  "router_trace": "$ROUTER_TRACE",
  "expert_trace": "$EXPERT_TRACE",
  "router_stage": "$ROUTER_STAGE",
  "expert_stage": "$EXPERT_STAGE"
}
EOF

echo "[hybrimoe-live-trace] root: $ROOT"
echo "[hybrimoe-live-trace] output: $OUT_ROOT"
echo "[hybrimoe-live-trace] model: $MODEL_PATH"
echo "[hybrimoe-live-trace] gguf: $GGUF_PATH"
echo "[hybrimoe-live-trace] cache_size=$CACHE_SIZE prefetch_size=$PREFETCH_SIZE"
echo "[hybrimoe-live-trace] python: $($PYTHON_BIN --version 2>&1)"

"$PYTHON_BIN" -m ktransformers.local_chat \
  --model_path "$MODEL_PATH" \
  --gguf_path "$GGUF_PATH" \
  --cache_size "$CACHE_SIZE" \
  --prefetch_size "$PREFETCH_SIZE" \
  --optimize_config_path "$OPTIMIZE_CONFIG_PATH" \
  "$@"

if [[ ! -s "$ROUTER_TRACE" ]]; then
  echo "[hybrimoe-live-trace] router trace is empty or missing: $ROUTER_TRACE" >&2
  exit 3
fi

if [[ ! -s "$EXPERT_TRACE" ]]; then
  echo "[hybrimoe-live-trace] expert trace is empty or missing: $EXPERT_TRACE" >&2
  exit 3
fi

echo "[hybrimoe-live-trace] analyzing router trace"
"$PYTHON_BIN" experiments/analyze_router_trace_prefetch.py \
  --trace "$ROUTER_TRACE" \
  --stage "$ROUTER_STAGE" \
  --cache-size "$CACHE_SIZE" \
  --prefetch-width "$ROUTER_PREFETCH_WIDTH" \
  --history-window "$ROUTER_HISTORY_WINDOW" \
  --transfer-latency "$ROUTER_TRANSFER_LATENCY" \
  --gate-threshold "$ROUTER_GATE_THRESHOLD" \
  --output-dir "${ROUTER_DIR}/analysis_${ROUTER_STAGE}"

echo "[hybrimoe-live-trace] analyzing expert cache trace"
"$PYTHON_BIN" experiments/analyze_expert_cache_trace.py \
  --trace "$EXPERT_TRACE" \
  --stage "$EXPERT_STAGE" \
  --output-dir "${EXPERT_DIR}/analysis_${EXPERT_STAGE}"

echo "[hybrimoe-live-trace] analyzing mechanism diagnosis"
"$PYTHON_BIN" experiments/analyze_mechanism_trace.py \
  --run-dir "$OUT_ROOT" \
  --stage "$EXPERT_STAGE" \
  --output-dir "${OUT_ROOT}/mechanism_analysis"

TAR_PATH="${OUT_ROOT}.tar"
tar -C "$(dirname "$OUT_ROOT")" -cf "$TAR_PATH" "$(basename "$OUT_ROOT")"

echo "[hybrimoe-live-trace] done"
echo "[hybrimoe-live-trace] output_dir=$OUT_ROOT"
echo "[hybrimoe-live-trace] tar=$TAR_PATH"
