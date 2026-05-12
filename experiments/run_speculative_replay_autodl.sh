#!/usr/bin/env bash
set -euo pipefail

# Offline speculative-prefetch replay over existing HybriMoE router traces.
# This does not load the model. It compares draft/oracle lookahead policies over
# traces already produced by run_live_trace_autodl.sh or run_prompt_stream_autodl.sh.

ROOT="${HYBRIMOE_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT_ROOT="${SPEC_REPLAY_INPUT_ROOT:-/root/autodl-tmp/hybrimoe_prompt_stream_runs}"
OUTPUT_DIR="${SPEC_REPLAY_OUTPUT_DIR:-${INPUT_ROOT}/spec_replay_analysis}"

SPEC_REPLAY_STAGE="${SPEC_REPLAY_STAGE:-decode}"
SPEC_REPLAY_DECODE_WINDOW="${SPEC_REPLAY_DECODE_WINDOW:-0}"
SPEC_REPLAY_POLICIES="${SPEC_REPLAY_POLICIES:-no_prefetch,always,cutoff,budgeted,utility_gate}"
SPEC_REPLAY_DRAFT_MODE="${SPEC_REPLAY_DRAFT_MODE:-oracle}"
SPEC_REPLAY_DRAFT_FIDELITY="${SPEC_REPLAY_DRAFT_FIDELITY:-1.0}"
SPEC_REPLAY_ACCEPT_PROB="${SPEC_REPLAY_ACCEPT_PROB:-0.8}"
SPEC_REPLAY_LOOKAHEAD_EVENTS="${SPEC_REPLAY_LOOKAHEAD_EVENTS:-32}"
SPEC_REPLAY_TRANSFER_LATENCY="${SPEC_REPLAY_TRANSFER_LATENCY:-4}"
SPEC_REPLAY_PREFETCH_SIZE_OVERRIDE="${SPEC_REPLAY_PREFETCH_SIZE_OVERRIDE:-0}"
SPEC_REPLAY_CACHE_SIZE_OVERRIDE="${SPEC_REPLAY_CACHE_SIZE_OVERRIDE:-0}"
SPEC_REPLAY_CUTOFF_LAYER="${SPEC_REPLAY_CUTOFF_LAYER:-12}"
SPEC_REPLAY_GATE_THRESHOLD="${SPEC_REPLAY_GATE_THRESHOLD:-0.25}"
SPEC_REPLAY_GATE_THRESHOLDS="${SPEC_REPLAY_GATE_THRESHOLDS:-}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

echo "[spec-replay] input_root=$INPUT_ROOT"
echo "[spec-replay] output_dir=$OUTPUT_DIR"
echo "[spec-replay] policies=$SPEC_REPLAY_POLICIES"
echo "[spec-replay] draft_mode=$SPEC_REPLAY_DRAFT_MODE fidelity=$SPEC_REPLAY_DRAFT_FIDELITY"
echo "[spec-replay] lookahead_events=$SPEC_REPLAY_LOOKAHEAD_EVENTS transfer_latency=$SPEC_REPLAY_TRANSFER_LATENCY"

"$PYTHON_BIN" experiments/replay_speculative_prefetch.py \
  --input-root "$INPUT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --stage "$SPEC_REPLAY_STAGE" \
  --decode-window "$SPEC_REPLAY_DECODE_WINDOW" \
  --policies "$SPEC_REPLAY_POLICIES" \
  --draft-mode "$SPEC_REPLAY_DRAFT_MODE" \
  --draft-fidelity "$SPEC_REPLAY_DRAFT_FIDELITY" \
  --accept-prob "$SPEC_REPLAY_ACCEPT_PROB" \
  --lookahead-events "$SPEC_REPLAY_LOOKAHEAD_EVENTS" \
  --transfer-latency "$SPEC_REPLAY_TRANSFER_LATENCY" \
  --prefetch-size-override "$SPEC_REPLAY_PREFETCH_SIZE_OVERRIDE" \
  --cache-size-override "$SPEC_REPLAY_CACHE_SIZE_OVERRIDE" \
  --cutoff-layer "$SPEC_REPLAY_CUTOFF_LAYER" \
  --gate-threshold "$SPEC_REPLAY_GATE_THRESHOLD" \
  --gate-thresholds "$SPEC_REPLAY_GATE_THRESHOLDS"

tar -C "$(dirname "$OUTPUT_DIR")" -cf "${OUTPUT_DIR}.tar" "$(basename "$OUTPUT_DIR")" 2>/dev/null || true

echo "[spec-replay] done"
echo "[spec-replay] output_dir=$OUTPUT_DIR"
