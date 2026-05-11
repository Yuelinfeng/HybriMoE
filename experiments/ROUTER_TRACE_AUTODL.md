# AutoDL Real Router Trace Workflow

This workflow captures real MoE router assignments during a kTransformers run,
then replays cache/prefetch policies over the captured expert-access trace.

## 1. Enable Router Tracing

Set these variables before launching your normal HybriMoE/kTransformers command:

```bash
export HYBRIMOE_ROUTER_TRACE=1
export HYBRIMOE_ROUTER_TRACE_PATH=/root/autodl-tmp/hybrimoe_router_trace/router_trace.jsonl
export HYBRIMOE_ROUTER_TRACE_DETAIL=summary
```

Then run the same model command you already use on AutoDL. For example:

```bash
python ktransformers/local_chat.py <your existing args>
```

The tracer is off by default. When enabled, it writes one JSONL record per MoE
router invocation. The default `summary` mode records per-expert assignment
counts, not full per-token top-k lists. Current hooks cover Mixtral, Qwen2-MoE,
DeepSeek-V2, and DeepSeek-V3 Python router paths. For detailed token-level
traces:

```bash
export HYBRIMOE_ROUTER_TRACE_DETAIL=tokens
export HYBRIMOE_ROUTER_TRACE_MAX_TOKENS_PER_EVENT=4096
```

## 2. Analyze The Trace

After generation finishes:

```bash
python experiments/analyze_router_trace_prefetch.py \
  --trace /root/autodl-tmp/hybrimoe_router_trace/router_trace.jsonl \
  --output-dir /root/autodl-tmp/hybrimoe_router_trace/prefetch_analysis
```

Useful filters:

```bash
# Decode-only analysis
python experiments/analyze_router_trace_prefetch.py \
  --trace /root/autodl-tmp/hybrimoe_router_trace/router_trace.jsonl \
  --stage decode \
  --output-dir /root/autodl-tmp/hybrimoe_router_trace/prefetch_analysis_decode

# Analyze selected MoE layers only
python experiments/analyze_router_trace_prefetch.py \
  --trace /root/autodl-tmp/hybrimoe_router_trace/router_trace.jsonl \
  --layers 3,4,5 \
  --output-dir /root/autodl-tmp/hybrimoe_router_trace/prefetch_analysis_layers_3_5
```

Key outputs:

```text
summary.csv
summary.json
trace_metadata.json
README.md
```

## 3. What This Proves

This workflow uses real model router assignments, so it is stronger than the
synthetic harness. It can test whether real expert routes exhibit:

- high `cache_hit_ratio`
- low `timely_useful_candidate_ratio`
- high redundant / late / unused prefetch ratios
- extra bytes or stall in the replayed prefetch policy

It still replays cache/prefetch movement offline. The next step after this is
to attach the same counters to the live expert cache/load path in
`ktransformers/operators/experts.py`.
