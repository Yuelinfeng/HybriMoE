# Live Expert Cache Trace on AutoDL

## Scripted Run

```bash
cd /root/autodl-tmp/benchmark/src/HybriMoE
chmod +x experiments/run_live_trace_autodl.sh

GGUF_PATH=/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF \
CACHE_SIZE=16 \
PREFETCH_SIZE=4 \
bash experiments/run_live_trace_autodl.sh
```

To compare prefetch settings:

```bash
cd /root/autodl-tmp/benchmark/src/HybriMoE
chmod +x experiments/run_live_trace_autodl.sh

for p in 0 4 8; do
  GGUF_PATH=/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF \
  CACHE_SIZE=16 \
  PREFETCH_SIZE="$p" \
  RUN_TAG="cache16_prefetch${p}" \
  bash experiments/run_live_trace_autodl.sh
done
```

这个 trace 接在 `KExpertsMarlin` 的真实 expert cache/load 路径上，用来补上 router replay 不能证明的部分：当前层专家是否已经在 GPU cache、哪些专家被 HybriMoE 放到 GPU/CPU 执行、以及下一层预取实际发起了哪些 expert load。

## 运行

```bash
mkdir -p /root/autodl-tmp/hybrimoe_router_trace
mkdir -p /root/autodl-tmp/hybrimoe_expert_trace

export HYBRIMOE_ROUTER_TRACE=1
export HYBRIMOE_ROUTER_TRACE_PATH=/root/autodl-tmp/hybrimoe_router_trace/router_trace.jsonl
export HYBRIMOE_ROUTER_TRACE_DETAIL=summary

export HYBRIMOE_EXPERT_TRACE=1
export HYBRIMOE_EXPERT_TRACE_PATH=/root/autodl-tmp/hybrimoe_expert_trace/expert_cache_trace.jsonl

python -m ktransformers.local_chat \
  --model_path deepseek-ai/DeepSeek-V2-Lite-Chat \
  --gguf_path /root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF \
  --cache_size 16 \
  --prefetch_size 4 \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml
```

如果你的 GGUF 实际目录不是 `/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF`，把 `--gguf_path` 改成真实存在的目录。要验证预取效用，不要只跑 `--prefetch_size 0`；建议至少比较 `0, 4, 8`。

## 分析

```bash
python experiments/analyze_expert_cache_trace.py \
  --trace /root/autodl-tmp/hybrimoe_expert_trace/expert_cache_trace.jsonl \
  --stage decode \
  --output-dir /root/autodl-tmp/hybrimoe_expert_trace/analysis_decode

python experiments/analyze_mechanism_trace.py \
  --run-dir /root/autodl-tmp/hybrimoe_live_trace/cache16_prefetch4 \
  --stage decode \
  --output-dir /root/autodl-tmp/hybrimoe_live_trace/cache16_prefetch4/mechanism_analysis

tar -C /root/autodl-tmp -cf /root/autodl-tmp/hybrimoe_expert_trace.tar \
  hybrimoe_router_trace hybrimoe_expert_trace
```

## 关键指标

- `resident_hit_expert_ratio`: 被当前层选中的 unique experts 中，已经在 GPU cache 的比例。
- `resident_hit_assignment_ratio`: 按 token-expert assignment 加权后的 GPU cache 命中比例。
- `cpu_assignment_ratio` / `gpu_assignment_ratio`: HSS 最终把多少 assignment 放到 CPU/GPU 执行。
- `prefetch_redundant_candidate_ratio`: 预取候选里本来已经 resident 的比例。
- `prefetch_candidate_used_by_next_layer_ratio`: 上一层为下一层预取的候选中，有多少在下一层真的被选中。
- `prefetch_issued_used_by_next_layer_ratio`: 实际发起 load 的预取专家中，有多少在下一层真的被选中。
- `prefetch_candidate_assignment_coverage_ratio`: 被预取候选覆盖的下一层 assignment 比例。
- `loaded_experts` / `evicted_experts`: 真实 cache load 与 eviction 次数，用来观察预取是否制造额外替换压力。
- `evictions_per_loaded_expert`: 每次 load 伴随的平均 eviction 压力，辅助判断 cache degradation。
- `prefetch_wait` / `expert_wait`: timing instrumentation 后可用，用来判断 prefetch 是否 ready before consume，以及 demand load 是否造成 CUDA event wait。
- `eviction_damage_miss_ratio`: 被 eviction 过、随后又以 miss 形式被请求的 expert 占比，用来定位 cache pollution 是否转化为后续 miss。

## 边界

这个 trace 比 router replay 更接近真实系统，因为它来自 live cache/load 决策路径。但它仍然不是完整 PCIe/stream timing trace：它能证明 cache residency、CPU/GPU fallback、预取候选是否被下一层使用，不能单独证明端到端 latency benefit。论文里的最终链条仍需要结合 latency、stall time 或 token throughput。
