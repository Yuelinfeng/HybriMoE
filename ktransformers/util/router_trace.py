"""Optional router-assignment tracing for MoE prefetch utility analysis."""

from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from ktransformers.util.trace_context import get_trace_context


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class RouterTraceRecorder:
    def __init__(self) -> None:
        self._configured = False
        self._enabled = False
        self._path: Path | None = None
        self._fh = None
        self._event_id = 0
        self._max_records = 0
        self._detail = "summary"
        self._max_tokens_per_event = 0

    def enabled(self) -> bool:
        self._configure_once()
        return self._enabled

    def record(
        self,
        *,
        model_type: str,
        layer_idx: int | None,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor | None,
        num_experts: int,
        batch_size: int,
        sequence_length: int,
        top_k: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._configure_once()
        if not self._enabled:
            return
        if self._max_records and self._event_id >= self._max_records:
            return

        topk_idx_cpu = topk_idx.detach().to("cpu", dtype=torch.long)
        flat_idx = topk_idx_cpu.reshape(-1)
        counts = torch.bincount(flat_idx, minlength=num_experts)
        nonzero = torch.nonzero(counts, as_tuple=False).flatten().tolist()
        expert_counts = [[int(expert), int(counts[expert].item())] for expert in nonzero]
        top_experts = sorted(expert_counts, key=lambda item: item[1], reverse=True)[:16]

        payload: dict[str, Any] = {
            "schema": "hybrimoe.router_assignments.v1",
            "event_id": self._event_id,
            "time_ns": time.time_ns(),
            "pid": os.getpid(),
            "model_type": model_type,
            "layer_idx": layer_idx,
            "stage": "decode" if sequence_length == 1 else "prefill",
            "batch_size": int(batch_size),
            "sequence_length": int(sequence_length),
            "top_k": int(top_k),
            "num_experts": int(num_experts),
            "assignment_count": int(flat_idx.numel()),
            "unique_expert_count": len(expert_counts),
            "expert_counts": expert_counts,
            "top_experts": top_experts,
        }
        merged_metadata = get_trace_context()
        if metadata:
            merged_metadata.update(metadata)
        if merged_metadata:
            payload["metadata"] = merged_metadata

        if topk_weight is not None:
            weight_cpu = topk_weight.detach().to("cpu", dtype=torch.float32)
            payload["routing_weight_sum"] = float(weight_cpu.sum().item())
            payload["routing_weight_mean"] = float(weight_cpu.mean().item()) if weight_cpu.numel() else 0.0

        if self._detail in {"tokens", "full"} and (
            self._max_tokens_per_event <= 0 or topk_idx_cpu.shape[0] <= self._max_tokens_per_event
        ):
            payload["topk_indices"] = topk_idx_cpu.tolist()
            if topk_weight is not None and self._detail == "full":
                payload["topk_weights"] = topk_weight.detach().to("cpu", dtype=torch.float32).tolist()

        self._write(payload)
        self._event_id += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _configure_once(self) -> None:
        if self._configured:
            return
        self._configured = True
        self._enabled = _env_flag("HYBRIMOE_ROUTER_TRACE", False)
        if not self._enabled:
            return
        raw_path = (
            os.environ.get("HYBRIMOE_ROUTER_TRACE_PATH")
            or os.environ.get("HYBRIMOE_ROUTER_TRACE_OUTPUT")
            or "results/router_trace/router_trace.jsonl"
        )
        self._path = Path(raw_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")
        self._detail = os.environ.get("HYBRIMOE_ROUTER_TRACE_DETAIL", "summary").strip().lower()
        self._max_records = int(os.environ.get("HYBRIMOE_ROUTER_TRACE_MAX_RECORDS", "0"))
        self._max_tokens_per_event = int(os.environ.get("HYBRIMOE_ROUTER_TRACE_MAX_TOKENS_PER_EVENT", "0"))

    def _write(self, payload: dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._fh.flush()


_RECORDER = RouterTraceRecorder()
atexit.register(_RECORDER.close)


def trace_router_assignments(**kwargs: Any) -> None:
    if _RECORDER.enabled():
        _RECORDER.record(**kwargs)
