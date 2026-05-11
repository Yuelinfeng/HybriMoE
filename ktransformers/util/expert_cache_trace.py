"""Optional expert cache/load tracing for HybriMoE utility analysis."""

from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path
from typing import Any


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _to_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach().to("cpu").reshape(-1).tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return [_to_int(values)]
    return [_to_int(value) for value in values]


def _assignment_count_pairs(assignment_counts: Any, expert_num: int | None = None) -> list[list[int]]:
    if assignment_counts is None:
        return []
    if isinstance(assignment_counts, dict):
        return [[_to_int(key), _to_int(value)] for key, value in assignment_counts.items() if _to_int(value) > 0]
    if hasattr(assignment_counts, "detach"):
        values = assignment_counts.detach().to("cpu").reshape(-1).tolist()
    elif hasattr(assignment_counts, "tolist"):
        values = assignment_counts.tolist()
    else:
        values = assignment_counts
    if not values:
        return []
    if isinstance(values[0], (list, tuple)) and len(values[0]) == 2:
        return [[_to_int(expert), _to_int(count)] for expert, count in values if _to_int(count) > 0]
    if expert_num is None:
        expert_num = len(values)
    return [[idx, _to_int(values[idx])] for idx in range(min(len(values), expert_num)) if _to_int(values[idx]) > 0]


def _sum_assignments(experts: list[int], count_map: dict[int, int]) -> int:
    if not count_map:
        return len(experts)
    return sum(count_map.get(expert, 0) for expert in experts)


class ExpertCacheTraceRecorder:
    def __init__(self) -> None:
        self._configured = False
        self._enabled = False
        self._path: Path | None = None
        self._fh = None
        self._event_id = 0
        self._max_records = 0

    def enabled(self) -> bool:
        self._configure_once()
        return self._enabled

    def record(
        self,
        *,
        event_type: str,
        layer_idx: int | None = None,
        target_layer_idx: int | None = None,
        generate: bool | None = None,
        selected_experts: Any = None,
        resident_hit_experts: Any = None,
        resident_miss_experts: Any = None,
        gpu_experts: Any = None,
        cpu_experts: Any = None,
        assignment_counts: Any = None,
        prefetch_experts: Any = None,
        prefetch_resident_experts: Any = None,
        issued_load_experts: Any = None,
        loaded_experts: Any = None,
        evicted_experts: Any = None,
        buffered_experts: Any = None,
        cache_load_size: int | None = None,
        prefetch_size: int | None = None,
        expert_num: int | None = None,
        device: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._configure_once()
        if not self._enabled:
            return
        if self._max_records and self._event_id >= self._max_records:
            return

        selected = _to_int_list(selected_experts)
        resident_hits = _to_int_list(resident_hit_experts)
        resident_misses = _to_int_list(resident_miss_experts)
        gpu = _to_int_list(gpu_experts)
        cpu = _to_int_list(cpu_experts)
        prefetch = _to_int_list(prefetch_experts)
        prefetch_resident = _to_int_list(prefetch_resident_experts)
        issued_loads = _to_int_list(issued_load_experts)
        loaded = _to_int_list(loaded_experts)
        evicted = _to_int_list(evicted_experts)
        buffered = _to_int_list(buffered_experts)
        count_pairs = _assignment_count_pairs(assignment_counts, expert_num)
        if selected:
            selected_set = set(selected)
            count_pairs = [item for item in count_pairs if item[0] in selected_set]
        count_map = {expert: count for expert, count in count_pairs}

        payload: dict[str, Any] = {
            "schema": "hybrimoe.expert_cache.v1",
            "event_id": self._event_id,
            "time_ns": time.time_ns(),
            "pid": os.getpid(),
            "event_type": event_type,
        }
        if layer_idx is not None:
            payload["layer_idx"] = int(layer_idx)
        if target_layer_idx is not None:
            payload["target_layer_idx"] = int(target_layer_idx)
        if generate is not None:
            payload["generate"] = bool(generate)
            payload["stage"] = "decode" if generate else "prefill"
        if device is not None:
            payload["device"] = str(device)
        if cache_load_size is not None:
            payload["cache_load_size"] = int(cache_load_size)
        if prefetch_size is not None:
            payload["prefetch_size"] = int(prefetch_size)
        if expert_num is not None:
            payload["expert_num"] = int(expert_num)
        if metadata:
            payload["metadata"] = metadata

        if selected or resident_hits or resident_misses or gpu or cpu:
            selected_total = len(resident_hits) + len(resident_misses)
            payload.update(
                {
                    "selected_experts": selected,
                    "selected_expert_count": len(selected),
                    "resident_hit_experts": resident_hits,
                    "resident_miss_experts": resident_misses,
                    "resident_hit_expert_count": len(resident_hits),
                    "resident_miss_expert_count": len(resident_misses),
                    "resident_hit_expert_ratio": (len(resident_hits) / selected_total) if selected_total else 0.0,
                    "gpu_experts": gpu,
                    "cpu_experts": cpu,
                    "gpu_expert_count": len(gpu),
                    "cpu_expert_count": len(cpu),
                    "assignment_counts": count_pairs,
                    "assignment_count": sum(count_map.values()) if count_map else len(selected),
                    "resident_hit_assignment_count": _sum_assignments(resident_hits, count_map),
                    "resident_miss_assignment_count": _sum_assignments(resident_misses, count_map),
                    "gpu_assignment_count": _sum_assignments(gpu, count_map),
                    "cpu_assignment_count": _sum_assignments(cpu, count_map),
                }
            )

        if prefetch or prefetch_resident or issued_loads:
            payload.update(
                {
                    "prefetch_experts": prefetch,
                    "prefetch_expert_count": len(prefetch),
                    "prefetch_resident_experts": prefetch_resident,
                    "prefetch_resident_expert_count": len(prefetch_resident),
                    "issued_load_experts": issued_loads,
                    "issued_load_expert_count": len(issued_loads),
                    "prefetch_redundant_candidate_ratio": (
                        len(prefetch_resident) / len(prefetch) if prefetch else 0.0
                    ),
                }
            )

        if loaded or evicted or buffered:
            payload.update(
                {
                    "loaded_experts": loaded,
                    "loaded_expert_count": len(loaded),
                    "evicted_experts": evicted,
                    "evicted_expert_count": len(evicted),
                    "buffered_experts": buffered,
                    "buffered_expert_count": len(buffered),
                }
            )

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
        self._enabled = _env_flag("HYBRIMOE_EXPERT_TRACE", False)
        if not self._enabled:
            return
        raw_path = (
            os.environ.get("HYBRIMOE_EXPERT_TRACE_PATH")
            or os.environ.get("HYBRIMOE_EXPERT_TRACE_OUTPUT")
            or "results/expert_cache_trace/expert_cache_trace.jsonl"
        )
        self._path = Path(raw_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")
        self._max_records = int(os.environ.get("HYBRIMOE_EXPERT_TRACE_MAX_RECORDS", "0"))

    def _write(self, payload: dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._fh.flush()


_RECORDER = ExpertCacheTraceRecorder()
atexit.register(_RECORDER.close)


def trace_expert_cache_event(**kwargs: Any) -> None:
    if _RECORDER.enabled():
        _RECORDER.record(**kwargs)
