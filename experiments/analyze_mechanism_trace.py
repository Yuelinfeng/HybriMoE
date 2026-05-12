#!/usr/bin/env python3
"""Diagnose mechanisms behind high cache residency but low prefetch utility."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _int_list(payload: dict[str, Any], key: str) -> list[int]:
    return [int(item) for item in payload.get(key, [])]


def _count_map(payload: dict[str, Any]) -> dict[int, int]:
    return {int(expert): int(count) for expert, count in payload.get("assignment_counts", [])}


def _source(payload: dict[str, Any]) -> str:
    if payload.get("source") is not None:
        return str(payload["source"])
    metadata = payload.get("metadata") or {}
    return str(metadata.get("source", "unknown"))


def _percent(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def _event_sort_key(payload: dict[str, Any]) -> tuple[int, int]:
    return int(payload.get("time_ns", 0)), int(payload.get("event_id", 0))


def _stage_ok(payload: dict[str, Any], stage: str) -> bool:
    return stage == "all" or payload.get("stage") == stage


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return sorted(events, key=_event_sort_key)


def load_router_jaccards(path: Path) -> dict[str, Any]:
    by_layer: dict[int, list[set[int]]] = defaultdict(list)
    unique_experts: set[int] = set()
    decode_events = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("schema") != "hybrimoe.router_assignments.v1" or event.get("stage") != "decode":
                continue
            selected = {int(expert) for expert, count in event.get("expert_counts", []) if int(count) > 0}
            unique_experts.update(selected)
            by_layer[int(event["layer_idx"])].append(selected)
            decode_events += 1

    jaccards = []
    for sets in by_layer.values():
        for previous, current in zip(sets, sets[1:]):
            union = previous | current
            jaccards.append(len(previous & current) / len(union) if union else 1.0)

    return {
        "decode_events": decode_events,
        "unique_experts": len(unique_experts),
        "avg_same_layer_jaccard": mean(jaccards) if jaccards else 0.0,
        "route_drift_ratio": 1.0 - (mean(jaccards) if jaccards else 0.0),
        "same_layer_jaccard_samples": len(jaccards),
    }


def match_prefetch(events: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    pending: dict[int, deque[dict[str, Any]]] = defaultdict(deque)
    totals = defaultdict(int)

    for event in events:
        event_type = event.get("event_type")
        if event_type == "prefetch_issue":
            target = event.get("target_layer_idx")
            if target is not None:
                pending[int(target)].append(event)
            continue
        if event_type != "placement" or not _stage_ok(event, stage):
            continue
        layer_idx = event.get("layer_idx")
        if layer_idx is None or not pending[int(layer_idx)]:
            continue

        prefetch = pending[int(layer_idx)].popleft()
        candidates = set(_int_list(prefetch, "prefetch_experts"))
        issued = set(_int_list(prefetch, "issued_load_experts"))
        selected = set(_int_list(event, "selected_experts"))
        counts = _count_map(event)
        assignment_count = int(event.get("assignment_count", len(selected)))
        candidate_overlap = candidates & selected
        issued_overlap = issued & selected

        totals["matched_windows"] += 1
        totals["candidates"] += len(candidates)
        totals["issued"] += len(issued)
        totals["candidate_overlap"] += len(candidate_overlap)
        totals["issued_overlap"] += len(issued_overlap)
        totals["assignments"] += assignment_count
        totals["candidate_overlap_assignments"] += sum(counts.get(expert, 0) for expert in candidate_overlap)
        totals["issued_overlap_assignments"] += sum(counts.get(expert, 0) for expert in issued_overlap)

    issued_used = _ratio(totals["issued_overlap"], totals["issued"])
    return {
        "matched_prefetch_windows": totals["matched_windows"],
        "prefetch_candidate_used_ratio": _ratio(totals["candidate_overlap"], totals["candidates"]),
        "prefetch_issued_used_ratio": issued_used,
        "prefetch_candidate_assignment_coverage": _ratio(totals["candidate_overlap_assignments"], totals["assignments"]),
        "prefetch_issued_assignment_coverage": _ratio(totals["issued_overlap_assignments"], totals["assignments"]),
        "prefetch_issued_unused_ratio": 1.0 - issued_used if totals["issued"] else 0.0,
    }


def summarize_expert_events(events: list[dict[str, Any]], late_wait_ms: float, stage: str) -> dict[str, Any]:
    totals = defaultdict(int)
    load_by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    prefetch_wait_ms = []
    expert_wait_ms = []
    selected_experts = 0
    resident_hits = 0
    resident_misses = 0
    assignments = 0
    cpu_assignments = 0
    gpu_assignments = 0
    prefetch_candidates = 0
    prefetch_resident = 0
    prefetch_issued = 0

    for event in events:
        event_type = event.get("event_type")
        totals[event_type] += 1
        if event_type == "placement" and _stage_ok(event, stage):
            selected_experts += int(event.get("selected_expert_count", len(_int_list(event, "selected_experts"))))
            resident_hits += int(event.get("resident_hit_expert_count", len(_int_list(event, "resident_hit_experts"))))
            resident_misses += int(event.get("resident_miss_expert_count", len(_int_list(event, "resident_miss_experts"))))
            assignments += int(event.get("assignment_count", 0))
            cpu_assignments += int(event.get("cpu_assignment_count", 0))
            gpu_assignments += int(event.get("gpu_assignment_count", 0))
        elif event_type == "prefetch_issue":
            prefetch_candidates += int(event.get("prefetch_expert_count", len(_int_list(event, "prefetch_experts"))))
            prefetch_resident += int(
                event.get("prefetch_resident_expert_count", len(_int_list(event, "prefetch_resident_experts")))
            )
            prefetch_issued += int(event.get("issued_load_expert_count", len(_int_list(event, "issued_load_experts"))))
        elif event_type in {"load", "evict", "buffer_load"}:
            source = _source(event)
            load_by_source[source]["events"] += 1
            load_by_source[source]["loaded"] += int(event.get("loaded_expert_count", 0))
            load_by_source[source]["evicted"] += int(event.get("evicted_expert_count", 0))
            load_by_source[source]["buffered"] += int(event.get("buffered_expert_count", 0))
        elif event_type == "prefetch_wait" and _stage_ok(event, stage):
            prefetch_wait_ms.append(float(event.get("wait_ms", 0.0)))
        elif event_type == "expert_wait" and _stage_ok(event, stage):
            expert_wait_ms.append(float(event.get("wait_ms", 0.0)))

    def wait_summary(values: list[float]) -> dict[str, Any]:
        if not values:
            return {
                "events": 0,
                "total_ms": None,
                "avg_ms": None,
                "max_ms": None,
                "late_ratio": None,
            }
        sorted_values = sorted(values)
        return {
            "events": len(values),
            "total_ms": sum(values),
            "avg_ms": mean(values),
            "max_ms": max(values),
            "p95_ms": sorted_values[int(0.95 * (len(sorted_values) - 1))],
            "late_ratio": sum(value > late_wait_ms for value in values) / len(values),
        }

    return {
        "event_counts": dict(totals),
        "placement_events": sum(1 for event in events if event.get("event_type") == "placement" and _stage_ok(event, stage)),
        "live_cache_hit_ratio": _ratio(resident_hits, selected_experts),
        "live_cache_miss_ratio": _ratio(resident_misses, selected_experts),
        "cpu_assignment_ratio": _ratio(cpu_assignments, assignments),
        "gpu_assignment_ratio": _ratio(gpu_assignments, assignments),
        "prefetch_candidates": prefetch_candidates,
        "prefetch_issued": prefetch_issued,
        "prefetch_redundant_ratio": _ratio(prefetch_resident, prefetch_candidates),
        "prefetch_issued_candidate_ratio": _ratio(prefetch_issued, prefetch_candidates),
        "load_by_source": {source: dict(values) for source, values in sorted(load_by_source.items())},
        "prefetch_wait": wait_summary(prefetch_wait_ms),
        "expert_wait": wait_summary(expert_wait_ms),
    }


def eviction_damage(events: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    evicted: dict[int, set[int]] = defaultdict(set)
    totals = defaultdict(int)

    for event in events:
        event_type = event.get("event_type")
        layer_idx = event.get("layer_idx")
        if layer_idx is None:
            continue
        layer_idx = int(layer_idx)

        if event_type == "evict":
            for expert in _int_list(event, "evicted_experts"):
                evicted[layer_idx].add(expert)
        elif event_type == "load":
            for expert in _int_list(event, "loaded_experts"):
                evicted[layer_idx].discard(expert)
        elif event_type == "placement" and _stage_ok(event, stage):
            misses = set(_int_list(event, "resident_miss_experts"))
            counts = _count_map(event)
            totals["resident_miss_experts"] += len(misses)
            totals["resident_miss_assignments"] += sum(counts.get(expert, 0) for expert in misses)
            damaged = misses & evicted[layer_idx]
            totals["eviction_damaged_miss_experts"] += len(damaged)
            totals["eviction_damaged_miss_assignments"] += sum(counts.get(expert, 0) for expert in damaged)
            for expert in damaged:
                evicted[layer_idx].discard(expert)

    return {
        **dict(totals),
        "eviction_damage_miss_ratio": _ratio(
            totals["eviction_damaged_miss_experts"], totals["resident_miss_experts"]
        ),
        "eviction_damage_assignment_ratio": _ratio(
            totals["eviction_damaged_miss_assignments"], totals["resident_miss_assignments"]
        ),
    }


def analyze_run(run_dir: Path, late_wait_ms: float, stage: str) -> dict[str, Any]:
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    expert_events = load_jsonl(run_dir / "expert_cache_trace" / "expert_cache_trace.jsonl")
    expert_summary = summarize_expert_events(expert_events, late_wait_ms, stage)
    prefetch_match = match_prefetch(expert_events, stage)
    damage = eviction_damage(expert_events, stage)
    route = load_router_jaccards(run_dir / "router_trace" / "router_trace.jsonl")
    return {
        "run": run_dir.name,
        "cache_size": int(cfg["cache_size"]),
        "prefetch_size": int(cfg["prefetch_size"]),
        "stage": stage,
        **expert_summary,
        **prefetch_match,
        **damage,
        **route,
    }


def find_runs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir is not None:
        return [args.run_dir]
    return sorted(
        path
        for path in args.input_root.iterdir()
        if path.is_dir()
        and (path / "run_config.json").exists()
        and (path / "expert_cache_trace" / "expert_cache_trace.jsonl").exists()
    )


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run",
        "cache_size",
        "prefetch_size",
        "stage",
        "placement_events",
        "decode_events",
        "live_cache_hit_ratio",
        "prefetch_redundant_ratio",
        "prefetch_issued_candidate_ratio",
        "prefetch_issued_used_ratio",
        "prefetch_issued_assignment_coverage",
        "prefetch_issued_unused_ratio",
        "route_drift_ratio",
        "avg_same_layer_jaccard",
        "eviction_damage_miss_ratio",
        "eviction_damage_assignment_ratio",
        "cpu_assignment_ratio",
        "prefetch_wait_late_ratio",
        "expert_wait_late_ratio",
        "prefetch_wait_total_ms",
        "expert_wait_total_ms",
    ]
    rows = []
    for summary in summaries:
        rows.append(
            {
                **{name: summary.get(name) for name in fieldnames},
                "prefetch_wait_late_ratio": summary["prefetch_wait"].get("late_ratio"),
                "expert_wait_late_ratio": summary["expert_wait"].get("late_ratio"),
                "prefetch_wait_total_ms": summary["prefetch_wait"].get("total_ms"),
                "expert_wait_total_ms": summary["expert_wait"].get("total_ms"),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_load_source_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = ["run", "cache_size", "prefetch_size", "source", "events", "loaded", "evicted", "buffered"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            for source, values in summary.get("load_by_source", {}).items():
                writer.writerow(
                    {
                        "run": summary["run"],
                        "cache_size": summary["cache_size"],
                        "prefetch_size": summary["prefetch_size"],
                        "source": source,
                        "events": values.get("events", 0),
                        "loaded": values.get("loaded", 0),
                        "evicted": values.get("evicted", 0),
                        "buffered": values.get("buffered", 0),
                    }
                )


def write_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Mechanism Diagnosis Report",
        "",
        "This report separates route drift, redundant candidates, unused issued prefetches, eviction damage, and timing wait evidence.",
        "",
        "| run | hit | redundant | issued used | issued coverage | route drift | eviction damage | prefetch wait late | expert wait late |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary["run"],
                    _percent(summary["live_cache_hit_ratio"]),
                    _percent(summary["prefetch_redundant_ratio"]),
                    _percent(summary["prefetch_issued_used_ratio"]),
                    _percent(summary["prefetch_issued_assignment_coverage"]),
                    _percent(summary["route_drift_ratio"]),
                    _percent(summary["eviction_damage_miss_ratio"]),
                    _percent(summary["prefetch_wait"].get("late_ratio")),
                    _percent(summary["expert_wait"].get("late_ratio")),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- `redundant` measures candidates already resident at issue time.",
        "- `issued used` measures issued load experts selected by the next matching placement.",
        "- `issued coverage` measures how many next-layer assignments are covered by issued prefetches.",
        "- `route drift` is `1 - same-layer consecutive decode Jaccard`.",
        "- `eviction damage` counts resident misses whose expert had been evicted since its last load.",
        "- wait ratios require traces generated after timing instrumentation; older traces show `n/a`.",
        "- `load_source_summary.csv` separates future traces by `demand`, `prefetch`, `init`, and `prefill_buffer` sources.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_mechanism_figure(path_prefix: Path, focus: dict[str, Any]) -> None:
    metrics = [
        ("Route drift", focus["route_drift_ratio"], "#4C78A8"),
        ("Redundant candidates", focus["prefetch_redundant_ratio"], "#F58518"),
        ("Unused issued prefetch", focus["prefetch_issued_unused_ratio"], "#E45756"),
        ("Eviction-damaged misses", focus["eviction_damage_miss_ratio"], "#72B7B2"),
        ("Prefetch wait late", focus["prefetch_wait"].get("late_ratio"), "#54A24B"),
        ("Expert wait late", focus["expert_wait"].get("late_ratio"), "#B279A2"),
    ]
    labels = [item[0] for item in metrics]
    values = [item[1] for item in metrics]
    colors = [item[2] for item in metrics]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    y = list(range(len(labels)))
    plotted_values = [0.0 if value is None else value * 100 for value in values]
    ax.barh(y, plotted_values, color=colors, alpha=0.88)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mechanism signal (%)")
    ax.set_title(f"Mechanism Diagnosis: {focus['run']}", loc="left")
    ax.set_xlim(0, 105)
    for idx, value in enumerate(values):
        if value is None:
            ax.text(1.0, idx, "n/a", va="center", ha="left", fontsize=7)
        else:
            ax.text(value * 100 + 1.0, idx, f"{value * 100:.1f}", va="center", ha="left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def choose_focus(summaries: list[dict[str, Any]], focus_run: str | None) -> dict[str, Any]:
    if focus_run:
        for summary in summaries:
            if summary["run"] == focus_run:
                return summary
        raise ValueError(f"focus run not found: {focus_run}")
    return sorted(summaries, key=lambda item: (item["live_cache_hit_ratio"], item["prefetch_size"]))[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-run")
    parser.add_argument("--stage", choices=["all", "prefill", "decode"], default="decode")
    parser.add_argument("--late-wait-ms", type=float, default=0.05)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [analyze_run(run_dir, args.late_wait_ms, args.stage) for run_dir in find_runs(args)]
    summaries.sort(key=lambda item: (item["cache_size"], item["prefetch_size"], item["run"]))

    (args.output_dir / "mechanism_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_summary_csv(args.output_dir / "mechanism_summary.csv", summaries)
    write_load_source_csv(args.output_dir / "load_source_summary.csv", summaries)
    write_report(args.output_dir / "mechanism_report.md", summaries)
    plot_mechanism_figure(args.output_dir / "mechanism_figure", choose_focus(summaries, args.focus_run))

    if args.print_json:
        print(json.dumps(summaries, indent=2))
    else:
        print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
