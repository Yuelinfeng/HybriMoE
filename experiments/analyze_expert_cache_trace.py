#!/usr/bin/env python3
"""Analyze live HybriMoE expert cache/load trace events."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _int_list(payload: dict[str, Any], key: str) -> list[int]:
    return [int(item) for item in payload.get(key, [])]


def _count_map(payload: dict[str, Any]) -> dict[int, int]:
    return {int(expert): int(count) for expert, count in payload.get("assignment_counts", [])}


def _sum_assignments(experts: set[int], counts: dict[int, int]) -> int:
    if not counts:
        return len(experts)
    return sum(counts.get(expert, 0) for expert in experts)


def load_events(path: Path, *, stage: str, max_events: int) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    skipped = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("schema") != "hybrimoe.expert_cache.v1":
                skipped += 1
                continue
            if payload.get("event_type") == "placement" and stage != "all" and payload.get("stage") != stage:
                skipped += 1
                continue
            events.append(payload)
            if max_events and len(events) >= max_events:
                break
    events.sort(key=lambda item: int(item.get("event_id", 0)))
    return events, skipped


def summarize_placements(events: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    totals = defaultdict(int)
    by_stage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for event in events:
        if event.get("event_type") != "placement":
            continue
        stage = str(event.get("stage", "unknown"))
        selected_count = int(event.get("selected_expert_count", len(_int_list(event, "selected_experts"))))
        hit_count = int(event.get("resident_hit_expert_count", len(_int_list(event, "resident_hit_experts"))))
        miss_count = int(event.get("resident_miss_expert_count", len(_int_list(event, "resident_miss_experts"))))
        assignment_count = int(event.get("assignment_count", selected_count))
        hit_assign = int(event.get("resident_hit_assignment_count", 0))
        miss_assign = int(event.get("resident_miss_assignment_count", 0))
        gpu_assign = int(event.get("gpu_assignment_count", 0))
        cpu_assign = int(event.get("cpu_assignment_count", 0))

        totals["events"] += 1
        totals["selected_experts"] += selected_count
        totals["resident_hit_experts"] += hit_count
        totals["resident_miss_experts"] += miss_count
        totals["assignments"] += assignment_count
        totals["resident_hit_assignments"] += hit_assign
        totals["resident_miss_assignments"] += miss_assign
        totals["gpu_assignments"] += gpu_assign
        totals["cpu_assignments"] += cpu_assign

        stage_bucket = by_stage[stage]
        stage_bucket["events"] += 1
        stage_bucket["selected_experts"] += selected_count
        stage_bucket["resident_hit_experts"] += hit_count
        stage_bucket["assignments"] += assignment_count
        stage_bucket["resident_hit_assignments"] += hit_assign
        stage_bucket["gpu_assignments"] += gpu_assign
        stage_bucket["cpu_assignments"] += cpu_assign

        rows.append(
            {
                "event_id": event.get("event_id"),
                "stage": stage,
                "layer_idx": event.get("layer_idx"),
                "selected_expert_count": selected_count,
                "resident_hit_expert_count": hit_count,
                "resident_miss_expert_count": miss_count,
                "resident_hit_expert_ratio": _ratio(hit_count, hit_count + miss_count),
                "assignment_count": assignment_count,
                "resident_hit_assignment_count": hit_assign,
                "resident_miss_assignment_count": miss_assign,
                "resident_hit_assignment_ratio": _ratio(hit_assign, assignment_count),
                "gpu_assignment_count": gpu_assign,
                "cpu_assignment_count": cpu_assign,
                "gpu_assignment_ratio": _ratio(gpu_assign, assignment_count),
                "cpu_assignment_ratio": _ratio(cpu_assign, assignment_count),
                "cache_load_size": event.get("cache_load_size"),
                "prefetch_size": event.get("prefetch_size"),
            }
        )

    summary = {
        "placement_events": totals["events"],
        "selected_experts": totals["selected_experts"],
        "resident_hit_experts": totals["resident_hit_experts"],
        "resident_miss_experts": totals["resident_miss_experts"],
        "resident_hit_expert_ratio": _ratio(totals["resident_hit_experts"], totals["selected_experts"]),
        "assignments": totals["assignments"],
        "resident_hit_assignments": totals["resident_hit_assignments"],
        "resident_miss_assignments": totals["resident_miss_assignments"],
        "resident_hit_assignment_ratio": _ratio(totals["resident_hit_assignments"], totals["assignments"]),
        "gpu_assignments": totals["gpu_assignments"],
        "cpu_assignments": totals["cpu_assignments"],
        "gpu_assignment_ratio": _ratio(totals["gpu_assignments"], totals["assignments"]),
        "cpu_assignment_ratio": _ratio(totals["cpu_assignments"], totals["assignments"]),
        "by_stage": {
            stage: {
                **dict(bucket),
                "resident_hit_expert_ratio": _ratio(bucket["resident_hit_experts"], bucket["selected_experts"]),
                "resident_hit_assignment_ratio": _ratio(bucket["resident_hit_assignments"], bucket["assignments"]),
                "gpu_assignment_ratio": _ratio(bucket["gpu_assignments"], bucket["assignments"]),
                "cpu_assignment_ratio": _ratio(bucket["cpu_assignments"], bucket["assignments"]),
            }
            for stage, bucket in sorted(by_stage.items())
        },
    }
    return summary, rows


def summarize_prefetch_issues(events: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    totals = defaultdict(int)
    for event in events:
        if event.get("event_type") != "prefetch_issue":
            continue
        candidates = int(event.get("prefetch_expert_count", len(_int_list(event, "prefetch_experts"))))
        already_resident = int(
            event.get("prefetch_resident_expert_count", len(_int_list(event, "prefetch_resident_experts")))
        )
        issued = int(event.get("issued_load_expert_count", len(_int_list(event, "issued_load_experts"))))
        totals["events"] += 1
        totals["candidates"] += candidates
        totals["already_resident"] += already_resident
        totals["issued"] += issued
        rows.append(
            {
                "event_id": event.get("event_id"),
                "source_layer_idx": event.get("layer_idx"),
                "target_layer_idx": event.get("target_layer_idx"),
                "prefetch_expert_count": candidates,
                "prefetch_resident_expert_count": already_resident,
                "issued_load_expert_count": issued,
                "prefetch_redundant_candidate_ratio": _ratio(already_resident, candidates),
                "cache_load_size": event.get("cache_load_size"),
                "prefetch_size": event.get("prefetch_size"),
            }
        )

    summary = {
        "prefetch_issue_events": totals["events"],
        "prefetch_candidate_experts": totals["candidates"],
        "prefetch_already_resident_experts": totals["already_resident"],
        "prefetch_issued_load_experts": totals["issued"],
        "prefetch_redundant_candidate_ratio": _ratio(totals["already_resident"], totals["candidates"]),
        "prefetch_issued_candidate_ratio": _ratio(totals["issued"], totals["candidates"]),
    }
    return summary, rows


def summarize_load_eviction(events: list[dict[str, Any]]) -> dict[str, Any]:
    totals = defaultdict(int)
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for event in events:
        event_type = str(event.get("event_type", ""))
        if event_type not in {"load", "buffer_load", "evict"}:
            continue
        loaded = int(event.get("loaded_expert_count", len(_int_list(event, "loaded_experts"))))
        buffered = int(event.get("buffered_expert_count", len(_int_list(event, "buffered_experts"))))
        evicted = int(event.get("evicted_expert_count", len(_int_list(event, "evicted_experts"))))
        totals["events"] += 1
        totals["loaded_experts"] += loaded
        totals["buffered_experts"] += buffered
        totals["evicted_experts"] += evicted
        by_type[event_type]["events"] += 1
        by_type[event_type]["loaded_experts"] += loaded
        by_type[event_type]["buffered_experts"] += buffered
        by_type[event_type]["evicted_experts"] += evicted
    return {
        "load_evict_events": totals["events"],
        "loaded_experts": totals["loaded_experts"],
        "buffered_experts": totals["buffered_experts"],
        "evicted_experts": totals["evicted_experts"],
        "evictions_per_loaded_expert": _ratio(totals["evicted_experts"], totals["loaded_experts"]),
        "by_type": {event_type: dict(bucket) for event_type, bucket in sorted(by_type.items())},
    }


def match_prefetch_to_next_placement(events: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pending: dict[int, deque[dict[str, Any]]] = defaultdict(deque)
    rows = []
    totals = defaultdict(int)

    for event in events:
        event_type = event.get("event_type")
        if event_type == "prefetch_issue":
            target_layer_idx = event.get("target_layer_idx")
            if target_layer_idx is not None:
                pending[int(target_layer_idx)].append(event)
            continue
        if event_type != "placement":
            continue
        layer_idx = event.get("layer_idx")
        if layer_idx is None or not pending[int(layer_idx)]:
            continue

        prefetch = pending[int(layer_idx)].popleft()
        candidates = set(_int_list(prefetch, "prefetch_experts"))
        issued = set(_int_list(prefetch, "issued_load_experts"))
        selected = set(_int_list(event, "selected_experts"))
        candidate_overlap = candidates & selected
        issued_overlap = issued & selected
        counts = _count_map(event)
        assignment_count = int(event.get("assignment_count", len(selected)))
        candidate_overlap_assignments = _sum_assignments(candidate_overlap, counts)
        issued_overlap_assignments = _sum_assignments(issued_overlap, counts)

        totals["matched_windows"] += 1
        totals["prefetch_candidates"] += len(candidates)
        totals["issued_loads"] += len(issued)
        totals["candidate_overlap"] += len(candidate_overlap)
        totals["issued_overlap"] += len(issued_overlap)
        totals["placement_assignments"] += assignment_count
        totals["candidate_overlap_assignments"] += candidate_overlap_assignments
        totals["issued_overlap_assignments"] += issued_overlap_assignments

        rows.append(
            {
                "prefetch_event_id": prefetch.get("event_id"),
                "placement_event_id": event.get("event_id"),
                "target_layer_idx": layer_idx,
                "stage": event.get("stage"),
                "prefetch_candidate_count": len(candidates),
                "issued_load_count": len(issued),
                "selected_expert_count": len(selected),
                "candidate_overlap_count": len(candidate_overlap),
                "issued_overlap_count": len(issued_overlap),
                "candidate_overlap_ratio": _ratio(len(candidate_overlap), len(candidates)),
                "issued_overlap_ratio": _ratio(len(issued_overlap), len(issued)),
                "placement_assignment_count": assignment_count,
                "candidate_overlap_assignment_count": candidate_overlap_assignments,
                "issued_overlap_assignment_count": issued_overlap_assignments,
                "candidate_overlap_assignment_ratio": _ratio(candidate_overlap_assignments, assignment_count),
                "issued_overlap_assignment_ratio": _ratio(issued_overlap_assignments, assignment_count),
            }
        )

    summary = {
        "matched_prefetch_windows": totals["matched_windows"],
        "matched_prefetch_candidates": totals["prefetch_candidates"],
        "matched_issued_loads": totals["issued_loads"],
        "prefetch_candidate_used_by_next_layer_ratio": _ratio(
            totals["candidate_overlap"], totals["prefetch_candidates"]
        ),
        "prefetch_issued_used_by_next_layer_ratio": _ratio(totals["issued_overlap"], totals["issued_loads"]),
        "prefetch_candidate_assignment_coverage_ratio": _ratio(
            totals["candidate_overlap_assignments"], totals["placement_assignments"]
        ),
        "prefetch_issued_assignment_coverage_ratio": _ratio(
            totals["issued_overlap_assignments"], totals["placement_assignments"]
        ),
        "candidate_overlap_experts": totals["candidate_overlap"],
        "issued_overlap_experts": totals["issued_overlap"],
        "matched_placement_assignments": totals["placement_assignments"],
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Expert Cache Trace Analysis",
        "",
        "This report uses live HybriMoE expert placement and prefetch events captured inside `KExpertsMarlin`.",
        "",
        "## Core Metrics",
        "",
        f"- resident_hit_expert_ratio: {summary['placement'].get('resident_hit_expert_ratio', 0.0):.6f}",
        f"- resident_hit_assignment_ratio: {summary['placement'].get('resident_hit_assignment_ratio', 0.0):.6f}",
        f"- cpu_assignment_ratio: {summary['placement'].get('cpu_assignment_ratio', 0.0):.6f}",
        f"- gpu_assignment_ratio: {summary['placement'].get('gpu_assignment_ratio', 0.0):.6f}",
        f"- prefetch_redundant_candidate_ratio: {summary['prefetch'].get('prefetch_redundant_candidate_ratio', 0.0):.6f}",
        f"- prefetch_candidate_used_by_next_layer_ratio: {summary['prefetch_match'].get('prefetch_candidate_used_by_next_layer_ratio', 0.0):.6f}",
        f"- prefetch_issued_used_by_next_layer_ratio: {summary['prefetch_match'].get('prefetch_issued_used_by_next_layer_ratio', 0.0):.6f}",
        f"- prefetch_candidate_assignment_coverage_ratio: {summary['prefetch_match'].get('prefetch_candidate_assignment_coverage_ratio', 0.0):.6f}",
        f"- loaded_experts: {summary['load_evict'].get('loaded_experts', 0)}",
        f"- evicted_experts: {summary['load_evict'].get('evicted_experts', 0)}",
        f"- evictions_per_loaded_expert: {summary['load_evict'].get('evictions_per_loaded_expert', 0.0):.6f}",
        "",
        "## Interpretation Boundary",
        "",
        "These are live placement and prefetch-decision metrics, not low-level PCIe timing metrics. They validate whether cached/prefetched experts are selected by the next layer and whether HybriMoE falls back to CPU/GPU execution, but they do not by themselves prove end-to-end latency benefit.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path, help="Path to expert_cache_trace.jsonl")
    parser.add_argument("--stage", choices=["all", "prefill", "decode"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("results/expert_cache_trace_analysis"))
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    events, skipped = load_events(args.trace, stage=args.stage, max_events=args.max_events)
    if not events:
        raise SystemExit("No usable expert cache trace events found.")

    placement_summary, placement_rows = summarize_placements(events)
    prefetch_summary, prefetch_rows = summarize_prefetch_issues(events)
    match_summary, match_rows = match_prefetch_to_next_placement(events)
    load_evict_summary = summarize_load_eviction(events)
    summary = {
        "trace_path": str(args.trace),
        "stage_filter": args.stage,
        "events_used": len(events),
        "events_skipped": skipped,
        "placement": placement_summary,
        "prefetch": prefetch_summary,
        "prefetch_match": match_summary,
        "load_evict": load_evict_summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "placement_events.csv", placement_rows)
    write_csv(args.output_dir / "prefetch_issues.csv", prefetch_rows)
    write_csv(args.output_dir / "prefetch_to_next_placement.csv", match_rows)
    write_report(args.output_dir / "analysis_report.md", summary)

    if args.print_json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"events_used={summary['events_used']} skipped={summary['events_skipped']}")
        print(f"resident_hit_expert_ratio={placement_summary.get('resident_hit_expert_ratio', 0.0):.6f}")
        print(f"resident_hit_assignment_ratio={placement_summary.get('resident_hit_assignment_ratio', 0.0):.6f}")
        print(f"cpu_assignment_ratio={placement_summary.get('cpu_assignment_ratio', 0.0):.6f}")
        print(
            "prefetch_candidate_used_by_next_layer_ratio="
            f"{match_summary.get('prefetch_candidate_used_by_next_layer_ratio', 0.0):.6f}"
        )
        print(
            "prefetch_issued_used_by_next_layer_ratio="
            f"{match_summary.get('prefetch_issued_used_by_next_layer_ratio', 0.0):.6f}"
        )
        print(f"loaded_experts={load_evict_summary.get('loaded_experts', 0)}")
        print(f"evicted_experts={load_evict_summary.get('evicted_experts', 0)}")
        print(f"wrote={args.output_dir}")


if __name__ == "__main__":
    main()
