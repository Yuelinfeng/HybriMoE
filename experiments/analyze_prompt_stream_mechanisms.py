#!/usr/bin/env python3
"""Aggregate prompt-stream traces by workload, phase, and prompt category."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Any


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def int_list(payload: dict[str, Any], key: str) -> list[int]:
    return [int(item) for item in payload.get(key, [])]


def count_map(payload: dict[str, Any]) -> dict[int, int]:
    return {int(expert): int(count) for expert, count in payload.get("assignment_counts", [])}


def sum_assignments(experts: set[int], counts: dict[int, int]) -> int:
    if not counts:
        return len(experts)
    return sum(counts.get(expert, 0) for expert in experts)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    events.sort(key=lambda item: (int(item.get("time_ns", 0)), int(item.get("event_id", 0))))
    return events


def stage_ok(payload: dict[str, Any], stage: str) -> bool:
    return stage == "all" or payload.get("stage") == stage


def metadata(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload.get("metadata")
    return meta if isinstance(meta, dict) else {}


def prompt_scope(payload: dict[str, Any]) -> tuple[str, str]:
    meta = metadata(payload)
    return (str(meta.get("stream_id", "unknown")), str(meta.get("prompt_id", "unknown")))


def group_key(run_cfg: dict[str, Any], payload: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    meta = metadata(payload)
    run = str(run_cfg.get("run_tag") or Path(str(run_cfg.get("expert_trace", "unknown"))).parents[1].name)
    return (
        run,
        str(meta.get("stream_id", "unknown")),
        str(meta.get("workload", "unknown")),
        str(meta.get("workload_phase", meta.get("phase", "unknown"))),
        str(meta.get("prompt_category", meta.get("category", "unknown"))),
        str(meta.get("seed", "unknown")),
    )


def new_bucket(run_cfg: dict[str, Any], key: tuple[str, str, str, str, str, str]) -> dict[str, Any]:
    run, stream_id, workload, phase, category, seed = key
    return {
        "run": run,
        "stream_id": stream_id,
        "workload": workload,
        "phase": phase,
        "category": category,
        "seed": seed,
        "cache_size": int(run_cfg.get("cache_size", -1)),
        "prefetch_size": int(run_cfg.get("prefetch_size", -1)),
        "placement_events": 0,
        "assignments": 0,
        "resident_hit_assignments": 0,
        "resident_miss_assignments": 0,
        "prefetch_issue_events": 0,
        "prefetch_candidates": 0,
        "prefetch_resident": 0,
        "prefetch_issued": 0,
        "matched_prefetch_windows": 0,
        "matched_candidates": 0,
        "matched_issued": 0,
        "candidate_overlap": 0,
        "issued_overlap": 0,
        "matched_assignments": 0,
        "candidate_assignment_coverage": 0,
        "issued_assignment_coverage": 0,
        "eviction_damage_misses": 0,
        "eviction_damage_assignments": 0,
        "cpu_assignments": 0,
        "gpu_assignments": 0,
        "prefetch_wait_events": 0,
        "prefetch_wait_total_ms": 0.0,
        "prefetch_wait_late_events": 0,
        "expert_wait_events": 0,
        "expert_wait_total_ms": 0.0,
        "expert_wait_late_events": 0,
        "route_jaccard_sum": 0.0,
        "route_jaccard_samples": 0,
    }


def update_placement(bucket: dict[str, Any], event: dict[str, Any]) -> None:
    assignments = int(event.get("assignment_count", event.get("selected_expert_count", 0)))
    bucket["placement_events"] += 1
    bucket["assignments"] += assignments
    bucket["resident_hit_assignments"] += int(event.get("resident_hit_assignment_count", 0))
    bucket["resident_miss_assignments"] += int(event.get("resident_miss_assignment_count", 0))
    bucket["cpu_assignments"] += int(event.get("cpu_assignment_count", 0))
    bucket["gpu_assignments"] += int(event.get("gpu_assignment_count", 0))


def update_prefetch_issue(bucket: dict[str, Any], event: dict[str, Any]) -> None:
    bucket["prefetch_issue_events"] += 1
    bucket["prefetch_candidates"] += int(event.get("prefetch_expert_count", len(int_list(event, "prefetch_experts"))))
    bucket["prefetch_resident"] += int(
        event.get("prefetch_resident_expert_count", len(int_list(event, "prefetch_resident_experts")))
    )
    bucket["prefetch_issued"] += int(event.get("issued_load_expert_count", len(int_list(event, "issued_load_experts"))))


def update_wait(bucket: dict[str, Any], event: dict[str, Any], late_wait_ms: float) -> None:
    wait_ms = event.get("wait_ms")
    if wait_ms is None:
        return
    wait_ms = float(wait_ms)
    if event.get("event_type") == "prefetch_wait":
        bucket["prefetch_wait_events"] += 1
        bucket["prefetch_wait_total_ms"] += wait_ms
        bucket["prefetch_wait_late_events"] += int(wait_ms > late_wait_ms)
    elif event.get("event_type") == "expert_wait":
        bucket["expert_wait_events"] += 1
        bucket["expert_wait_total_ms"] += wait_ms
        bucket["expert_wait_late_events"] += int(wait_ms > late_wait_ms)


def update_prefetch_match(bucket: dict[str, Any], prefetch: dict[str, Any], placement: dict[str, Any]) -> None:
    candidates = set(int_list(prefetch, "prefetch_experts"))
    issued = set(int_list(prefetch, "issued_load_experts"))
    selected = set(int_list(placement, "selected_experts"))
    counts = count_map(placement)
    candidate_overlap = candidates & selected
    issued_overlap = issued & selected
    assignments = int(placement.get("assignment_count", len(selected)))

    bucket["matched_prefetch_windows"] += 1
    bucket["matched_candidates"] += len(candidates)
    bucket["matched_issued"] += len(issued)
    bucket["candidate_overlap"] += len(candidate_overlap)
    bucket["issued_overlap"] += len(issued_overlap)
    bucket["matched_assignments"] += assignments
    bucket["candidate_assignment_coverage"] += sum_assignments(candidate_overlap, counts)
    bucket["issued_assignment_coverage"] += sum_assignments(issued_overlap, counts)


def analyze_expert_events(
    run_cfg: dict[str, Any],
    events: list[dict[str, Any]],
    stage: str,
    late_wait_ms: float,
) -> dict[tuple[str, str, str, str, str, str], dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    pending_prefetch: dict[int, deque[dict[str, Any]]] = defaultdict(deque)
    evicted_by_layer: dict[int, set[int]] = defaultdict(set)
    current_scope: tuple[str, str] | None = None

    def bucket_for(payload: dict[str, Any]) -> dict[str, Any]:
        key = group_key(run_cfg, payload)
        if key not in buckets:
            buckets[key] = new_bucket(run_cfg, key)
        return buckets[key]

    for event in events:
        event_type = event.get("event_type")
        scope = prompt_scope(event)
        if current_scope is None:
            current_scope = scope
        elif scope != current_scope:
            pending_prefetch.clear()
            current_scope = scope
        if event_type == "load":
            layer_idx = event.get("layer_idx")
            if layer_idx is not None:
                evicted_by_layer[int(layer_idx)].difference_update(int_list(event, "loaded_experts"))
            continue
        if event_type == "evict":
            layer_idx = event.get("layer_idx")
            if layer_idx is not None:
                evicted_by_layer[int(layer_idx)].update(int_list(event, "evicted_experts"))
            continue
        if event_type == "prefetch_issue":
            target = event.get("target_layer_idx")
            if target is not None:
                pending_prefetch[int(target)].append(event)
            update_prefetch_issue(bucket_for(event), event)
            continue
        if event_type in {"prefetch_wait", "expert_wait"} and stage_ok(event, stage):
            update_wait(bucket_for(event), event, late_wait_ms)
            continue
        if event_type != "placement" or not stage_ok(event, stage):
            continue

        bucket = bucket_for(event)
        update_placement(bucket, event)

        layer_idx = event.get("layer_idx")
        if layer_idx is not None:
            misses = set(int_list(event, "resident_miss_experts"))
            counts = count_map(event)
            damaged = misses & evicted_by_layer[int(layer_idx)]
            bucket["eviction_damage_misses"] += len(damaged)
            bucket["eviction_damage_assignments"] += sum_assignments(damaged, counts)
            if pending_prefetch[int(layer_idx)]:
                update_prefetch_match(bucket, pending_prefetch[int(layer_idx)].popleft(), event)

    return buckets


def analyze_router_events(
    run_cfg: dict[str, Any],
    events: list[dict[str, Any]],
    stage: str,
    buckets: dict[tuple[str, str, str, str, str, str], dict[str, Any]],
) -> None:
    previous: dict[tuple[tuple[str, str, str, str, str, str], int], set[int]] = {}
    current_scope: tuple[str, str] | None = None
    for event in events:
        if event.get("schema") != "hybrimoe.router_assignments.v1" or not stage_ok(event, stage):
            continue
        scope = prompt_scope(event)
        if current_scope is None:
            current_scope = scope
        elif scope != current_scope:
            previous = {}
            current_scope = scope
        layer_idx = event.get("layer_idx")
        if layer_idx is None:
            continue
        key = group_key(run_cfg, event)
        if key not in buckets:
            buckets[key] = new_bucket(run_cfg, key)
        selected = {int(expert) for expert, count in event.get("expert_counts", []) if int(count) > 0}
        prev_key = (key, int(layer_idx))
        if prev_key in previous:
            union = previous[prev_key] | selected
            jaccard = len(previous[prev_key] & selected) / len(union) if union else 1.0
            buckets[key]["route_jaccard_sum"] += jaccard
            buckets[key]["route_jaccard_samples"] += 1
        previous[prev_key] = selected


def finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    row = dict(bucket)
    row["live_cache_hit_ratio"] = ratio(row["resident_hit_assignments"], row["assignments"])
    row["prefetch_redundant_ratio"] = ratio(row["prefetch_resident"], row["prefetch_candidates"])
    row["prefetch_issued_candidate_ratio"] = ratio(row["prefetch_issued"], row["prefetch_candidates"])
    row["prefetch_issued_used_ratio"] = ratio(row["issued_overlap"], row["matched_issued"])
    row["prefetch_issued_unused_ratio"] = 1.0 - row["prefetch_issued_used_ratio"] if row["matched_issued"] else 0.0
    row["prefetch_candidate_assignment_coverage"] = ratio(
        row["candidate_assignment_coverage"], row["matched_assignments"]
    )
    row["prefetch_issued_assignment_coverage"] = ratio(row["issued_assignment_coverage"], row["matched_assignments"])
    row["eviction_damage_assignment_ratio"] = ratio(row["eviction_damage_assignments"], row["resident_miss_assignments"])
    row["cpu_assignment_ratio"] = ratio(row["cpu_assignments"], row["assignments"])
    row["prefetch_wait_late_ratio"] = ratio(row["prefetch_wait_late_events"], row["prefetch_wait_events"])
    row["expert_wait_late_ratio"] = ratio(row["expert_wait_late_events"], row["expert_wait_events"])
    row["avg_same_layer_jaccard"] = ratio(row["route_jaccard_sum"], row["route_jaccard_samples"])
    row["route_drift_ratio"] = 1.0 - row["avg_same_layer_jaccard"] if row["route_jaccard_samples"] else 0.0
    return row


def find_runs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [args.run_dir]
    runs = []
    for cfg_path in args.input_root.rglob("run_config.json"):
        run_dir = cfg_path.parent
        if (run_dir / "expert_cache_trace" / "expert_cache_trace.jsonl").exists():
            runs.append(run_dir)
    return sorted(set(runs))


def analyze_run(run_dir: Path, stage: str, late_wait_ms: float) -> list[dict[str, Any]]:
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    cfg.setdefault("run_tag", run_dir.name)
    expert_events = load_jsonl(run_dir / "expert_cache_trace" / "expert_cache_trace.jsonl")
    buckets = analyze_expert_events(cfg, expert_events, stage, late_wait_ms)
    router_path = run_dir / "router_trace" / "router_trace.jsonl"
    if router_path.exists():
        analyze_router_events(cfg, load_jsonl(router_path), stage, buckets)
    return [finalize_bucket(bucket) for bucket in buckets.values()]


def collapse_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], dict[str, Any]] = {}
    additive = [
        "placement_events",
        "assignments",
        "resident_hit_assignments",
        "resident_miss_assignments",
        "prefetch_issue_events",
        "prefetch_candidates",
        "prefetch_resident",
        "prefetch_issued",
        "matched_prefetch_windows",
        "matched_candidates",
        "matched_issued",
        "candidate_overlap",
        "issued_overlap",
        "matched_assignments",
        "candidate_assignment_coverage",
        "issued_assignment_coverage",
        "eviction_damage_assignments",
        "cpu_assignments",
        "gpu_assignments",
        "prefetch_wait_events",
        "prefetch_wait_total_ms",
        "prefetch_wait_late_events",
        "expert_wait_events",
        "expert_wait_total_ms",
        "expert_wait_late_events",
        "route_jaccard_sum",
        "route_jaccard_samples",
    ]
    for row in rows:
        key = tuple(row.get(item) for item in keys)
        if key not in buckets:
            buckets[key] = {item: row.get(item) for item in keys}
            for name in additive:
                buckets[key][name] = 0
            buckets[key]["cache_size"] = row.get("cache_size")
            buckets[key]["prefetch_size"] = row.get("prefetch_size")
        for name in additive:
            buckets[key][name] += row.get(name, 0)
    return [finalize_bucket(bucket) for bucket in buckets.values()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    leading = [
        "run",
        "stream_id",
        "workload",
        "phase",
        "category",
        "seed",
        "cache_size",
        "prefetch_size",
    ]
    fieldnames = [name for name in leading if name in fieldnames] + [name for name in fieldnames if name not in leading]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def universality_stats(rows: list[dict[str, Any]], high_cache_min: int) -> dict[str, Any]:
    high_cache = [
        row for row in rows if int(row.get("cache_size", -1)) >= high_cache_min and int(row.get("prefetch_size", 0)) > 0
    ]
    phenomenon = [
        row
        for row in high_cache
        if row["live_cache_hit_ratio"] >= 0.80
        and row["prefetch_issued_assignment_coverage"] <= 0.05
        and row["prefetch_redundant_ratio"] >= 0.90
    ]
    dynamic = [row for row in rows if str(row.get("workload", "")).startswith("shifted")]
    drift_high = [row for row in dynamic if row["route_drift_ratio"] >= 0.60]
    return {
        "group_count": len(rows),
        "high_cache_prefetch_group_count": len(high_cache),
        "high_hit_low_utility_group_count": len(phenomenon),
        "high_hit_low_utility_fraction": ratio(len(phenomenon), len(high_cache)),
        "dynamic_group_count": len(dynamic),
        "dynamic_high_route_drift_count": len(drift_high),
        "dynamic_high_route_drift_fraction": ratio(len(drift_high), len(dynamic)),
        "mean_high_cache_hit": mean([row["live_cache_hit_ratio"] for row in high_cache]) if high_cache else 0.0,
        "mean_high_cache_issued_coverage": mean([row["prefetch_issued_assignment_coverage"] for row in high_cache])
        if high_cache
        else 0.0,
        "mean_high_cache_redundant": mean([row["prefetch_redundant_ratio"] for row in high_cache]) if high_cache else 0.0,
    }


def write_report(path: Path, stats: dict[str, Any], run_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Prompt Stream Universality Report",
        "",
        "This report aggregates live HybriMoE traces by run, workload phase, and prompt category.",
        "",
        "## Universality Criteria",
        "",
        "- high cache regime: cache_size >= threshold and prefetch_size > 0",
        "- phenomenon: cache hit >= 80%, issued assignment coverage <= 5%, redundant candidates >= 90%",
        "- dynamic drift: route drift >= 60% for shifted workloads",
        "",
        "## Summary",
        "",
    ]
    for key, value in stats.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Run-Level Metrics",
            "",
            "| run | workload | cache | prefetch | hit | issued coverage | redundant | route drift | eviction damage | prefetch wait late |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(run_rows, key=lambda item: (str(item.get("run")), int(item.get("cache_size", -1)), int(item.get("prefetch_size", -1)))):
        lines.append(
            "| {run} | {workload} | {cache} | {prefetch} | {hit:.2%} | {coverage:.2%} | {redundant:.2%} | {drift:.2%} | {damage:.2%} | {wait:.2%} |".format(
                run=row.get("run", "unknown"),
                workload=row.get("workload", "unknown"),
                cache=row.get("cache_size", "n/a"),
                prefetch=row.get("prefetch_size", "n/a"),
                hit=row["live_cache_hit_ratio"],
                coverage=row["prefetch_issued_assignment_coverage"],
                redundant=row["prefetch_redundant_ratio"],
                drift=row["route_drift_ratio"],
                damage=row["eviction_damage_assignment_ratio"],
                wait=row["prefetch_wait_late_ratio"],
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("results/prompt_stream_runs"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/prompt_stream_aggregate"))
    parser.add_argument("--stage", choices=["all", "prefill", "decode"], default="decode")
    parser.add_argument("--late-wait-ms", type=float, default=0.05)
    parser.add_argument("--high-cache-min", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_dir in find_runs(args):
        rows.extend(analyze_run(run_dir, args.stage, args.late_wait_ms))

    rows.sort(key=lambda item: (str(item.get("run")), str(item.get("workload")), str(item.get("phase")), str(item.get("category"))))
    run_rows = collapse_rows(rows, ["run", "workload", "cache_size", "prefetch_size"])
    category_rows = collapse_rows(rows, ["workload", "phase", "category", "cache_size", "prefetch_size"])
    workload_rows = collapse_rows(rows, ["workload", "cache_size", "prefetch_size"])

    write_csv(args.output_dir / "group_summary.csv", rows)
    write_csv(args.output_dir / "run_summary.csv", run_rows)
    write_csv(args.output_dir / "category_summary.csv", category_rows)
    write_csv(args.output_dir / "workload_summary.csv", workload_rows)

    stats = universality_stats(run_rows, args.high_cache_min)
    (args.output_dir / "universality_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_report(args.output_dir / "universality_report.md", stats, run_rows)
    print(f"[prompt-stream-analysis] groups={len(rows)} runs={len(run_rows)} output={args.output_dir}")


if __name__ == "__main__":
    main()
