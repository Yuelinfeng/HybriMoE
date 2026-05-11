#!/usr/bin/env python3
"""Analyze real MoE router traces with cache/prefetch utility metrics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from prefetch_utility_validation import CacheSimulator, SimulatorConfig, WorkloadConfig, write_csv


def _parse_int_set(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    values = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.add(int(item))
    return values


def load_router_trace(
    path: Path,
    *,
    stage: str,
    layer_filter: set[int] | None,
    max_events: int,
) -> tuple[list[list[int]], dict[str, int | str]]:
    trace: list[list[int]] = []
    num_experts = 0
    skipped = 0
    records = 0
    model_type = "unknown"

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("schema") != "hybrimoe.router_assignments.v1":
                skipped += 1
                continue
            if stage != "all" and payload.get("stage") != stage:
                skipped += 1
                continue
            layer_idx = payload.get("layer_idx")
            if layer_filter is not None and layer_idx not in layer_filter:
                skipped += 1
                continue

            model_type = payload.get("model_type", model_type)
            num_experts = max(num_experts, int(payload.get("num_experts", 0)))
            assignments = _assignments_from_event(payload)
            if assignments:
                trace.append(assignments)
                records += 1
            if max_events and records >= max_events:
                break

    metadata = {
        "trace_path": str(path),
        "model_type": model_type,
        "events_used": len(trace),
        "events_skipped": skipped,
        "num_experts": num_experts,
        "max_assignments_per_event": max((len(item) for item in trace), default=0),
    }
    return trace, metadata


def _assignments_from_event(payload: dict) -> list[int]:
    if "topk_indices" in payload:
        assignments: list[int] = []
        for token_topk in payload["topk_indices"]:
            assignments.extend(int(expert) for expert in token_topk)
        return assignments

    assignments = []
    for expert, count in payload.get("expert_counts", []):
        assignments.extend([int(expert)] * int(count))
    return assignments


def analyze_trace(trace: list[list[int]], metadata: dict[str, int | str], cfg: SimulatorConfig):
    if not trace:
        raise ValueError("Router trace contains no usable events after filtering.")
    workload = WorkloadConfig(
        name=f"router_trace_{metadata['model_type']}",
        shifted=False,
        mixed=False,
        steps=len(trace),
        num_experts=int(metadata["num_experts"]),
        hot_pool_size=0,
        assignments_per_step=int(metadata["max_assignments_per_event"]),
    )
    rows = []
    for policy in ["no_prefetch", "history_prefetch", "utility_gate"]:
        rows.append(CacheSimulator(trace, workload, cfg, policy).run())
    return rows


def write_report(path: Path, rows, metadata: dict[str, int | str], cfg: SimulatorConfig) -> None:
    by_policy = {row.policy: row for row in rows}
    history = by_policy.get("history_prefetch")
    no_prefetch = by_policy.get("no_prefetch")
    gate = by_policy.get("utility_gate")

    lines = [
        "# Router Trace Prefetch Utility Analysis",
        "",
        "This report uses real router assignments captured from a model run, then replays cache/prefetch policies over that expert-access trace.",
        "",
        "## Trace",
        "",
        f"- trace_path: {metadata['trace_path']}",
        f"- model_type: {metadata['model_type']}",
        f"- events_used: {metadata['events_used']}",
        f"- events_skipped: {metadata['events_skipped']}",
        f"- num_experts: {metadata['num_experts']}",
        f"- max_assignments_per_event: {metadata['max_assignments_per_event']}",
        "",
        "## Config",
        "",
        f"- cache_size: {cfg.cache_size}",
        f"- prefetch_width: {cfg.prefetch_width}",
        f"- history_window: {cfg.history_window}",
        f"- transfer_latency: {cfg.transfer_latency}",
        "",
        "## Summary",
        "",
        "| policy | cache_hit_ratio | timely_useful_candidate_ratio | redundant_ratio | issued_late_ratio | issued_unused_ratio | stall_time | bytes_moved | eviction_damage_ratio | phenomenon_flag |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {hit:.4f} | {utility:.4f} | {red:.4f} | {late:.4f} | {unused:.4f} | {stall:.1f} | {bytes_moved} | {damage:.4f} | {flag} |".format(
                policy=row.policy,
                hit=row.cache_hit_ratio,
                utility=row.timely_useful_candidate_ratio,
                red=row.redundant_prefetch_ratio,
                late=row.issued_late_prefetch_ratio,
                unused=row.issued_unused_prefetch_ratio,
                stall=row.stall_time,
                bytes_moved=row.bytes_moved,
                damage=row.eviction_damage_ratio,
                flag=row.phenomenon_flag,
            )
        )

    if history and no_prefetch:
        lines.extend(
            [
                "",
                "## Delta: history_prefetch - no_prefetch",
                "",
                f"- cache_hit_delta: {history.cache_hit_ratio - no_prefetch.cache_hit_ratio:.4f}",
                f"- stall_time_delta: {history.stall_time - no_prefetch.stall_time:.1f}",
                f"- bytes_moved_delta: {history.bytes_moved - no_prefetch.bytes_moved}",
                f"- demand_bytes_delta: {history.demand_bytes_moved - no_prefetch.demand_bytes_moved}",
            ]
        )
    if gate and history:
        lines.extend(
            [
                "",
                "## Gate Check",
                "",
                f"- utility_gate_prefetch_issued: {gate.prefetch_issued}",
                f"- history_prefetch_issued: {history.prefetch_issued}",
                "- If the gate issues zero prefetches, it is only a conservative-abstention baseline, not yet a validated utility scheduler.",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This analysis uses real router assignments, but cache movement is still replayed by the analysis script. A full system claim needs the same counters attached to the live expert cache/load path.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_table(rows) -> None:
    fields = [
        "policy",
        "cache_hit_ratio",
        "timely_useful_candidate_ratio",
        "redundant_prefetch_ratio",
        "issued_late_prefetch_ratio",
        "issued_unused_prefetch_ratio",
        "stall_time",
        "bytes_moved",
        "phenomenon_flag",
    ]
    widths = {field: len(field) for field in fields}
    rendered = []
    for row in rows:
        data = asdict(row)
        item = {}
        for field in fields:
            value = data[field]
            if isinstance(value, float):
                value = f"{value:.4f}"
            else:
                value = str(value)
            item[field] = value
            widths[field] = max(widths[field], len(value))
        rendered.append(item)
    print("  ".join(field.ljust(widths[field]) for field in fields))
    print("  ".join("-" * widths[field] for field in fields))
    for item in rendered:
        print("  ".join(item[field].ljust(widths[field]) for field in fields))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="Router JSONL trace emitted by HYBRIMOE_ROUTER_TRACE.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/router_trace_prefetch_analysis"))
    parser.add_argument("--stage", choices=["all", "prefill", "decode"], default="all")
    parser.add_argument("--layers", help="Comma-separated layer indices to include, e.g. 3,4,5.")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=24)
    parser.add_argument("--prefetch-width", type=int, default=20)
    parser.add_argument("--history-window", type=int, default=36)
    parser.add_argument("--transfer-latency", type=int, default=4)
    parser.add_argument("--gate-threshold", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulatorConfig(
        cache_size=args.cache_size,
        prefetch_width=args.prefetch_width,
        history_window=args.history_window,
        transfer_latency=args.transfer_latency,
        gate_threshold=args.gate_threshold,
    )
    trace, metadata = load_router_trace(
        args.trace,
        stage=args.stage,
        layer_filter=_parse_int_set(args.layers),
        max_events=args.max_events,
    )
    rows = analyze_trace(trace, metadata, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "trace_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(args.output_dir / "README.md", rows, metadata, cfg)
    print_table(rows)
    print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
