#!/usr/bin/env python3
"""Replay draft-conditioned expert prefetch policies from router traces.

This is a stage-3 prototype.  It does not run speculative decoding itself.
Instead, it treats future router assignments as an oracle or noisy-draft
lookahead, then replays several prefetch admission policies over the same
expert demand stream:

- no_prefetch
- always
- cutoff
- budgeted
- utility_gate

The goal is to test whether rejecting low-utility speculative candidates can
reduce redundant / unused / late / eviction-damaging prefetches before wiring a
real draft model into the inference path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DemandEvent:
    index: int
    event_id: int
    time_ns: int
    layer_idx: int
    stage: str
    expert_counts: tuple[tuple[int, int], ...]
    assignment_count: int
    unique_expert_count: int
    num_experts: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Candidate:
    target_index: int
    target_layer: int
    expert: int
    confidence: float
    assignment_count: int
    distance: int
    source: str


@dataclass
class PrefetchRecord:
    record_id: int
    layer_idx: int
    expert: int
    issue_time: int
    ready_time: int
    target_time: int
    confidence: float
    consumed: bool = False
    consumed_late: bool = False
    evicted_before_use: bool = False


@dataclass
class CacheEntry:
    loaded_by: str
    prefetch_id: int | None


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in parse_csv_list(raw)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    events.sort(key=lambda item: (int(item.get("time_ns", 0)), int(item.get("event_id", 0))))
    return events


def metadata(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("metadata")
    return value if isinstance(value, dict) else {}


def prompt_scope(payload: dict[str, Any]) -> tuple[str, str]:
    meta = metadata(payload)
    return str(meta.get("stream_id", "unknown")), str(meta.get("prompt_id", "unknown"))


def event_sort_key(payload: dict[str, Any]) -> tuple[int, int]:
    return int(payload.get("time_ns", 0)), int(payload.get("event_id", 0))


def stage_ok(payload: dict[str, Any], stage: str) -> bool:
    return stage == "all" or payload.get("stage") == stage


def filter_router_decode_window(events: list[dict[str, Any]], stage: str, decode_window: int) -> list[dict[str, Any]]:
    if decode_window <= 0 or stage != "decode":
        return events

    by_prompt: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("schema") == "hybrimoe.router_assignments.v1" and event.get("stage") == "decode":
            by_prompt[prompt_scope(event)].append(event)

    cutoff_by_prompt: dict[tuple[str, str], tuple[int, int]] = {}
    for scope, records in by_prompt.items():
        layers = {event.get("layer_idx") for event in records if event.get("layer_idx") is not None}
        layers_per_token = max(1, len(layers))
        target_events = decode_window * layers_per_token
        if len(records) >= target_events:
            cutoff_by_prompt[scope] = event_sort_key(records[target_events - 1])

    if not cutoff_by_prompt:
        return events
    return [
        event
        for event in events
        if prompt_scope(event) not in cutoff_by_prompt or event_sort_key(event) <= cutoff_by_prompt[prompt_scope(event)]
    ]


def parse_demand_events(path: Path, stage: str, decode_window: int) -> list[DemandEvent]:
    raw_events = filter_router_decode_window(load_jsonl(path), stage, decode_window)
    records = []
    for payload in raw_events:
        if payload.get("schema") != "hybrimoe.router_assignments.v1" or not stage_ok(payload, stage):
            continue
        layer_idx = payload.get("layer_idx")
        if layer_idx is None:
            continue
        expert_counts = tuple((int(expert), int(count)) for expert, count in payload.get("expert_counts", []))
        if not expert_counts:
            continue
        records.append(
            DemandEvent(
                index=len(records),
                event_id=int(payload.get("event_id", len(records))),
                time_ns=int(payload.get("time_ns", 0)),
                layer_idx=int(layer_idx),
                stage=str(payload.get("stage", "unknown")),
                expert_counts=expert_counts,
                assignment_count=int(payload.get("assignment_count", sum(count for _, count in expert_counts))),
                unique_expert_count=int(payload.get("unique_expert_count", len(expert_counts))),
                num_experts=int(payload.get("num_experts", 0)),
                metadata=metadata(payload),
            )
        )
    return records


class SpeculativeReplay:
    def __init__(self, events: list[DemandEvent], args: argparse.Namespace, policy: str, run_cfg: dict[str, Any]):
        self.events = events
        self.args = args
        self.policy = policy
        self.policy_kind = policy.split("@", 1)[0]
        self.run_cfg = run_cfg
        self.cache_size = int(args.cache_size_override or run_cfg.get("cache_size") or args.cache_size)
        self.prefetch_size = int(args.prefetch_size_override or run_cfg.get("prefetch_size") or args.prefetch_size)
        if self.policy_kind == "no_prefetch":
            self.prefetch_size = 0
        self.rng = random.Random(args.seed)
        self.num_experts = max([event.num_experts for event in events if event.num_experts] or [args.num_experts])
        self.cache: dict[int, OrderedDict[int, CacheEntry]] = defaultdict(OrderedDict)
        self.pending: dict[tuple[int, int], PrefetchRecord] = {}
        self.records: list[PrefetchRecord] = []
        self.evicted_by_layer: dict[int, set[int]] = defaultdict(set)
        self.rows: dict[str, Any] = defaultdict(int)
        self.rows.update(
            {
                "policy": policy,
                "run": str(run_cfg.get("run_tag", "unknown")),
                "stage": args.stage,
                "cache_size": self.cache_size,
                "prefetch_size": self.prefetch_size,
                "lookahead_events": args.lookahead_events,
                "draft_mode": args.draft_mode,
                "draft_fidelity": args.draft_fidelity,
                "accept_prob": args.accept_prob,
                "transfer_latency": args.transfer_latency,
                "gate_threshold": args.gate_threshold if policy == "utility_gate" else "",
                "cutoff_layer": args.cutoff_layer if policy == "cutoff" else "",
            }
        )

    def run(self) -> dict[str, Any]:
        for time, event in enumerate(self.events):
            self._materialize_ready(time)
            self._consume_demand(time, event)
            self._issue_prefetches(time)
        self._materialize_ready(len(self.events) + self.args.transfer_latency + 1)
        self._finalize_unused()
        return self._finalize()

    def _materialize_ready(self, time: int) -> None:
        ready_keys = [key for key, record in self.pending.items() if record.ready_time <= time]
        for key in ready_keys:
            record = self.pending.pop(key)
            self._insert_cache(record.layer_idx, record.expert, CacheEntry("prefetch", record.record_id))

    def _consume_demand(self, time: int, event: DemandEvent) -> None:
        layer_cache = self.cache[event.layer_idx]
        self.rows["placement_events"] += 1
        self.rows["assignments"] += event.assignment_count
        self.rows["unique_demands"] += event.unique_expert_count

        selected = {expert for expert, _ in event.expert_counts}
        damaged = selected & self.evicted_by_layer[event.layer_idx]
        self.rows["eviction_damage_experts"] += len(damaged)

        for expert, count in event.expert_counts:
            key = (event.layer_idx, expert)
            if expert in layer_cache:
                self.rows["cache_hit_assignments"] += count
                entry = layer_cache.pop(expert)
                layer_cache[expert] = entry
                if entry.prefetch_id is not None:
                    record = self.records[entry.prefetch_id]
                    if not record.consumed:
                        record.consumed = True
                        self.rows["prefetch_timely_useful_experts"] += 1
                    self.rows["prefetch_timely_useful_assignments"] += count
                continue

            self.rows["cache_miss_assignments"] += count
            if expert in damaged:
                self.rows["eviction_damage_assignments"] += count
            if key in self.pending:
                record = self.pending[key]
                self.rows["late_prefetch_experts"] += 1
                self.rows["late_prefetch_assignments"] += count
                record.consumed_late = True

            self.rows["demand_loads"] += 1
            self._insert_cache(event.layer_idx, expert, CacheEntry("demand", None))

    def _issue_prefetches(self, time: int) -> None:
        if self.policy_kind == "no_prefetch" or self.prefetch_size <= 0:
            return
        candidates = self._draft_candidates(time)
        if self.policy_kind == "cutoff" and self.args.cutoff_layer >= 0:
            candidates = [candidate for candidate in candidates if candidate.target_layer <= self.args.cutoff_layer]
        if self.policy_kind in {"budgeted", "utility_gate", "cutoff", "always"}:
            candidates = sorted(
                candidates,
                key=lambda candidate: (candidate.confidence, candidate.assignment_count, -candidate.distance),
                reverse=True,
            )

        issued_this_step = 0
        budget_limit = math.inf if self.policy_kind == "always" else self.prefetch_size
        for candidate in candidates:
            self.rows["prefetch_candidates"] += 1
            key = (candidate.target_layer, candidate.expert)
            if self._is_resident(candidate.target_layer, candidate.expert):
                self.rows["prefetch_redundant_resident"] += 1
                continue
            if key in self.pending:
                self.rows["prefetch_redundant_pending"] += 1
                continue
            if issued_this_step >= budget_limit:
                self.rows["prefetch_budget_rejected"] += 1
                continue
            if self.policy_kind == "utility_gate" and not self._admit(candidate, time):
                self.rows["prefetch_gate_rejected"] += 1
                continue

            record = PrefetchRecord(
                record_id=len(self.records),
                layer_idx=candidate.target_layer,
                expert=candidate.expert,
                issue_time=time,
                ready_time=time + self.args.transfer_latency,
                target_time=candidate.target_index,
                confidence=candidate.confidence,
            )
            self.records.append(record)
            self.pending[key] = record
            self.rows["prefetch_issued"] += 1
            issued_this_step += 1

    def _draft_candidates(self, time: int) -> list[Candidate]:
        if time + 1 >= len(self.events):
            return []
        candidates: dict[tuple[int, int, int], Candidate] = {}
        end = min(len(self.events), time + 1 + self.args.lookahead_events)
        for target_index in range(time + 1, end):
            target = self.events[target_index]
            distance = target_index - time
            decay = self.args.accept_prob ** max(0, distance - 1)
            for expert, count in target.expert_counts:
                assignment_prob = count / max(1, target.assignment_count)
                confidence = assignment_prob * decay * self.args.draft_fidelity
                if self.args.draft_mode == "noisy_oracle" and self.rng.random() > self.args.draft_fidelity:
                    continue
                key = (target_index, target.layer_idx, expert)
                candidates[key] = Candidate(
                    target_index=target_index,
                    target_layer=target.layer_idx,
                    expert=expert,
                    confidence=confidence,
                    assignment_count=count,
                    distance=distance,
                    source="oracle",
                )

            if self.args.draft_mode == "noisy_oracle" and self.args.draft_fidelity < 1.0:
                false_count = max(1, int((1.0 - self.args.draft_fidelity) * len(target.expert_counts)))
                true_experts = {expert for expert, _ in target.expert_counts}
                for _ in range(false_count):
                    expert = self.rng.randrange(max(1, self.num_experts))
                    if expert in true_experts:
                        continue
                    key = (target_index, target.layer_idx, expert)
                    candidates[key] = Candidate(
                        target_index=target_index,
                        target_layer=target.layer_idx,
                        expert=expert,
                        confidence=(1.0 - self.args.draft_fidelity) * decay / max(1, target.assignment_count),
                        assignment_count=1,
                        distance=distance,
                        source="false_positive",
                    )
        return list(candidates.values())

    def _admit(self, candidate: Candidate, time: int) -> bool:
        cache_pressure = len(self.cache[candidate.target_layer]) / max(1, self.cache_size)
        pending_pressure = len(self.pending) / max(1, self.prefetch_size)
        ready_time = time + self.args.transfer_latency
        lateness = max(0, ready_time - candidate.target_index) / max(1, self.args.transfer_latency)
        expected_stall_saved = candidate.confidence * candidate.assignment_count * self.args.miss_penalty
        transfer_cost = self.args.transfer_cost
        eviction_cost = self.args.eviction_cost * cache_pressure
        contention_cost = self.args.contention_cost * pending_pressure
        lateness_penalty = self.args.lateness_cost * lateness
        score = expected_stall_saved - transfer_cost - eviction_cost - contention_cost - lateness_penalty
        self.rows["gate_score_sum"] += score
        self.rows["gate_score_samples"] += 1
        return score > self.args.gate_threshold

    def _is_resident(self, layer_idx: int, expert: int) -> bool:
        return expert in self.cache[layer_idx]

    def _insert_cache(self, layer_idx: int, expert: int, entry: CacheEntry) -> None:
        layer_cache = self.cache[layer_idx]
        if expert in layer_cache:
            old_entry = layer_cache.pop(expert)
            if entry.loaded_by == "prefetch" and old_entry.loaded_by != "prefetch":
                self.rows["prefetch_ready_redundant"] += 1
        while len(layer_cache) >= self.cache_size:
            victim, victim_entry = layer_cache.popitem(last=False)
            self.evicted_by_layer[layer_idx].add(victim)
            self.rows["evictions"] += 1
            if victim_entry.loaded_by == "prefetch":
                self.rows["prefetch_evictions"] += 1
                if victim_entry.prefetch_id is not None:
                    record = self.records[victim_entry.prefetch_id]
                    if not record.consumed:
                        record.evicted_before_use = True
                        self.rows["prefetch_evicted_unused"] += 1
            elif entry.loaded_by == "prefetch":
                self.rows["demand_entries_evicted_by_prefetch"] += 1
        layer_cache[expert] = entry
        if entry.loaded_by == "prefetch":
            self.rows["prefetch_ready"] += 1

    def _finalize_unused(self) -> None:
        for record in self.records:
            if not record.consumed and not record.consumed_late:
                self.rows["prefetch_unused"] += 1

    def _finalize(self) -> dict[str, Any]:
        row = dict(self.rows)
        assignments = row.get("assignments", 0)
        issued = row.get("prefetch_issued", 0)
        candidates = row.get("prefetch_candidates", 0)
        redundant = row.get("prefetch_redundant_resident", 0) + row.get("prefetch_redundant_pending", 0)
        row["cache_hit_assignment_ratio"] = ratio(row.get("cache_hit_assignments", 0), assignments)
        row["cache_miss_assignment_ratio"] = ratio(row.get("cache_miss_assignments", 0), assignments)
        row["prefetch_issued_candidate_ratio"] = ratio(issued, candidates)
        row["prefetch_redundant_candidate_ratio"] = ratio(redundant, candidates)
        row["prefetch_rejected_candidate_ratio"] = ratio(
            row.get("prefetch_gate_rejected", 0) + row.get("prefetch_budget_rejected", 0), candidates
        )
        row["prefetch_timely_useful_assignment_ratio"] = ratio(
            row.get("prefetch_timely_useful_assignments", 0), assignments
        )
        row["prefetch_timely_useful_issued_ratio"] = ratio(row.get("prefetch_timely_useful_experts", 0), issued)
        row["prefetch_late_assignment_ratio"] = ratio(row.get("late_prefetch_assignments", 0), assignments)
        row["prefetch_unused_issued_ratio"] = ratio(row.get("prefetch_unused", 0), issued)
        row["prefetch_evicted_unused_ratio"] = ratio(row.get("prefetch_evicted_unused", 0), issued)
        row["eviction_damage_assignment_ratio"] = ratio(row.get("eviction_damage_assignments", 0), row.get("cache_miss_assignments", 0))
        row["demand_load_ratio"] = ratio(row.get("demand_loads", 0), row.get("unique_demands", 0))
        row["prefetch_load_ratio"] = ratio(row.get("prefetch_issued", 0), row.get("unique_demands", 0))
        row["estimated_demand_stall_steps"] = row.get("cache_miss_assignments", 0) * self.args.transfer_latency
        row["estimated_prefetch_bytes_mb"] = row.get("prefetch_issued", 0) * self.args.expert_bytes_mb
        row["estimated_demand_bytes_mb"] = row.get("demand_loads", 0) * self.args.expert_bytes_mb
        row["estimated_total_bytes_mb"] = row["estimated_prefetch_bytes_mb"] + row["estimated_demand_bytes_mb"]
        row["avg_gate_score"] = ratio(row.get("gate_score_sum", 0.0), row.get("gate_score_samples", 0))
        return row


def find_runs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [args.run_dir]
    runs = []
    for cfg_path in args.input_root.rglob("run_config.json"):
        run_dir = cfg_path.parent
        if (run_dir / "router_trace" / "router_trace.jsonl").exists():
            runs.append(run_dir)
    return sorted(set(runs))


def load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
    else:
        cfg = {}
    cfg.setdefault("run_tag", run_dir.name)
    return cfg


def analyze_run(run_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = load_run_config(run_dir)
    router_path = run_dir / "router_trace" / "router_trace.jsonl"
    events = parse_demand_events(router_path, args.stage, args.decode_window)
    rows = []
    for policy in parse_csv_list(args.policies):
        if policy == "utility_gate" and args.gate_thresholds:
            for threshold in parse_float_list(args.gate_thresholds):
                sweep_args = argparse.Namespace(**vars(args))
                sweep_args.gate_threshold = threshold
                replay = SpeculativeReplay(events, sweep_args, f"utility_gate@{threshold:g}", cfg)
                row = replay.run()
                row["router_events"] = len(events)
                row["run_dir"] = str(run_dir)
                rows.append(row)
        else:
            replay = SpeculativeReplay(events, args, policy, cfg)
            row = replay.run()
            row["router_events"] = len(events)
            row["run_dir"] = str(run_dir)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    leading = [
        "run",
        "policy",
        "cache_size",
        "prefetch_size",
        "stage",
        "draft_mode",
        "draft_fidelity",
        "lookahead_events",
    ]
    fieldnames = [name for name in leading if name in fieldnames] + [name for name in fieldnames if name not in leading]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collapse_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    additive_names = [
        "placement_events",
        "assignments",
        "unique_demands",
        "cache_hit_assignments",
        "cache_miss_assignments",
        "prefetch_candidates",
        "prefetch_redundant_resident",
        "prefetch_redundant_pending",
        "prefetch_gate_rejected",
        "prefetch_budget_rejected",
        "prefetch_issued",
        "prefetch_ready",
        "prefetch_timely_useful_experts",
        "prefetch_timely_useful_assignments",
        "late_prefetch_experts",
        "late_prefetch_assignments",
        "prefetch_unused",
        "prefetch_evicted_unused",
        "eviction_damage_assignments",
        "demand_loads",
        "evictions",
        "prefetch_evictions",
        "demand_entries_evicted_by_prefetch",
        "gate_score_sum",
        "gate_score_samples",
    ]
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        policy = str(row["policy"])
        if policy not in buckets:
            buckets[policy] = {name: 0 for name in additive_names}
            for name in [
                "policy",
                "stage",
                "draft_mode",
                "draft_fidelity",
                "accept_prob",
                "lookahead_events",
                "transfer_latency",
                "expert_bytes_mb",
            ]:
                if name in row:
                    buckets[policy][name] = row[name]
            buckets[policy]["run_count"] = 0
        buckets[policy]["run_count"] += 1
        for name in additive_names:
            buckets[policy][name] += row.get(name, 0)
    return [finalize_collapsed(bucket) for bucket in buckets.values()]


def finalize_collapsed(row: dict[str, Any]) -> dict[str, Any]:
    assignments = row.get("assignments", 0)
    issued = row.get("prefetch_issued", 0)
    candidates = row.get("prefetch_candidates", 0)
    redundant = row.get("prefetch_redundant_resident", 0) + row.get("prefetch_redundant_pending", 0)
    row["cache_hit_assignment_ratio"] = ratio(row.get("cache_hit_assignments", 0), assignments)
    row["prefetch_issued_candidate_ratio"] = ratio(issued, candidates)
    row["prefetch_redundant_candidate_ratio"] = ratio(redundant, candidates)
    row["prefetch_rejected_candidate_ratio"] = ratio(
        row.get("prefetch_gate_rejected", 0) + row.get("prefetch_budget_rejected", 0), candidates
    )
    row["prefetch_timely_useful_assignment_ratio"] = ratio(row.get("prefetch_timely_useful_assignments", 0), assignments)
    row["prefetch_timely_useful_issued_ratio"] = ratio(row.get("prefetch_timely_useful_experts", 0), issued)
    row["prefetch_late_assignment_ratio"] = ratio(row.get("late_prefetch_assignments", 0), assignments)
    row["prefetch_unused_issued_ratio"] = ratio(row.get("prefetch_unused", 0), issued)
    row["prefetch_evicted_unused_ratio"] = ratio(row.get("prefetch_evicted_unused", 0), issued)
    row["eviction_damage_assignment_ratio"] = ratio(row.get("eviction_damage_assignments", 0), row.get("cache_miss_assignments", 0))
    row["demand_load_ratio"] = ratio(row.get("demand_loads", 0), row.get("unique_demands", 0))
    row["prefetch_load_ratio"] = ratio(row.get("prefetch_issued", 0), row.get("unique_demands", 0))
    transfer_latency = float(row.get("transfer_latency", 0) or 0)
    expert_bytes_mb = float(row.get("expert_bytes_mb", 1.0) or 1.0)
    row["estimated_demand_stall_steps"] = row.get("cache_miss_assignments", 0) * transfer_latency
    row["estimated_prefetch_bytes_mb"] = row.get("prefetch_issued", 0) * expert_bytes_mb
    row["estimated_demand_bytes_mb"] = row.get("demand_loads", 0) * expert_bytes_mb
    row["estimated_total_bytes_mb"] = row["estimated_prefetch_bytes_mb"] + row["estimated_demand_bytes_mb"]
    row["avg_gate_score"] = ratio(row.get("gate_score_sum", 0.0), row.get("gate_score_samples", 0))
    return row


def write_report(path: Path, policy_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Speculative Prefetch Replay Report",
        "",
        "This is an offline replay. Future router assignments are used as oracle/noisy-draft lookahead.",
        "It validates admission-policy direction before connecting a real draft model.",
        "",
        "| policy | hit | timely utility | redundant | unused issued | late | eviction damage | prefetch loads | total MB | stall steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(policy_rows, key=lambda item: str(item["policy"])):
        lines.append(
            "| {policy} | {hit:.2%} | {utility:.2%} | {redundant:.2%} | {unused:.2%} | {late:.2%} | {damage:.2%} | {loads} | {mb:.1f} | {stall:.1f} |".format(
                policy=row["policy"],
                hit=row["cache_hit_assignment_ratio"],
                utility=row["prefetch_timely_useful_assignment_ratio"],
                redundant=row["prefetch_redundant_candidate_ratio"],
                unused=row["prefetch_unused_issued_ratio"],
                late=row["prefetch_late_assignment_ratio"],
                damage=row["eviction_damage_assignment_ratio"],
                loads=int(row.get("prefetch_issued", 0)),
                mb=float(row.get("estimated_total_bytes_mb", 0.0)),
                stall=float(row.get("estimated_demand_stall_steps", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            "",
            "- `always` is a speculative-lookahead upper-pressure baseline, not a proposed policy.",
            "- `utility_gate` is the minimal admission prototype.",
            "- This replay does not include draft-model compute overhead or token acceptance cost.",
            "- A positive result here only justifies wiring the gate into live prefetch scheduling.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("results/prompt_stream_runs"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/speculative_replay"))
    parser.add_argument("--stage", choices=["all", "prefill", "decode"], default="decode")
    parser.add_argument("--decode-window", type=int, default=0)
    parser.add_argument("--policies", default="no_prefetch,always,cutoff,budgeted,utility_gate")
    parser.add_argument("--draft-mode", choices=["oracle", "noisy_oracle"], default="oracle")
    parser.add_argument("--draft-fidelity", type=float, default=1.0)
    parser.add_argument("--accept-prob", type=float, default=0.8)
    parser.add_argument("--lookahead-events", type=int, default=32)
    parser.add_argument("--transfer-latency", type=int, default=4)
    parser.add_argument("--cache-size", type=int, default=56)
    parser.add_argument("--prefetch-size", type=int, default=8)
    parser.add_argument("--cache-size-override", type=int, default=0)
    parser.add_argument("--prefetch-size-override", type=int, default=0)
    parser.add_argument("--cutoff-layer", type=int, default=12)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--expert-bytes-mb", type=float, default=1.0)
    parser.add_argument("--miss-penalty", type=float, default=4.0)
    parser.add_argument("--transfer-cost", type=float, default=0.65)
    parser.add_argument("--eviction-cost", type=float, default=0.9)
    parser.add_argument("--contention-cost", type=float, default=0.55)
    parser.add_argument("--lateness-cost", type=float, default=1.25)
    parser.add_argument("--gate-threshold", type=float, default=0.25)
    parser.add_argument(
        "--gate-thresholds",
        default="",
        help="Optional comma-separated threshold sweep for utility_gate, for example '-0.5,0,0.25,0.5'.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_dir in find_runs(args):
        rows.extend(analyze_run(run_dir, args))

    rows.sort(key=lambda item: (str(item.get("run")), str(item.get("policy"))))
    policy_rows = collapse_policy_rows(rows)
    write_csv(args.output_dir / "spec_replay_summary.csv", rows)
    write_csv(args.output_dir / "spec_replay_policy_summary.csv", policy_rows)
    (args.output_dir / "spec_replay_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_report(args.output_dir / "spec_replay_report.md", policy_rows)
    print(f"[spec-replay] runs={len({row['run'] for row in rows})} rows={len(rows)} output={args.output_dir}")


if __name__ == "__main__":
    main()
