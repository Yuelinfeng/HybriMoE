#!/usr/bin/env python3
"""Validate cache-residency vs prefetch-utility decoupling for MoE offloading.

This is a lightweight research harness. It does not require model weights or
CUDA. The goal is to test whether a workload can show high cache hit ratio while
prefetch actions remain redundant, late, unused, or polluting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class WorkloadConfig:
    name: str
    shifted: bool
    mixed: bool
    steps: int = 240
    num_experts: int = 64
    hot_pool_size: int = 14
    assignments_per_step: int = 8
    phase_len: int = 60
    hot_prob: float = 0.86
    seed_offset: int = 0


@dataclass(frozen=True)
class SimulatorConfig:
    cache_size: int = 24
    prefetch_width: int = 20
    history_window: int = 36
    transfer_latency: int = 4
    timely_ttl: int = 10
    expert_bytes: int = 1
    damage_horizon: int = 8
    gate_threshold: float = 0.25


@dataclass
class PrefetchRecord:
    prefetch_id: int
    expert: int
    issue_time: int
    ready_time: int
    used_time: int | None = None
    late_use_time: int | None = None
    evicted_time: int | None = None
    admitted: bool = True


@dataclass
class CacheEntry:
    expert: int
    loaded_by: str
    loaded_time: int
    ready_time: int
    prefetch_id: int | None = None


@dataclass
class RunMetrics:
    workload: str
    policy: str
    accesses: int
    cache_hits: int
    cache_hit_ratio: float
    cache_hit_from_prefetch: int
    prefetch_candidates: int
    prefetch_rejected: int
    prefetch_redundant: int
    prefetch_already_pending: int
    prefetch_issued: int
    prefetch_used_timely: int
    prefetch_used_late: int
    prefetch_unused: int
    timely_useful_candidate_ratio: float
    timely_useful_issued_ratio: float
    redundant_prefetch_ratio: float
    late_prefetch_ratio: float
    unused_prefetch_ratio: float
    issued_late_prefetch_ratio: float
    issued_unused_prefetch_ratio: float
    ready_before_consume_ratio: float
    stall_time: float
    bytes_moved: int
    prefetch_bytes_moved: int
    demand_bytes_moved: int
    evictions: int
    prefetch_evictions: int
    damaging_prefetch_evictions: int
    eviction_damage_ratio: float
    issue_f1_proxy: float
    phenomenon_flag: bool


WORKLOADS = [
    WorkloadConfig("stable_homogeneous", shifted=False, mixed=False, seed_offset=11),
    WorkloadConfig("shifted_homogeneous", shifted=True, mixed=False, seed_offset=23),
    WorkloadConfig("stable_mixed", shifted=False, mixed=True, seed_offset=37),
    WorkloadConfig("shifted_mixed", shifted=True, mixed=True, seed_offset=41),
]


def make_hot_pool(num_experts: int, hot_pool_size: int, phase: int, stream: int) -> list[int]:
    start = (phase * 17 + stream * 11) % num_experts
    return [(start + i * 3) % num_experts for i in range(hot_pool_size)]


def generate_workload(cfg: WorkloadConfig, seed: int) -> list[list[int]]:
    rng = random.Random(seed + cfg.seed_offset)
    trace: list[list[int]] = []
    base_pools = [
        make_hot_pool(cfg.num_experts, cfg.hot_pool_size, 0, stream)
        for stream in range(3 if cfg.mixed else 1)
    ]

    for step in range(cfg.steps):
        phase = step // cfg.phase_len if cfg.shifted else 0
        if cfg.mixed:
            stream = (step // 5 + phase) % 3
            hot_pool = make_hot_pool(cfg.num_experts, cfg.hot_pool_size, phase, stream)
        else:
            hot_pool = make_hot_pool(cfg.num_experts, cfg.hot_pool_size, phase, 0) if cfg.shifted else base_pools[0]

        assignments: list[int] = []
        for _ in range(cfg.assignments_per_step):
            if rng.random() < cfg.hot_prob:
                expert = rng.choice(hot_pool)
            else:
                expert = rng.randrange(cfg.num_experts)
            assignments.append(expert)
        trace.append(assignments)
    return trace


def build_next_use(trace: list[list[int]], num_experts: int) -> list[dict[int, int]]:
    next_seen = {expert: math.inf for expert in range(num_experts)}
    next_use: list[dict[int, int]] = [dict() for _ in trace]
    for time in range(len(trace) - 1, -1, -1):
        next_use[time] = dict(next_seen)
        for expert in set(trace[time]):
            next_seen[expert] = time
    return next_use


class CacheSimulator:
    def __init__(self, trace: list[list[int]], workload: WorkloadConfig, cfg: SimulatorConfig, policy: str):
        self.trace = trace
        self.workload = workload
        self.cfg = cfg
        self.policy = policy
        self.next_use = build_next_use(trace, workload.num_experts)
        self.cache: OrderedDict[int, CacheEntry] = OrderedDict()
        self.pending: dict[int, PrefetchRecord] = {}
        self.records: list[PrefetchRecord] = []
        self.history: deque[int] = deque(maxlen=cfg.history_window * workload.assignments_per_step)
        self.prefetch_id = 0

        self.accesses = 0
        self.cache_hits = 0
        self.cache_hit_from_prefetch = 0
        self.prefetch_candidates = 0
        self.prefetch_rejected = 0
        self.prefetch_redundant = 0
        self.prefetch_already_pending = 0
        self.prefetch_used_timely = 0
        self.prefetch_used_late = 0
        self.stall_time = 0.0
        self.prefetch_bytes_moved = 0
        self.demand_bytes_moved = 0
        self.evictions = 0
        self.prefetch_evictions = 0
        self.damaging_prefetch_evictions = 0

    def run(self) -> RunMetrics:
        for time, assignments in enumerate(self.trace):
            self._materialize_ready_prefetches(time)
            self._issue_prefetches(time)
            for expert in assignments:
                self._consume(time, expert)
                self.history.append(expert)

        self._materialize_ready_prefetches(len(self.trace) + self.cfg.transfer_latency + 1)
        issued_records = [record for record in self.records if record.admitted]
        unused_records = [
            record
            for record in issued_records
            if record.used_time is None and record.late_use_time is None
        ]
        for record in unused_records:
            if record.evicted_time is None:
                record.evicted_time = len(self.trace)

        prefetch_unused = len(unused_records)
        issued = len(issued_records)
        useful_candidates = self.prefetch_used_timely
        candidate_denom = self.prefetch_candidates or 1
        issued_denom = issued or 1
        used_total = self.prefetch_used_timely + self.prefetch_used_late
        ready_denom = used_total or 1
        bytes_moved = self.prefetch_bytes_moved + self.demand_bytes_moved
        cache_hit_ratio = self.cache_hits / self.accesses if self.accesses else 0.0
        timely_candidate_ratio = useful_candidates / candidate_denom
        timely_issued_ratio = self.prefetch_used_timely / issued_denom
        redundant_ratio = self.prefetch_redundant / candidate_denom
        late_candidate_ratio = self.prefetch_used_late / candidate_denom
        unused_candidate_ratio = prefetch_unused / candidate_denom
        issued_late_ratio = self.prefetch_used_late / issued_denom
        issued_unused_ratio = prefetch_unused / issued_denom
        ready_before_consume_ratio = self.prefetch_used_timely / ready_denom
        eviction_damage_ratio = (
            self.damaging_prefetch_evictions / self.prefetch_evictions if self.prefetch_evictions else 0.0
        )
        issue_f1_proxy = self._issue_f1_proxy()
        phenomenon_flag = (
            cache_hit_ratio >= 0.70
            and timely_candidate_ratio <= 0.30
            and (redundant_ratio >= 0.45 or issued_late_ratio >= 0.20 or issued_unused_ratio >= 0.35)
        )

        return RunMetrics(
            workload=self.workload.name,
            policy=self.policy,
            accesses=self.accesses,
            cache_hits=self.cache_hits,
            cache_hit_ratio=cache_hit_ratio,
            cache_hit_from_prefetch=self.cache_hit_from_prefetch,
            prefetch_candidates=self.prefetch_candidates,
            prefetch_rejected=self.prefetch_rejected,
            prefetch_redundant=self.prefetch_redundant,
            prefetch_already_pending=self.prefetch_already_pending,
            prefetch_issued=issued,
            prefetch_used_timely=self.prefetch_used_timely,
            prefetch_used_late=self.prefetch_used_late,
            prefetch_unused=prefetch_unused,
            timely_useful_candidate_ratio=timely_candidate_ratio,
            timely_useful_issued_ratio=timely_issued_ratio,
            redundant_prefetch_ratio=redundant_ratio,
            late_prefetch_ratio=late_candidate_ratio,
            unused_prefetch_ratio=unused_candidate_ratio,
            issued_late_prefetch_ratio=issued_late_ratio,
            issued_unused_prefetch_ratio=issued_unused_ratio,
            ready_before_consume_ratio=ready_before_consume_ratio,
            stall_time=self.stall_time,
            bytes_moved=bytes_moved,
            prefetch_bytes_moved=self.prefetch_bytes_moved,
            demand_bytes_moved=self.demand_bytes_moved,
            evictions=self.evictions,
            prefetch_evictions=self.prefetch_evictions,
            damaging_prefetch_evictions=self.damaging_prefetch_evictions,
            eviction_damage_ratio=eviction_damage_ratio,
            issue_f1_proxy=issue_f1_proxy,
            phenomenon_flag=phenomenon_flag,
        )

    def _issue_f1_proxy(self) -> float:
        predicted = self.prefetch_candidates - self.prefetch_rejected
        if predicted <= 0:
            return 0.0
        useful = self.prefetch_used_timely
        precision = useful / predicted
        future_unique = sum(len(set(step)) for step in self.trace)
        recall = useful / future_unique if future_unique else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _consume(self, time: int, expert: int) -> None:
        self.accesses += 1
        if expert in self.cache:
            entry = self.cache.pop(expert)
            self.cache[expert] = entry
            self.cache_hits += 1
            if entry.loaded_by == "prefetch" and entry.prefetch_id is not None:
                record = self.records[entry.prefetch_id]
                if record.used_time is None and record.late_use_time is None:
                    record.used_time = time
                    self.prefetch_used_timely += 1
                    self.cache_hit_from_prefetch += 1
                    entry.loaded_by = "demand-after-prefetch-hit"
            return

        if expert in self.pending:
            record = self.pending.pop(expert)
            record.late_use_time = time
            self.prefetch_used_late += 1
            self.stall_time += max(0, record.ready_time - time)
            self._insert_cache(time, expert, "late-prefetch", record.prefetch_id)
            return

        self.stall_time += self.cfg.transfer_latency
        self.demand_bytes_moved += self.cfg.expert_bytes
        self._insert_cache(time, expert, "demand", None)

    def _materialize_ready_prefetches(self, time: int) -> None:
        ready = [expert for expert, record in self.pending.items() if record.ready_time <= time]
        for expert in ready:
            record = self.pending.pop(expert)
            self._insert_cache(record.ready_time, expert, "prefetch", record.prefetch_id)

    def _issue_prefetches(self, time: int) -> None:
        if self.policy == "no_prefetch":
            return
        for expert, score in self._prefetch_candidates():
            self.prefetch_candidates += 1
            if expert in self.cache:
                self.prefetch_redundant += 1
                continue
            if expert in self.pending:
                self.prefetch_already_pending += 1
                continue
            if self.policy == "utility_gate" and not self._admit(score):
                self.prefetch_rejected += 1
                continue
            self.prefetch_id += 1
            record = PrefetchRecord(
                prefetch_id=len(self.records),
                expert=expert,
                issue_time=time,
                ready_time=time + self.cfg.transfer_latency,
            )
            self.records.append(record)
            self.pending[expert] = record
            self.prefetch_bytes_moved += self.cfg.expert_bytes

    def _prefetch_candidates(self) -> list[tuple[int, float]]:
        if not self.history:
            return []
        counts = Counter(self.history)
        total = sum(counts.values())
        ranked = counts.most_common(self.cfg.prefetch_width)
        return [(expert, count / total) for expert, count in ranked]

    def _admit(self, predicted_reuse_prob: float) -> bool:
        cache_pressure = len(self.cache) / self.cfg.cache_size
        pending_pressure = len(self.pending) / max(1, self.cfg.prefetch_width)
        expected_stall_saved = predicted_reuse_prob * self.cfg.transfer_latency
        transfer_cost = 0.65
        eviction_cost = 0.9 * cache_pressure
        contention_cost = 0.55 * pending_pressure
        score = expected_stall_saved - transfer_cost - eviction_cost - contention_cost
        return score > self.cfg.gate_threshold

    def _insert_cache(self, time: int, expert: int, loaded_by: str, prefetch_id: int | None) -> None:
        if expert in self.cache:
            self.cache.pop(expert)
        while len(self.cache) >= self.cfg.cache_size:
            victim, victim_entry = self.cache.popitem(last=False)
            self.evictions += 1
            if loaded_by == "prefetch":
                self.prefetch_evictions += 1
                victim_next_use = self.next_use[min(time, len(self.trace) - 1)].get(victim, math.inf)
                if victim_next_use != math.inf and victim_next_use - time <= self.cfg.damage_horizon:
                    self.damaging_prefetch_evictions += 1
            if victim_entry.loaded_by == "prefetch" and victim_entry.prefetch_id is not None:
                victim_record = self.records[victim_entry.prefetch_id]
                if victim_record.used_time is None and victim_record.late_use_time is None:
                    victim_record.evicted_time = time
        self.cache[expert] = CacheEntry(
            expert=expert,
            loaded_by=loaded_by,
            loaded_time=time,
            ready_time=time,
            prefetch_id=prefetch_id,
        )


def run_experiments(seed: int, cfg: SimulatorConfig) -> list[RunMetrics]:
    policies = ["no_prefetch", "history_prefetch", "utility_gate"]
    rows: list[RunMetrics] = []
    for workload in WORKLOADS:
        trace = generate_workload(workload, seed)
        for policy in policies:
            simulator = CacheSimulator(trace, workload, cfg, policy)
            rows.append(simulator.run())
    return rows


def write_csv(path: Path, rows: Iterable[RunMetrics]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_report(path: Path, rows: list[RunMetrics], cfg: SimulatorConfig) -> None:
    flagged = [row for row in rows if row.phenomenon_flag and row.policy != "no_prefetch"]
    best_lines = []
    for row in rows:
        if row.policy == "history_prefetch":
            best_lines.append(
                "| {workload} | {hit:.3f} | {utility:.3f} | {red:.3f} | {late:.3f} | {unused:.3f} | {stall:.1f} | {pollution:.3f} |".format(
                    workload=row.workload,
                    hit=row.cache_hit_ratio,
                    utility=row.timely_useful_candidate_ratio,
                    red=row.redundant_prefetch_ratio,
                    late=row.issued_late_prefetch_ratio,
                    unused=row.issued_unused_prefetch_ratio,
                    stall=row.stall_time,
                    pollution=row.eviction_damage_ratio,
                )
            )
    text = [
        "# Prefetch Utility Phenomenon Validation",
        "",
        "This run tests whether cache hit ratio can stay high while prefetch action utility is low.",
        "",
        "## Config",
        "",
        f"- cache_size: {cfg.cache_size}",
        f"- prefetch_width: {cfg.prefetch_width}",
        f"- history_window: {cfg.history_window}",
        f"- transfer_latency: {cfg.transfer_latency}",
        f"- timely_ttl: {cfg.timely_ttl}",
        "",
        "## Claim Check",
        "",
        f"- phenomenon cells flagged: {len(flagged)}",
        "- flag rule: cache_hit_ratio >= 0.70, timely_useful_candidate_ratio <= 0.30, and either redundant candidate ratio >= 0.45, issued late ratio >= 0.20, or issued unused ratio >= 0.35",
        "",
        "## History-Prefetch Summary",
        "",
        "| workload | cache_hit_ratio | timely_useful_candidate_ratio | redundant_candidate_ratio | issued_late_ratio | issued_unused_ratio | stall_time | eviction_damage_ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *best_lines,
        "",
        "## Interpretation Boundary",
        "",
        "This is a controlled trace-level harness, not full model inference. It can validate metric semantics and a plausible causal pattern before expensive kTransformers instrumentation. A production claim still needs real router traces, model execution, and hardware transfer timing.",
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def print_table(rows: list[RunMetrics]) -> None:
    columns = [
        "workload",
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
    widths = {column: len(column) for column in columns}
    rendered = []
    for row in rows:
        data = asdict(row)
        item = {}
        for column in columns:
            value = data[column]
            if isinstance(value, float):
                value = f"{value:.3f}"
            else:
                value = str(value)
            item[column] = value
            widths[column] = max(widths[column], len(value))
        rendered.append(item)
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for item in rendered:
        print("  ".join(item[column].ljust(widths[column]) for column in columns))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/prefetch_utility_validation"))
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
    rows = run_experiments(args.seed, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir / "README.md", rows, cfg)
    print_table(rows)
    print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
