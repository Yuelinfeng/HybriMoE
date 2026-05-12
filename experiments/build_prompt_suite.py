#!/usr/bin/env python3
"""Build prompt banks and workload streams for HybriMoE universality experiments."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


CATEGORIES = [
    "general_chat",
    "knowledge_qa",
    "math_reasoning",
    "code_generation",
    "code_debugging",
    "long_summarization",
    "translation_multilingual",
    "structured_extraction",
    "planning_agentic",
    "mixed_phase",
]

TOPICS = [
    "distributed training",
    "GPU memory management",
    "database indexing",
    "compiler optimization",
    "network congestion",
    "operating system scheduling",
    "financial risk",
    "medical triage workflow",
    "renewable energy planning",
    "supply chain forecasting",
]

DOMAINS = [
    "machine learning systems",
    "cloud infrastructure",
    "robotics",
    "cybersecurity",
    "data engineering",
    "scientific computing",
]

CODE_TASKS = [
    "an LRU cache",
    "a streaming top-k counter",
    "a JSON log parser",
    "a retry wrapper with exponential backoff",
    "a task scheduler",
    "a trie-based prefix matcher",
]

LANGUAGES = ["Python", "C++", "Rust", "Java"]

TEMPLATE_BANK = {
    "general_chat": [
        "Explain {topic} to a new engineer. Include the main intuition, two tradeoffs, and one practical example.",
        "Compare two approaches to {topic}. Give a concise recommendation for a production team.",
        "A teammate is confused about {topic}. Write a clear response that separates symptoms, root causes, and next steps.",
    ],
    "knowledge_qa": [
        "Answer this technical question about {domain}: what are the main bottlenecks, and how are they usually measured?",
        "Give a factual overview of {topic}. Mention key terms, common failure modes, and evaluation criteria.",
        "What should a system designer know about {topic} before choosing an implementation strategy?",
    ],
    "math_reasoning": [
        "Solve a resource allocation problem: {n} jobs share {m} identical workers, each job needs {k} stages. Derive the bottleneck and compute the maximum throughput.",
        "A cache has capacity {n}, requests follow three phases with hot-set sizes {m}, {k}, and {r}. Reason about hit-rate changes after each phase shift.",
        "Given transfer latency {n} ms and compute time {m} ms per task, derive when prefetching helps and when it only adds contention.",
    ],
    "code_generation": [
        "Write {language} code for {code_task}. Include edge-case handling and a short usage example.",
        "Implement {code_task} in {language}. Keep the API small and explain the complexity.",
        "Create a production-style {language} function for {code_task}. Add simple tests in the same answer.",
    ],
    "code_debugging": [
        "Debug this {language} snippet conceptually: a {code_task} sometimes returns stale results after a workload shift. Explain likely causes and fixes.",
        "Review a {language} implementation of {code_task} that is slow under bursty input. List bugs, performance risks, and tests.",
        "A {code_task} implementation works on stable data but fails after distribution shift. Propose a debugging plan and corrected logic.",
    ],
    "long_summarization": [
        "Summarize a long design review about {topic}. Preserve decisions, unresolved risks, and action items.",
        "Condense a multi-section report on {domain}. Separate background, evidence, counterarguments, and final recommendation.",
        "Read a hypothetical incident report about {topic} and produce an executive summary plus technical root-cause notes.",
    ],
    "translation_multilingual": [
        "Translate a technical explanation of {topic} from English to Spanish, then summarize it in English in three bullets.",
        "Rewrite a {domain} operations note for an international team. Keep terminology consistent across English and French phrases.",
        "Translate and simplify a user-facing explanation of {topic} for readers who know basic English but not the technical jargon.",
    ],
    "structured_extraction": [
        "Extract entities, metrics, assumptions, and risks from a paragraph about {topic}. Return compact JSON.",
        "Convert a messy operations note about {domain} into a table with fields: problem, signal, likely cause, action, owner.",
        "Given a report about {topic}, produce a schema with measurements, thresholds, and unknowns.",
    ],
    "planning_agentic": [
        "Plan a two-week investigation into {topic}. Include milestones, experiments, failure criteria, and reporting artifacts.",
        "Design an agent workflow for monitoring {domain}. Specify tools, state, triggers, and escalation rules.",
        "Create a step-by-step execution plan for improving {topic} under limited budget and noisy measurements.",
    ],
    "mixed_phase": [
        "First answer a knowledge question about {topic}. Then write pseudocode for a measurement harness. Finally summarize expected failure modes.",
        "Analyze {domain}, switch to a small coding task for {code_task}, then produce a structured checklist for validation.",
        "Start with a high-level explanation of {topic}, then solve a small quantitative example with {n} inputs, then draft a concise implementation plan.",
    ],
}


SHIFTED_HOMOGENEOUS_PHASES = [
    ("knowledge_phase", "knowledge_qa"),
    ("math_phase", "math_reasoning"),
    ("code_phase", "code_generation"),
    ("summary_phase", "long_summarization"),
]

SHIFTED_MIXED_DISTRIBUTIONS = [
    ("chat_qa_heavy", {"general_chat": 4, "knowledge_qa": 4, "long_summarization": 2}),
    ("code_heavy", {"code_generation": 4, "code_debugging": 4, "structured_extraction": 2}),
    ("reasoning_planning_heavy", {"math_reasoning": 4, "planning_agentic": 4, "mixed_phase": 2}),
    ("language_structure_heavy", {"translation_multilingual": 3, "structured_extraction": 4, "mixed_phase": 3}),
]


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in parse_csv_list(raw)]


def render_prompt(category: str, idx: int, rng: random.Random, max_new_tokens: int) -> dict[str, Any]:
    template = rng.choice(TEMPLATE_BANK[category])
    values = {
        "topic": rng.choice(TOPICS),
        "domain": rng.choice(DOMAINS),
        "code_task": rng.choice(CODE_TASKS),
        "language": rng.choice(LANGUAGES),
        "n": rng.randint(4, 64),
        "m": rng.randint(2, 16),
        "k": rng.randint(2, 8),
        "r": rng.randint(8, 96),
    }
    return {
        "prompt_id": f"{category}_{idx:04d}",
        "category": category,
        "prompt": template.format(**values),
        "max_new_tokens": max_new_tokens,
        "source": "synthetic_template_v1",
    }


def build_prompt_bank(prompts_per_category: int, seed: int, max_new_tokens: int) -> list[dict[str, Any]]:
    records = []
    for category_index, category in enumerate(CATEGORIES):
        rng = random.Random(seed * 1009 + category_index)
        for idx in range(prompts_per_category):
            records.append(render_prompt(category, idx, rng, max_new_tokens))
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def index_by_category(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
    return grouped


def choose_from_category(
    grouped: dict[str, list[dict[str, Any]]],
    category: str,
    cursor: dict[str, int],
    rng: random.Random,
) -> dict[str, Any]:
    pool = grouped[category]
    if not pool:
        raise ValueError(f"No prompts available for category: {category}")
    start = cursor[category] % len(pool)
    cursor[category] += 1
    # The random offset prevents all streams from walking each category in the same order.
    return dict(pool[(start + rng.randrange(len(pool))) % len(pool)])


def attach_stream_fields(
    record: dict[str, Any],
    *,
    stream_id: str,
    workload: str,
    phase: str,
    phase_index: int,
    sequence_index: int,
    seed: int,
) -> dict[str, Any]:
    item = dict(record)
    item.update(
        {
            "stream_id": stream_id,
            "workload": workload,
            "phase": phase,
            "phase_index": phase_index,
            "sequence_index": sequence_index,
            "seed": seed,
        }
    )
    return item


def weighted_choice(distribution: dict[str, int], rng: random.Random) -> str:
    total = sum(distribution.values())
    draw = rng.randint(1, total)
    running = 0
    for category, weight in distribution.items():
        running += weight
        if draw <= running:
            return category
    return next(iter(distribution))


def build_stable_homogeneous_streams(
    grouped: dict[str, list[dict[str, Any]]],
    categories: list[str],
    stream_length: int,
    seeds: list[int],
) -> dict[str, list[dict[str, Any]]]:
    streams = {}
    for seed in seeds:
        for category in categories:
            rng = random.Random(seed * 10007 + CATEGORIES.index(category))
            cursor: dict[str, int] = defaultdict(int)
            stream_id = f"stable_homogeneous_{category}_seed{seed}"
            rows = [
                attach_stream_fields(
                    choose_from_category(grouped, category, cursor, rng),
                    stream_id=stream_id,
                    workload="stable_homogeneous",
                    phase=category,
                    phase_index=0,
                    sequence_index=idx,
                    seed=seed,
                )
                for idx in range(stream_length)
            ]
            streams[stream_id] = rows
    return streams


def build_stable_mixed_streams(
    grouped: dict[str, list[dict[str, Any]]], stream_length: int, seeds: list[int]
) -> dict[str, list[dict[str, Any]]]:
    streams = {}
    for seed in seeds:
        rng = random.Random(seed * 7919 + 17)
        cursor: dict[str, int] = defaultdict(int)
        stream_id = f"stable_mixed_seed{seed}"
        rows = []
        for idx in range(stream_length):
            category = rng.choice(CATEGORIES)
            rows.append(
                attach_stream_fields(
                    choose_from_category(grouped, category, cursor, rng),
                    stream_id=stream_id,
                    workload="stable_mixed",
                    phase="uniform_mix",
                    phase_index=0,
                    sequence_index=idx,
                    seed=seed,
                )
            )
        streams[stream_id] = rows
    return streams


def build_shifted_homogeneous_streams(
    grouped: dict[str, list[dict[str, Any]]], stream_length: int, seeds: list[int]
) -> dict[str, list[dict[str, Any]]]:
    streams = {}
    phase_len = max(1, stream_length // len(SHIFTED_HOMOGENEOUS_PHASES))
    for seed in seeds:
        rng = random.Random(seed * 1543 + 31)
        cursor: dict[str, int] = defaultdict(int)
        stream_id = f"shifted_homogeneous_seed{seed}"
        rows = []
        idx = 0
        for phase_index, (phase, category) in enumerate(SHIFTED_HOMOGENEOUS_PHASES):
            while idx < stream_length and len(rows) < (phase_index + 1) * phase_len:
                rows.append(
                    attach_stream_fields(
                        choose_from_category(grouped, category, cursor, rng),
                        stream_id=stream_id,
                        workload="shifted_homogeneous",
                        phase=phase,
                        phase_index=phase_index,
                        sequence_index=idx,
                        seed=seed,
                    )
                )
                idx += 1
        while idx < stream_length:
            phase_index = len(SHIFTED_HOMOGENEOUS_PHASES) - 1
            phase, category = SHIFTED_HOMOGENEOUS_PHASES[-1]
            rows.append(
                attach_stream_fields(
                    choose_from_category(grouped, category, cursor, rng),
                    stream_id=stream_id,
                    workload="shifted_homogeneous",
                    phase=phase,
                    phase_index=phase_index,
                    sequence_index=idx,
                    seed=seed,
                )
            )
            idx += 1
        streams[stream_id] = rows
    return streams


def build_shifted_mixed_streams(
    grouped: dict[str, list[dict[str, Any]]], stream_length: int, seeds: list[int]
) -> dict[str, list[dict[str, Any]]]:
    streams = {}
    phase_len = max(1, stream_length // len(SHIFTED_MIXED_DISTRIBUTIONS))
    for seed in seeds:
        rng = random.Random(seed * 3571 + 47)
        cursor: dict[str, int] = defaultdict(int)
        stream_id = f"shifted_mixed_seed{seed}"
        rows = []
        idx = 0
        for phase_index, (phase, distribution) in enumerate(SHIFTED_MIXED_DISTRIBUTIONS):
            while idx < stream_length and len(rows) < (phase_index + 1) * phase_len:
                category = weighted_choice(distribution, rng)
                rows.append(
                    attach_stream_fields(
                        choose_from_category(grouped, category, cursor, rng),
                        stream_id=stream_id,
                        workload="shifted_mixed",
                        phase=phase,
                        phase_index=phase_index,
                        sequence_index=idx,
                        seed=seed,
                    )
                )
                idx += 1
        while idx < stream_length:
            phase_index = len(SHIFTED_MIXED_DISTRIBUTIONS) - 1
            phase, distribution = SHIFTED_MIXED_DISTRIBUTIONS[-1]
            category = weighted_choice(distribution, rng)
            rows.append(
                attach_stream_fields(
                    choose_from_category(grouped, category, cursor, rng),
                    stream_id=stream_id,
                    workload="shifted_mixed",
                    phase=phase,
                    phase_index=phase_index,
                    sequence_index=idx,
                    seed=seed,
                )
            )
            idx += 1
        streams[stream_id] = rows
    return streams


def write_experiment_matrix(
    path: Path,
    stream_paths: list[Path],
    cache_sizes: list[int],
    prefetch_sizes: list[int],
) -> None:
    def infer_workload(stream_id: str) -> str:
        for name in ["stable_homogeneous", "stable_mixed", "shifted_homogeneous", "shifted_mixed"]:
            if stream_id.startswith(name):
                return name
        return "unknown"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "stream_file", "stream_id", "workload", "cache_size", "prefetch_size"],
        )
        writer.writeheader()
        for stream_path in stream_paths:
            stream_id = stream_path.stem
            workload = infer_workload(stream_id)
            for cache_size in cache_sizes:
                for prefetch_size in prefetch_sizes:
                    writer.writerow(
                        {
                            "run_id": f"{stream_id}_cache{cache_size}_prefetch{prefetch_size}",
                            "stream_file": str(stream_path),
                            "stream_id": stream_id,
                            "workload": workload,
                            "cache_size": cache_size,
                            "prefetch_size": prefetch_size,
                        }
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/prompt_suite_v1"))
    parser.add_argument("--prompts-per-category", type=int, default=100)
    parser.add_argument("--stream-length", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0, help="Seed for prompt text generation.")
    parser.add_argument("--stream-seeds", default="0,1,2")
    parser.add_argument(
        "--workloads",
        default="stable_homogeneous,stable_mixed,shifted_homogeneous,shifted_mixed",
        help="Comma-separated workload families to emit.",
    )
    parser.add_argument(
        "--homogeneous-categories",
        default="general_chat,math_reasoning,code_generation,long_summarization,mixed_phase",
        help="Comma-separated categories used for stable_homogeneous streams; use 'all' for every category.",
    )
    parser.add_argument("--cache-sizes", default="16,32,48,56")
    parser.add_argument("--prefetch-sizes", default="0,4,8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = args.output_dir / "streams"
    stream_dir.mkdir(parents=True, exist_ok=True)

    records = build_prompt_bank(args.prompts_per_category, args.seed, args.max_new_tokens)
    write_jsonl(args.output_dir / "prompts.jsonl", records)
    grouped = index_by_category(records)

    stream_seeds = parse_int_list(args.stream_seeds)
    workloads = set(parse_csv_list(args.workloads))
    if args.homogeneous_categories.strip().lower() == "all":
        homogeneous_categories = CATEGORIES
    else:
        homogeneous_categories = parse_csv_list(args.homogeneous_categories)
    invalid = sorted(set(homogeneous_categories) - set(CATEGORIES))
    if invalid:
        raise ValueError(f"Unknown homogeneous categories: {invalid}")

    streams: dict[str, list[dict[str, Any]]] = {}
    if "stable_homogeneous" in workloads:
        streams.update(
            build_stable_homogeneous_streams(grouped, homogeneous_categories, args.stream_length, stream_seeds)
        )
    if "stable_mixed" in workloads:
        streams.update(build_stable_mixed_streams(grouped, args.stream_length, stream_seeds))
    if "shifted_homogeneous" in workloads:
        streams.update(build_shifted_homogeneous_streams(grouped, args.stream_length, stream_seeds))
    if "shifted_mixed" in workloads:
        streams.update(build_shifted_mixed_streams(grouped, args.stream_length, stream_seeds))

    stream_paths = []
    for stream_id, rows in sorted(streams.items()):
        path = stream_dir / f"{stream_id}.jsonl"
        write_jsonl(path, rows)
        stream_paths.append(path)

    write_experiment_matrix(
        args.output_dir / "experiment_matrix.csv",
        stream_paths,
        parse_int_list(args.cache_sizes),
        parse_int_list(args.prefetch_sizes),
    )

    manifest = {
        "schema": "hybrimoe.prompt_suite.v1",
        "categories": CATEGORIES,
        "prompts_per_category": args.prompts_per_category,
        "prompt_count": len(records),
        "stream_length": args.stream_length,
        "stream_seeds": stream_seeds,
        "stream_count": len(stream_paths),
        "streams": [str(path) for path in stream_paths],
        "cache_sizes": parse_int_list(args.cache_sizes),
        "prefetch_sizes": parse_int_list(args.prefetch_sizes),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[prompt-suite] prompts={len(records)} streams={len(stream_paths)} output={args.output_dir}")


if __name__ == "__main__":
    main()
