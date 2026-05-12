#!/usr/bin/env python3
"""Build HybriMoE prompt streams from a public/local dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CATEGORIES = [
    "open_qa",
    "closed_qa",
    "classification",
    "summarization",
    "information_extraction",
    "creative_writing",
    "brainstorming",
    "general",
]

CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


KEYWORD_BUCKETS = [
    ("code", re.compile(r"\b(code|python|java|c\+\+|rust|debug|function|algorithm|program)\b", re.I)),
    ("math_reasoning", re.compile(r"\b(calculate|solve|equation|probability|proof|derive|math)\b", re.I)),
    ("summarization", re.compile(r"\b(summarize|summary|tl;dr|condense)\b", re.I)),
    ("translation", re.compile(r"\b(translate|translation|spanish|french|chinese|german)\b", re.I)),
    ("structured_extraction", re.compile(r"\b(json|extract|table|schema|entity|entities|field)\b", re.I)),
    ("planning", re.compile(r"\b(plan|steps|strategy|roadmap|schedule|workflow)\b", re.I)),
]


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in parse_csv_list(raw)]


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_hf_dataset(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with `python -m pip install datasets`."
        ) from exc

    kwargs: dict[str, Any] = {"split": args.split}
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config, **kwargs)
    else:
        dataset = load_dataset(args.dataset_name, **kwargs)
    return dataset


def first_present(row: dict[str, Any], fields: list[str]) -> Any:
    for field in fields:
        if field and field in row and row[field] not in (None, ""):
            return row[field]
    return None


def messages_to_prompt(messages: Any) -> str:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return messages.strip()
    if not isinstance(messages, list):
        return ""
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content}" if role else content)
    return "\n".join(parts).strip()


def normalize_category(raw: Any, prompt: str) -> str:
    if raw not in (None, ""):
        value = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
        return value[:80] if value else "general"
    for category, pattern in KEYWORD_BUCKETS:
        if pattern.search(prompt):
            return category
    return "general"


def row_to_prompt(row: dict[str, Any], args: argparse.Namespace, idx: int) -> dict[str, Any] | None:
    if args.preset == "mmlu":
        return mmlu_row_to_prompt(row, args, idx)

    prompt = ""
    if args.messages_field:
        prompt = messages_to_prompt(row.get(args.messages_field))
    if not prompt and args.text_field:
        prompt = str(row.get(args.text_field, "")).strip()
    if not prompt:
        instruction = first_present(row, parse_csv_list(args.instruction_fields))
        input_text = first_present(row, parse_csv_list(args.input_fields))
        if instruction:
            prompt = str(instruction).strip()
            if input_text:
                prompt = f"{prompt}\n\nInput:\n{str(input_text).strip()}"
    if not prompt:
        return None
    if len(prompt) < args.min_chars:
        return None
    if args.max_chars and len(prompt) > args.max_chars:
        prompt = prompt[: args.max_chars]
    category_raw = row.get(args.category_field) if args.category_field else None
    category = normalize_category(category_raw, prompt)
    prompt_id_raw = row.get(args.id_field) if args.id_field else None
    prompt_id = str(prompt_id_raw) if prompt_id_raw not in (None, "") else f"dataset_{idx:06d}"
    return {
        "prompt_id": prompt_id,
        "category": category,
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "source": args.dataset_name or str(args.input_jsonl),
    }


def mmlu_row_to_prompt(row: dict[str, Any], args: argparse.Namespace, idx: int) -> dict[str, Any] | None:
    question = str(row.get("question", "")).strip()
    choices = row.get("choices")
    if not question or not isinstance(choices, list) or not choices:
        return None

    choice_lines = []
    for choice_idx, choice in enumerate(choices):
        label = CHOICE_LABELS[choice_idx] if choice_idx < len(CHOICE_LABELS) else str(choice_idx)
        choice_lines.append(f"{label}. {str(choice).strip()}")

    prompt = (
        "Answer the following multiple-choice question. "
        "Choose the single best option and include a brief explanation.\n\n"
        f"Question: {question}\n\n"
        "Choices:\n"
        + "\n".join(choice_lines)
        + "\n\nAnswer:"
    )
    if args.max_chars and len(prompt) > args.max_chars:
        return None

    answer = row.get("answer")
    answer_label = None
    if isinstance(answer, int) and 0 <= answer < len(CHOICE_LABELS):
        answer_label = CHOICE_LABELS[answer]
    elif answer not in (None, ""):
        answer_label = str(answer)

    subject = row.get("subject") or row.get("category") or row.get("task") or "mmlu"
    category = str(subject).strip().lower().replace(" ", "_").replace("-", "_")
    return {
        "prompt_id": f"mmlu_{idx:06d}",
        "category": category,
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "source": args.dataset_name or str(args.input_jsonl),
        "answer": answer,
        "answer_label": answer_label,
        "subject": str(subject),
    }


def load_prompt_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = load_jsonl(args.input_jsonl) if args.input_jsonl else load_hf_dataset(args)
    records = []
    seen: set[str] = set()
    for idx, row in enumerate(rows):
        item = row_to_prompt(dict(row), args, idx)
        if item is None:
            continue
        key = item["prompt"]
        if args.dedup and key in seen:
            continue
        seen.add(key)
        records.append(item)
        if args.max_prompts and len(records) >= args.max_prompts:
            break
    if not records:
        raise ValueError("No usable prompts were extracted from the dataset.")
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def group_records(records: list[dict[str, Any]], min_category_count: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["category"])].append(record)
    strong = {category: rows for category, rows in grouped.items() if len(rows) >= min_category_count}
    if not strong:
        strong = {"all": records}
    return strong


def choose(grouped: dict[str, list[dict[str, Any]]], category: str, rng: random.Random) -> dict[str, Any]:
    return dict(rng.choice(grouped[category]))


def attach(record: dict[str, Any], stream_id: str, workload: str, phase: str, phase_index: int, idx: int, seed: int):
    item = dict(record)
    item.update(
        {
            "stream_id": stream_id,
            "workload": workload,
            "phase": phase,
            "phase_index": phase_index,
            "sequence_index": idx,
            "seed": seed,
        }
    )
    return item


def stable_mixed(grouped: dict[str, list[dict[str, Any]]], stream_length: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed * 7919 + 17)
    categories = sorted(grouped)
    stream_id = f"stable_mixed_seed{seed}"
    rows = []
    for idx in range(stream_length):
        category = rng.choice(categories)
        rows.append(attach(choose(grouped, category, rng), stream_id, "stable_mixed", "dataset_mix", 0, idx, seed))
    return rows


def shifted_mixed(grouped: dict[str, list[dict[str, Any]]], stream_length: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed * 3571 + 47)
    categories = sorted(grouped)
    if len(categories) < 2:
        return stable_mixed(grouped, stream_length, seed)
    phase_count = min(4, len(categories))
    phase_len = max(1, stream_length // phase_count)
    stream_id = f"shifted_mixed_seed{seed}"
    rows = []
    for phase_index in range(phase_count):
        focus = categories[phase_index::phase_count] or categories
        phase = "phase_" + "_".join(focus[:2])
        while len(rows) < min(stream_length, (phase_index + 1) * phase_len):
            category = rng.choice(focus)
            rows.append(
                attach(choose(grouped, category, rng), stream_id, "shifted_mixed", phase, phase_index, len(rows), seed)
            )
    while len(rows) < stream_length:
        category = rng.choice(categories)
        rows.append(attach(choose(grouped, category, rng), stream_id, "shifted_mixed", "tail_mix", phase_count, len(rows), seed))
    return rows


def stable_homogeneous(grouped: dict[str, list[dict[str, Any]]], stream_length: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    streams = {}
    for category in sorted(grouped):
        rng = random.Random(seed * 10007 + hash(category) % 997)
        stream_id = f"stable_homogeneous_{category}_seed{seed}"
        streams[stream_id] = [
            attach(choose(grouped, category, rng), stream_id, "stable_homogeneous", category, 0, idx, seed)
            for idx in range(stream_length)
        ]
    return streams


def shifted_homogeneous(grouped: dict[str, list[dict[str, Any]]], stream_length: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed * 1543 + 31)
    categories = sorted(grouped)
    phase_count = min(4, len(categories))
    phase_len = max(1, stream_length // phase_count)
    stream_id = f"shifted_homogeneous_seed{seed}"
    rows = []
    for phase_index, category in enumerate(categories[:phase_count]):
        while len(rows) < min(stream_length, (phase_index + 1) * phase_len):
            rows.append(
                attach(choose(grouped, category, rng), stream_id, "shifted_homogeneous", category, phase_index, len(rows), seed)
            )
    while len(rows) < stream_length:
        category = categories[-1]
        rows.append(attach(choose(grouped, category, rng), stream_id, "shifted_homogeneous", category, phase_count - 1, len(rows), seed))
    return rows


def write_matrix(path: Path, stream_paths: list[Path], cache_sizes: list[int], prefetch_sizes: list[int]) -> None:
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
            for cache_size in cache_sizes:
                for prefetch_size in prefetch_sizes:
                    writer.writerow(
                        {
                            "run_id": f"{stream_id}_cache{cache_size}_prefetch{prefetch_size}",
                            "stream_file": str(stream_path),
                            "stream_id": stream_id,
                            "workload": infer_workload(stream_id),
                            "cache_size": cache_size,
                            "prefetch_size": prefetch_size,
                        }
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/dataset_prompt_suite"))
    parser.add_argument("--preset", choices=["generic", "mmlu"], default="generic")
    parser.add_argument("--dataset-name", default="databricks/databricks-dolly-15k")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--text-field", default="")
    parser.add_argument("--messages-field", default="")
    parser.add_argument("--instruction-fields", default="instruction,prompt,question")
    parser.add_argument("--input-fields", default="context,input")
    parser.add_argument("--category-field", default="category")
    parser.add_argument("--id-field", default="")
    parser.add_argument("--max-prompts", type=int, default=1000)
    parser.add_argument("--stream-length", type=int, default=100)
    parser.add_argument("--stream-seeds", default="0,1,2")
    parser.add_argument("--workloads", default="stable_mixed,shifted_mixed")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=4096)
    parser.add_argument("--min-category-count", type=int, default=20)
    parser.add_argument("--cache-sizes", default="16,32,48,56")
    parser.add_argument("--prefetch-sizes", default="0,4,8")
    parser.add_argument("--dedup", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = args.output_dir / "streams"
    stream_dir.mkdir(parents=True, exist_ok=True)

    records = load_prompt_records(args)
    grouped = group_records(records, args.min_category_count)
    write_jsonl(args.output_dir / "prompts.jsonl", records)

    stream_paths: list[Path] = []
    workloads = set(parse_csv_list(args.workloads))
    for seed in parse_int_list(args.stream_seeds):
        streams: dict[str, list[dict[str, Any]]] = {}
        if "stable_mixed" in workloads:
            streams[f"stable_mixed_seed{seed}"] = stable_mixed(grouped, args.stream_length, seed)
        if "shifted_mixed" in workloads:
            streams[f"shifted_mixed_seed{seed}"] = shifted_mixed(grouped, args.stream_length, seed)
        if "stable_homogeneous" in workloads:
            streams.update(stable_homogeneous(grouped, args.stream_length, seed))
        if "shifted_homogeneous" in workloads:
            streams[f"shifted_homogeneous_seed{seed}"] = shifted_homogeneous(grouped, args.stream_length, seed)

        for stream_id, rows in sorted(streams.items()):
            path = stream_dir / f"{stream_id}.jsonl"
            write_jsonl(path, rows)
            stream_paths.append(path)

    write_matrix(
        args.output_dir / "experiment_matrix.csv",
        stream_paths,
        parse_int_list(args.cache_sizes),
        parse_int_list(args.prefetch_sizes),
    )
    category_counts = {category: len(rows) for category, rows in sorted(grouped.items())}
    manifest = {
        "schema": "hybrimoe.dataset_prompt_suite.v1",
        "dataset_name": args.dataset_name if not args.input_jsonl else str(args.input_jsonl),
        "split": args.split,
        "prompt_count": len(records),
        "stream_count": len(stream_paths),
        "stream_length": args.stream_length,
        "stream_seeds": parse_int_list(args.stream_seeds),
        "workloads": sorted(workloads),
        "category_counts": category_counts,
        "streams": [str(path) for path in stream_paths],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[dataset-prompt-suite] dataset={manifest['dataset_name']} prompts={len(records)} "
        f"categories={len(grouped)} streams={len(stream_paths)} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
