#!/usr/bin/env python3
"""Plot paper-style motivation figures from HybriMoE live trace runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _as_float(value: Any) -> float:
    return float(value)


def _load_router_policy(run_dir: Path, policy: str) -> dict[str, str]:
    path = run_dir / "router_trace" / "analysis_decode" / "summary.csv"
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("policy") == policy:
                return row
    raise ValueError(f"policy={policy} not found in {path}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prefetch_size(run_dir: Path) -> int:
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        cfg = _load_json(cfg_path)
        if "prefetch_size" in cfg:
            return int(cfg["prefetch_size"])
    match = re.search(r"prefetch(\d+)", run_dir.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer prefetch size from {run_dir}")


def _expert_set_from_router_event(event: dict[str, Any]) -> set[int]:
    if "expert_counts" in event:
        return {int(expert) for expert, count in event["expert_counts"] if int(count) > 0}
    result: set[int] = set()
    for token_topk in event.get("topk_indices", []):
        result.update(int(expert) for expert in token_topk)
    return result


def _same_layer_jaccards(router_trace: Path) -> list[float]:
    by_layer: dict[int, list[set[int]]] = {}
    with router_trace.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("schema") != "hybrimoe.router_assignments.v1":
                continue
            if event.get("stage") != "decode":
                continue
            layer_idx = int(event["layer_idx"])
            by_layer.setdefault(layer_idx, []).append(_expert_set_from_router_event(event))

    values: list[float] = []
    for sets in by_layer.values():
        for previous, current in zip(sets, sets[1:]):
            union = previous | current
            values.append(len(previous & current) / len(union) if union else 1.0)
    return values


def load_runs(input_root: Path) -> list[dict[str, Any]]:
    runs = []
    for run_dir in sorted(input_root.iterdir()):
        if not run_dir.is_dir():
            continue
        expert_summary_path = run_dir / "expert_cache_trace" / "analysis_decode" / "summary.json"
        router_trace_path = run_dir / "router_trace" / "router_trace.jsonl"
        if not expert_summary_path.exists() or not router_trace_path.exists():
            continue

        expert = _load_json(expert_summary_path)
        router_history = _load_router_policy(run_dir, "history_prefetch")
        router_no_prefetch = _load_router_policy(run_dir, "no_prefetch")
        placement = expert["placement"]
        prefetch = expert["prefetch"]
        match = expert["prefetch_match"]
        load_evict = expert["load_evict"]
        jaccards = _same_layer_jaccards(router_trace_path)

        runs.append(
            {
                "run": run_dir.name,
                "prefetch_size": _prefetch_size(run_dir),
                "live_cache_hit": placement["resident_hit_expert_ratio"],
                "live_cache_hit_assignment": placement["resident_hit_assignment_ratio"],
                "cpu_assignment_ratio": placement["cpu_assignment_ratio"],
                "prefetch_candidates": prefetch["prefetch_candidate_experts"],
                "prefetch_redundant_ratio": prefetch["prefetch_redundant_candidate_ratio"],
                "prefetch_issued": prefetch["prefetch_issued_load_experts"],
                "issued_used_next_ratio": match["prefetch_issued_used_by_next_layer_ratio"],
                "candidate_used_next_ratio": match["prefetch_candidate_used_by_next_layer_ratio"],
                "issued_assignment_coverage": match["prefetch_issued_assignment_coverage_ratio"],
                "candidate_assignment_coverage": match["prefetch_candidate_assignment_coverage_ratio"],
                "loads": load_evict["loaded_experts"],
                "evicts": load_evict["evicted_experts"],
                "evictions_per_load": load_evict["evictions_per_loaded_expert"],
                "router_cache_hit": _as_float(router_history["cache_hit_ratio"]),
                "router_timely_candidate": _as_float(router_history["timely_useful_candidate_ratio"]),
                "router_timely_issued": _as_float(router_history["timely_useful_issued_ratio"]),
                "router_issued_unused": _as_float(router_history["issued_unused_prefetch_ratio"]),
                "router_stall": _as_float(router_history["stall_time"]),
                "no_prefetch_router_stall": _as_float(router_no_prefetch["stall_time"]),
                "same_layer_jaccards": jaccards,
                "avg_same_layer_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
            }
        )
    if not runs:
        raise ValueError(f"No usable run directories found under {input_root}")
    return sorted(runs, key=lambda item: item["prefetch_size"])


def write_figure_data(out_dir: Path, runs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run",
        "prefetch_size",
        "live_cache_hit",
        "issued_used_next_ratio",
        "issued_assignment_coverage",
        "prefetch_redundant_ratio",
        "loads",
        "evicts",
        "extra_loads",
        "extra_evicts",
        "avg_same_layer_jaccard",
        "router_cache_hit",
        "router_timely_candidate",
        "router_issued_unused",
    ]
    baseline = runs[0]
    with (out_dir / "figure_data.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = {name: run.get(name) for name in fieldnames}
            row["extra_loads"] = run["loads"] - baseline["loads"]
            row["extra_evicts"] = run["evicts"] - baseline["evicts"]
            writer.writerow(row)


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cache_utility(out_dir: Path, runs: list[dict[str, Any]]) -> None:
    x = list(range(len(runs)))
    labels = [str(run["prefetch_size"]) for run in runs]
    width = 0.24
    metrics = [
        ("Live cache hit", [run["live_cache_hit"] * 100 for run in runs], "#4C78A8"),
        ("Issued prefetch used", [run["issued_used_next_ratio"] * 100 for run in runs], "#F58518"),
        ("Assignment covered", [run["issued_assignment_coverage"] * 100 for run in runs], "#54A24B"),
    ]

    fig, ax = plt.subplots(figsize=(4.25, 2.45))
    offsets = [-width, 0.0, width]
    for offset, (label, values, color) in zip(offsets, metrics):
        bars = ax.bar([item + offset for item in x], values, width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            if value > 0:
                text = f"{value:.0f}" if value >= 9 else f"{value:.1f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.0,
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )

    ax.set_title("Residency vs. Prefetch Utility", loc="left", pad=24)
    ax.set_xlabel("Prefetch width")
    ax.set_ylabel("Ratio (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 62)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.16))
    _save(fig, out_dir, "fig1_cache_utility_disconnect")


def plot_cache_pressure(out_dir: Path, runs: list[dict[str, Any]]) -> None:
    x = list(range(len(runs)))
    labels = [str(run["prefetch_size"]) for run in runs]
    baseline = runs[0]
    extra_loads = [run["loads"] - baseline["loads"] for run in runs]
    extra_evicts = [run["evicts"] - baseline["evicts"] for run in runs]
    redundant = [
        (run["prefetch_redundant_ratio"] * 100 if run["prefetch_candidates"] else math.nan)
        for run in runs
    ]
    width = 0.28

    fig, ax = plt.subplots(figsize=(4.25, 2.45))
    load_bars = ax.bar([item - width / 2 for item in x], extra_loads, width=width, label="Extra loads", color="#4C78A8")
    evict_bars = ax.bar([item + width / 2 for item in x], extra_evicts, width=width, label="Extra evictions", color="#E45756")
    for bars in (load_bars, evict_bars):
        for bar in bars:
            value = bar.get_height()
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 10,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )
    ax.set_title("Prefetch-Induced Cache Churn", loc="left", pad=24)
    ax.set_xlabel("Prefetch width")
    ax.set_ylabel("Extra events vs width 0")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(max(extra_loads), max(extra_evicts)) * 1.22 + 1)
    ax.legend(frameon=False, loc="upper left")

    ax2 = ax.twinx()
    line = ax2.plot(x, redundant, marker="o", color="#72B7B2", linewidth=1.8, label="Redundant candidates")
    ax2.set_ylabel("Redundant candidates (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    handles = [load_bars, evict_bars, line[0]]
    labels = ["Extra loads", "Extra evictions", "Redundant candidates"]
    ax.legend(handles, labels, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.16))
    _save(fig, out_dir, "fig2_prefetch_cache_pressure")


def plot_routing_instability(out_dir: Path, runs: list[dict[str, Any]]) -> None:
    data = [run["same_layer_jaccards"] for run in runs]
    labels = [str(run["prefetch_size"]) for run in runs]
    fig, ax = plt.subplots(figsize=(4.25, 2.45))
    box = ax.boxplot(
        data,
        labels=labels,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.3},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#B7D1EA")
        patch.set_alpha(0.95)

    means = [run["avg_same_layer_jaccard"] for run in runs]
    ax.plot(range(1, len(runs) + 1), means, marker="D", color="#F58518", linewidth=1.5, label="Mean")
    ax.set_title("Weak Short-Term Expert Reuse", loc="left", pad=10)
    ax.set_xlabel("Prefetch width")
    ax.set_ylabel("Same-layer Jaccard")
    ax.set_ylim(0, 0.82)
    ax.legend(frameon=False, loc="upper right")
    _save(fig, out_dir, "fig3_dynamic_routing_instability")


def write_readme(out_dir: Path, runs: list[dict[str, Any]]) -> None:
    lines = [
        "# Live Trace Motivation Figures",
        "",
        "These figures are generated from live HybriMoE expert-cache traces and router traces.",
        "",
        "Generated files:",
        "",
        "- `fig1_cache_utility_disconnect.pdf/png`: cache residency versus issued prefetch utility.",
        "- `fig2_prefetch_cache_pressure.pdf/png`: extra load/evict pressure as prefetch width increases.",
        "- `fig3_dynamic_routing_instability.pdf/png`: same-layer consecutive decode expert-set Jaccard.",
        "- `figure_data.csv`: numeric data used by the plots.",
        "",
        "Recommended paper placement: Motivation / Observation / Pathology Analysis.",
        "Do not present these short 9-token decode runs as final end-to-end performance evaluation.",
        "",
        "Summary:",
        "",
    ]
    for run in runs:
        lines.append(
            f"- prefetch={run['prefetch_size']}: live_hit={run['live_cache_hit']:.3f}, "
            f"issued_used={run['issued_used_next_ratio']:.3f}, "
            f"assignment_coverage={run['issued_assignment_coverage']:.3f}, "
            f"loads={run['loads']}, evicts={run['evicts']}, "
            f"avg_jaccard={run['avg_same_layer_jaccard']:.3f}"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(".codex_tmp/live_trace_uploaded"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/live_trace_figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_style()
    runs = load_runs(args.input_root)
    write_figure_data(args.output_dir, runs)
    plot_cache_utility(args.output_dir, runs)
    plot_cache_pressure(args.output_dir, runs)
    plot_routing_instability(args.output_dir, runs)
    write_readme(args.output_dir, runs)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
