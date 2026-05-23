#!/usr/bin/env python3
"""
Aggregate per-fold metrics into mean ± std summary tables and a comparison bar chart.

Usage:
    python 06_aggregate_results.py

Reads all results/*/val_metrics.json and results/*/test_metrics.json.
Writes:
    results/aggregate/all_val_results.csv
    results/aggregate/all_test_results.csv
    results/aggregate/summary_table.csv
    results/aggregate/summary_table.md
    results/aggregate/comparison_bar_chart.png
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_KEYS = ["AP", "AP_50", "AP_75", "AP_small", "AR_1", "AR_10", "AR_100"]

# Display order for the final table (most important first)
TABLE_METRICS = ["AP_50", "AP", "AP_75", "AR_100"]

# Friendly model ordering for charts (roughly: baselines last)
MODEL_ORDER = [
    "yolov8n", "yolov8s", "yolov8m",
    "yolo11n", "yolo11s", "yolo11m",
    "yolo26n", "yolo26s", "yolo26m",
    "mog2", "knn",
]


def load_all_metrics(results_dir: Path, split: str) -> list[dict]:
    rows = []
    for metrics_json in sorted(results_dir.glob(f"*/{split}_metrics.json")):
        with open(metrics_json) as f:
            rows.append(json.load(f))
    return rows


def write_flat_csv(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {path}")


def aggregate(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """
    Returns { model_name: { metric: {"mean": float, "std": float, "values": list} } }
    """
    per_model: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        model = row["model"]
        for key in METRIC_KEYS:
            val = row.get(key, -1.0)
            if val >= 0:   # -1.0 means N/A (e.g. AP_large for small-object data)
                per_model[model][key].append(val)

    result = {}
    for model, metrics in per_model.items():
        result[model] = {}
        for key, values in metrics.items():
            arr = np.array(values)
            result[model][key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "n": len(arr),
                "values": [round(v, 6) for v in values],
            }
    return result


def sort_models(models: list[str]) -> list[str]:
    """Sort models in preferred display order; unknown models go last."""
    ordered = [m for m in MODEL_ORDER if m in models]
    extras = sorted(set(models) - set(ordered))
    return ordered + extras


def write_summary_csv(agg_val: dict, agg_test: dict, path: Path):
    all_models = sort_models(list(set(agg_val) | set(agg_test)))
    rows = []
    for model in all_models:
        row = {"model": model}
        for metric in TABLE_METRICS:
            for split, agg in [("test", agg_test), ("val", agg_val)]:
                if model in agg and metric in agg[model]:
                    m = agg[model][metric]["mean"]
                    s = agg[model][metric]["std"]
                    row[f"{split}_{metric}_mean"] = round(m, 4)
                    row[f"{split}_{metric}_std"] = round(s, 4)
                else:
                    row[f"{split}_{metric}_mean"] = "N/A"
                    row[f"{split}_{metric}_std"] = "N/A"
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model"] + [
        f"{split}_{m}_{stat}"
        for m in TABLE_METRICS
        for split in ["test", "val"]
        for stat in ["mean", "std"]
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {path}")
    return rows


def write_summary_md(agg_val: dict, agg_test: dict, path: Path):
    all_models = sort_models(list(set(agg_val) | set(agg_test)))

    def fmt(agg: dict, model: str, metric: str) -> str:
        if model not in agg or metric not in agg[model]:
            return "N/A"
        d = agg[model][metric]
        return f"{d['mean']:.3f} ± {d['std']:.3f}"

    lines = [
        "# Detection Benchmark Results",
        "",
        "Primary metric: **Test AP_50** (PASCAL VOC style, appropriate for small bee objects).",
        "Reported as mean ± std across 5 fold models evaluated on the fixed test set.",
        "",
        "| Model | Test AP_50 | Test AP | Test AR_100 | Val AP_50 | Val AP |",
        "|-------|-----------|--------|------------|----------|--------|",
    ]
    for model in all_models:
        row = (
            f"| {model} "
            f"| {fmt(agg_test, model, 'AP_50')} "
            f"| {fmt(agg_test, model, 'AP')} "
            f"| {fmt(agg_test, model, 'AR_100')} "
            f"| {fmt(agg_val, model, 'AP_50')} "
            f"| {fmt(agg_val, model, 'AP')} |"
        )
        lines.append(row)

    lines += [
        "",
        "## Notes",
        "- AP_large = −1 for all models (no large bees in dataset) — reported as N/A above.",
        "- Fold 4 has fewer val frames (264) due to data distribution constraints.",
        "- MOG2/KNN warmup is unreliable for very short videos (< 30 frames).",
        "- Results sorted by model family and size (n/s/m), baselines last.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written: {path}")


def plot_comparison(agg_test: dict, path: Path):
    all_models = sort_models(list(agg_test.keys()))
    # Only models with AP_50 data
    models = [m for m in all_models if "AP_50" in agg_test.get(m, {})]
    if not models:
        print("No AP_50 data to plot.")
        return

    ap50_means = [agg_test[m]["AP_50"]["mean"] for m in models]
    ap50_stds = [agg_test[m]["AP_50"]["std"] for m in models]
    ap_means = [agg_test[m].get("AP", {}).get("mean", 0) for m in models]
    ap_stds = [agg_test[m].get("AP", {}).get("std", 0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.2), 6))
    bars1 = ax.bar(x - width / 2, ap50_means, width, yerr=ap50_stds, capsize=4,
                   label="AP@0.5", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, ap_means, width, yerr=ap_stds, capsize=4,
                   label="AP@0.5:0.95", color="darkorange", alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Detection Benchmark — Test Set Performance (mean ± std, 5 folds)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Written: {path}")


def main():
    bench_dir = Path(__file__).resolve().parent.parent
    results_dir = bench_dir / "results"
    agg_dir = results_dir / "aggregate"

    val_rows = load_all_metrics(results_dir, "val")
    test_rows = load_all_metrics(results_dir, "test")

    if not val_rows and not test_rows:
        sys.exit(f"No metrics JSON files found under {results_dir}.\nRun 05_evaluate.py first.")

    print(f"Loaded {len(val_rows)} val metrics and {len(test_rows)} test metrics.")

    write_flat_csv(val_rows, agg_dir / "all_val_results.csv")
    write_flat_csv(test_rows, agg_dir / "all_test_results.csv")

    agg_val = aggregate(val_rows)
    agg_test = aggregate(test_rows)

    write_summary_csv(agg_val, agg_test, agg_dir / "summary_table.csv")
    write_summary_md(agg_val, agg_test, agg_dir / "summary_table.md")
    plot_comparison(agg_test, agg_dir / "comparison_bar_chart.png")

    print("\nAggregation complete.")
    print(f"  Primary results: {agg_dir / 'summary_table.md'}")
    print(f"  Bar chart      : {agg_dir / 'comparison_bar_chart.png'}")


if __name__ == "__main__":
    main()
