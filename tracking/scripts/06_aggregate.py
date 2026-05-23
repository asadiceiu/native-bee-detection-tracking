#!/usr/bin/env python3
"""
Aggregate tracker metrics into a summary table matching paper Table 2 format.

Usage:
    python 06_aggregate.py

Reads:
    results/{tracker_name}/metrics.json for all 6 tracker configurations

Output:
    results/aggregate/tracking_summary.csv
    results/aggregate/tracking_summary.md
    results/aggregate/tracking_bar_chart.png
"""

import json
from pathlib import Path

import numpy as np

TRACKER_CONFIGS = [
    ("kalman_yolo",    "Kalman (IoU)",  "YOLO"),
    ("kalman_gt",      "Kalman (IoU)",  "Oracle"),
    ("bytetrack_yolo", "ByteTrack",     "YOLO"),
    ("bytetrack_gt",   "ByteTrack",     "Oracle"),
    ("botsort_yolo",   "BoT-SORT",      "YOLO"),
    ("botsort_gt",     "BoT-SORT",      "Oracle"),
]


def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def main():
    bench_dir = Path(__file__).resolve().parent.parent
    results_base = bench_dir / "results"
    agg_dir = results_base / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tracker_key, tracker_label, mode in TRACKER_CONFIGS:
        metrics_path = results_base / tracker_key / "metrics.json"
        if not metrics_path.exists():
            print(f"SKIP {tracker_key} — metrics.json not found")
            continue
        data = load_metrics(metrics_path)
        agg = data["aggregate"]
        rows.append({
            "tracker_key": tracker_key,
            "Tracker": tracker_label,
            "Mode": mode,
            "Precision": fmt(agg["precision_mean"], agg["precision_std"]),
            "Recall": fmt(agg["recall_mean"], agg["recall_std"]),
            "F1": fmt(agg["f1_mean"], agg["f1_std"]),
            "MOTA": fmt(agg["mota_mean"], agg["mota_std"]),
            "IDSW": str(agg["idsw_total"]),
            "IDF1": fmt(agg["idf1_mean"], agg["idf1_std"]),
            # raw for chart
            "_precision_mean": agg["precision_mean"],
            "_precision_std": agg["precision_std"],
            "_recall_mean": agg["recall_mean"],
            "_recall_std": agg["recall_std"],
            "_f1_mean": agg["f1_mean"],
            "_f1_std": agg["f1_std"],
            "_mota_mean": agg["mota_mean"],
            "_mota_std": agg["mota_std"],
            "_idf1_mean": agg["idf1_mean"],
            "_idf1_std": agg["idf1_std"],
        })

    if not rows:
        print("No metrics.json files found. Run 05_evaluate.py first.")
        return

    # ── Markdown table ────────────────────────────────────────────────────────
    headers = ["Tracker", "Mode", "Precision", "Recall", "F1", "MOTA", "IDSW", "IDF1"]
    col_widths = [max(len(h), max(len(str(r[h])) for r in rows)) for h in headers]

    def fmt_row(values):
        parts = [str(v).ljust(w) for v, w in zip(values, col_widths)]
        return "| " + " | ".join(parts) + " |"

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"

    lines = [
        "# Tracking Benchmark Results",
        "",
        "Mean ± std across 13 sequences (IoU threshold 0.5 for matching).",
        "",
        "## YOLO Detections (end-to-end)",
        "",
        fmt_row(headers),
        sep,
    ]

    for row in rows:
        if row["Mode"] == "YOLO":
            lines.append(fmt_row([row[h] for h in headers]))

    lines += [
        "",
        "## GT Oracle (tracker quality only)",
        "",
        fmt_row(headers),
        sep,
    ]

    for row in rows:
        if row["Mode"] == "Oracle":
            lines.append(fmt_row([row[h] for h in headers]))

    n_unique_trackers = len({r["Tracker"] for r in rows})
    n_modes = len({r["Mode"] for r in rows})
    lines += ["", f"*{n_unique_trackers} trackers × {n_modes} modes × 13 sequences*"]

    md_path = agg_dir / "tracking_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Summary table : {md_path}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    import csv
    csv_path = agg_dir / "tracking_summary.csv"
    csv_headers = ["Tracker", "Mode", "Precision", "Recall", "F1", "MOTA", "IDSW", "IDF1"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row[h] for h in csv_headers})
    print(f"CSV           : {csv_path}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        trackers = ["Kalman (IoU)", "ByteTrack", "BoT-SORT"]
        metrics_to_plot = [("MOTA", "_mota_mean", "_mota_std"),
                           ("F1",   "_f1_mean",   "_f1_std"),
                           ("IDF1", "_idf1_mean", "_idf1_std")]

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 5))
        x = np.arange(len(trackers))
        width = 0.35

        row_by_key = {r["tracker_key"]: r for r in rows}

        for ax, (metric_label, mean_key, std_key) in zip(axes, metrics_to_plot):
            yolo_means = [row_by_key[t_key][mean_key] if t_key in row_by_key else 0.0
                          for t_key, _, mode in TRACKER_CONFIGS if mode == "YOLO"]
            yolo_stds  = [row_by_key[t_key][std_key]  if t_key in row_by_key else 0.0
                          for t_key, _, mode in TRACKER_CONFIGS if mode == "YOLO"]
            gt_means   = [row_by_key[t_key][mean_key] if t_key in row_by_key else 0.0
                          for t_key, _, mode in TRACKER_CONFIGS if mode == "Oracle"]
            gt_stds    = [row_by_key[t_key][std_key]  if t_key in row_by_key else 0.0
                          for t_key, _, mode in TRACKER_CONFIGS if mode == "Oracle"]

            ax.bar(x - width / 2, yolo_means, width, yerr=yolo_stds,
                   label="YOLO", capsize=4, alpha=0.85, color="#4C72B0")
            ax.bar(x + width / 2, gt_means, width, yerr=gt_stds,
                   label="Oracle", capsize=4, alpha=0.85, color="#DD8452")

            ax.set_title(metric_label, fontsize=13, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(trackers, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.legend(fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.suptitle("Tracking Benchmark: YOLO vs GT Oracle", fontsize=14, fontweight="bold")
        plt.tight_layout()
        chart_path = agg_dir / "tracking_bar_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Bar chart     : {chart_path}")
    except ImportError:
        print("matplotlib not installed — skipping bar chart")

    print()
    print("=== Summary ===")
    print()
    for line in lines[:30]:
        print(line)


if __name__ == "__main__":
    main()
