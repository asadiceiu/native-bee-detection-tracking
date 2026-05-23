#!/usr/bin/env python3
"""
Evaluate all tracker outputs using trackeval (MOT Challenge metrics).

Usage:
    python 05_evaluate.py

Requires GT files produced by 01_prepare_gt.py.
Reads all results/{tracker_name}/*.txt files and evaluates each tracker.

Output:
    results/{tracker_name}/metrics.json
        per-sequence and aggregate metrics: MOTA, MOTP, IDF1, IDSW, Precision, Recall, F1
"""

import json
import sys
from pathlib import Path


TRACKER_NAMES = [
    "kalman_yolo",
    "kalman_gt",
    "bytetrack_yolo",
    "bytetrack_gt",
    "botsort_yolo",
    "botsort_gt",
]

EXCLUDED = {
    "Camera_43_20240611_143758",
    "Camera_43_20240619_143401",
    "Camera_43_20240619_150759",
}


def load_mot_file(path: Path) -> dict[int, list]:
    """Parse MOT format file → {frame_id: [(track_id, x1, y1, w, h)]}"""
    frames: dict[int, list] = {}
    if not path.exists():
        return frames
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        fid = int(parts[0])
        tid = int(parts[1])
        x1, y1, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        frames.setdefault(fid, []).append((tid, x1, y1, w, h))
    return frames


def load_gt_file(path: Path) -> dict[int, list]:
    """Same as load_mot_file (GT uses the same format)."""
    return load_mot_file(path)


def box_iou(b1: tuple, b2: tuple) -> float:
    """IoU between (x1, y1, w, h) boxes."""
    ax1, ay1, aw, ah = b1
    bx1, by1, bw, bh = b2
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def evaluate_sequence(gt_frames: dict, pred_frames: dict, iou_thresh: float = 0.5):
    """
    Compute CLEAR metrics for one sequence.

    Returns dict with TP, FP, FN, IDSW, n_gt per sequence.
    Uses greedy Hungarian matching on IoU.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    total_gt = 0

    # Track previous GT→pred assignment for IDSW computation
    prev_match: dict[int, int] = {}  # gt_id → pred_id

    all_frame_ids = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))

    for fid in all_frame_ids:
        gt_dets = gt_frames.get(fid, [])   # [(tid, x1, y1, w, h)]
        pr_dets = pred_frames.get(fid, [])

        total_gt += len(gt_dets)

        if len(gt_dets) == 0:
            total_fp += len(pr_dets)
            prev_match = {}
            continue

        if len(pr_dets) == 0:
            total_fn += len(gt_dets)
            prev_match = {}
            continue

        # Build IoU cost matrix
        cost = np.zeros((len(gt_dets), len(pr_dets)))
        for gi, (gt_tid, *gt_box) in enumerate(gt_dets):
            for pi, (pr_tid, *pr_box) in enumerate(pr_dets):
                cost[gi, pi] = 1.0 - box_iou(tuple(gt_box), tuple(pr_box))

        gi_arr, pi_arr = linear_sum_assignment(cost)

        matched_gt = set()
        matched_pr = set()
        current_match: dict[int, int] = {}

        for gi, pi in zip(gi_arr, pi_arr):
            if cost[gi, pi] <= 1.0 - iou_thresh:
                gt_tid = gt_dets[gi][0]
                pr_tid = pr_dets[pi][0]
                total_tp += 1
                matched_gt.add(gi)
                matched_pr.add(pi)
                current_match[gt_tid] = pr_tid
                # IDSW: same GT bee was matched to a different pred track
                if gt_tid in prev_match and prev_match[gt_tid] != pr_tid:
                    total_idsw += 1

        total_fp += len(pr_dets) - len(matched_pr)
        total_fn += len(gt_dets) - len(matched_gt)
        prev_match = current_match

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "idsw": total_idsw,
        "n_gt": total_gt,
    }


def compute_idf1(gt_frames: dict, pred_frames: dict) -> float:
    """
    Simplified IDF1: matches predicted tracks to GT tracks via longest common
    subsequence counting (identity-based), then computes IDP, IDR, IDF1.

    Uses identity matching: for each GT track, the best-matching predicted track
    is the one with maximum TP overlapping frames (IoU > 0.5).
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    # Collect unique track IDs
    gt_ids = sorted({tid for dets in gt_frames.values() for tid, *_ in dets})
    pr_ids = sorted({tid for dets in pred_frames.values() for tid, *_ in dets})

    if not gt_ids or not pr_ids:
        return 0.0

    # Count TP overlaps for each (gt_id, pr_id) pair
    overlap = np.zeros((len(gt_ids), len(pr_ids)), dtype=int)
    gi_map = {tid: i for i, tid in enumerate(gt_ids)}
    pi_map = {tid: i for i, tid in enumerate(pr_ids)}

    all_frame_ids = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))
    for fid in all_frame_ids:
        gt_dets = gt_frames.get(fid, [])
        pr_dets = pred_frames.get(fid, [])
        for gt_tid, *gt_box in gt_dets:
            for pr_tid, *pr_box in pr_dets:
                if box_iou(tuple(gt_box), tuple(pr_box)) >= 0.5:
                    overlap[gi_map[gt_tid], pi_map[pr_tid]] += 1

    # Hungarian to find best global identity assignment
    gi_arr, pi_arr = linear_sum_assignment(-overlap)

    total_idtp = int(overlap[gi_arr, pi_arr].sum())

    # Count total ground-truth and predicted detections per track
    gt_counts = {}
    for dets in gt_frames.values():
        for tid, *_ in dets:
            gt_counts[tid] = gt_counts.get(tid, 0) + 1

    pr_counts = {}
    for dets in pred_frames.values():
        for tid, *_ in dets:
            pr_counts[tid] = pr_counts.get(tid, 0) + 1

    total_gt_dets = sum(gt_counts.values())
    total_pr_dets = sum(pr_counts.values())

    idp = total_idtp / total_pr_dets if total_pr_dets > 0 else 0.0
    idr = total_idtp / total_gt_dets if total_gt_dets > 0 else 0.0
    idf1 = 2 * idp * idr / (idp + idr) if (idp + idr) > 0 else 0.0
    return idf1


def main():
    bench_dir = Path(__file__).resolve().parent.parent
    gt_base = bench_dir / "gt"
    results_base = bench_dir / "results"

    sequences = sorted(
        [d for d in gt_base.iterdir() if d.is_dir() and d.name not in EXCLUDED],
        key=lambda d: d.name,
    )

    if not sequences:
        sys.exit(f"No GT directories found under {gt_base}.\nRun 01_prepare_gt.py first.")

    print(f"GT sequences : {len(sequences)}")
    print()

    for tracker_name in TRACKER_NAMES:
        tracker_dir = results_base / tracker_name
        if not tracker_dir.exists():
            print(f"SKIP {tracker_name} — no results directory")
            continue

        txt_files = list(tracker_dir.glob("*.txt"))
        if not txt_files:
            print(f"SKIP {tracker_name} — no .txt files")
            continue

        print(f"Evaluating: {tracker_name} ({len(txt_files)} sequences found)")

        per_seq = {}
        for seq_dir in sequences:
            seq_name = seq_dir.name
            gt_file = gt_base / seq_name / "gt" / "gt.txt"
            pred_file = tracker_dir / f"{seq_name}.txt"

            if not gt_file.exists():
                print(f"  SKIP {seq_name} — GT file missing")
                continue
            if not pred_file.exists():
                print(f"  SKIP {seq_name} — predictions missing")
                continue

            gt_frames = load_gt_file(gt_file)
            pred_frames = load_mot_file(pred_file)

            metrics = evaluate_sequence(gt_frames, pred_frames)
            idf1 = compute_idf1(gt_frames, pred_frames)

            tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
            idsw = metrics["idsw"]
            n_gt = metrics["n_gt"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            mota = 1.0 - (fn + fp + idsw) / n_gt if n_gt > 0 else 0.0

            per_seq[seq_name] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mota": round(mota, 4),
                "idf1": round(idf1, 4),
                "idsw": idsw,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "n_gt": n_gt,
            }

        import numpy as np

        prec_vals = np.array([v["precision"] for v in per_seq.values()])
        rec_vals = np.array([v["recall"] for v in per_seq.values()])
        f1_vals = np.array([v["f1"] for v in per_seq.values()])
        mota_vals = np.array([v["mota"] for v in per_seq.values()])
        idf1_vals = np.array([v["idf1"] for v in per_seq.values()])
        total_idsw = sum(v["idsw"] for v in per_seq.values())

        aggregate = {
            "precision_mean": round(float(prec_vals.mean()), 4),
            "precision_std": round(float(prec_vals.std()), 4),
            "recall_mean": round(float(rec_vals.mean()), 4),
            "recall_std": round(float(rec_vals.std()), 4),
            "f1_mean": round(float(f1_vals.mean()), 4),
            "f1_std": round(float(f1_vals.std()), 4),
            "mota_mean": round(float(mota_vals.mean()), 4),
            "mota_std": round(float(mota_vals.std()), 4),
            "idf1_mean": round(float(idf1_vals.mean()), 4),
            "idf1_std": round(float(idf1_vals.std()), 4),
            "idsw_total": total_idsw,
            "n_sequences": len(per_seq),
        }

        output = {"tracker": tracker_name, "per_sequence": per_seq, "aggregate": aggregate}
        out_path = tracker_dir / "metrics.json"
        out_path.write_text(json.dumps(output, indent=2))

        p = aggregate["precision_mean"]
        r = aggregate["recall_mean"]
        f = aggregate["f1_mean"]
        m = aggregate["mota_mean"]
        i = aggregate["idf1_mean"]
        print(f"  P={p:.3f}  R={r:.3f}  F1={f:.3f}  MOTA={m:.3f}  IDF1={i:.3f}  IDSW={total_idsw}")
        print(f"  → {out_path}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
