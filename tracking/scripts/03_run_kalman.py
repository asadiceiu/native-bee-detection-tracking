#!/usr/bin/env python3
"""
Run the IoU-based Kalman tracker on all 13 sequences.

Usage:
    python 03_run_kalman.py --source yolo
    python 03_run_kalman.py --source gt

Source modes:
    yolo : reads results/detections/{seq}.csv (run 02_run_detections.py first)
    gt   : reads GT labels from clean_tracking/{seq}/labels/ (oracle mode)

Output per sequence (MOT format):
    results/kalman_{yolo|gt}/{seq}.txt
        frame_id, track_id, left, top, width, height, conf, class, visibility
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

IMG_W = 1280
IMG_H = 720

EXCLUDED = {
    "Camera_43_20240611_143758",
    "Camera_43_20240619_143401",
    "Camera_43_20240619_150759",
}


def parse_frame_id(filename: str) -> int:
    stem = Path(filename).stem
    return int(stem.split("_frame_")[-1])


def load_yolo_detections(csv_path: Path) -> dict[int, np.ndarray]:
    """Return {frame_id: (N, 4) array of [x1, y1, x2, y2]}."""
    frames: dict[int, list] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            fid = int(row["frame_id"])
            box = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
            frames.setdefault(fid, []).append(box)
    return {fid: np.array(boxes) for fid, boxes in frames.items()}


def load_gt_detections(labels_dir: Path) -> dict[int, np.ndarray]:
    """Read YOLO labels; return {seq_frame_id: (N, 4) array of [x1, y1, x2, y2]}.

    Uses 1-indexed sequential frame IDs (sorted by absolute frame number) to
    normalize sequences that don't start at frame 1 in the original video.
    """
    label_files = sorted(labels_dir.glob("*.txt"), key=lambda p: parse_frame_id(p.name))
    frames: dict[int, list] = {}
    for seq_idx, lf in enumerate(label_files):
        fid = seq_idx + 1
        boxes = []
        for line in lf.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx_n, cy_n, w_n, h_n = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            w_px = w_n * IMG_W
            h_px = h_n * IMG_H
            x1 = cx_n * IMG_W - w_px / 2
            y1 = cy_n * IMG_H - h_px / 2
            boxes.append([x1, y1, x1 + w_px, y1 + h_px])
        if boxes:
            frames[fid] = boxes
    return {fid: np.array(boxes) for fid, boxes in frames.items()}


def run_tracker(frames: dict[int, np.ndarray], max_frames: int, tracker) -> list[str]:
    """Run tracker over all frames; return MOT-format lines."""
    tracker.reset()
    rows = []
    for fid in range(1, max_frames + 1):
        dets = frames.get(fid, np.empty((0, 4)))
        active = tracker.update(dets)
        for track_id, box in active:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            rows.append(f"{fid},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run IoU-based Kalman tracker on 13 sequences")
    parser.add_argument("--source", required=True, choices=["yolo", "gt"])
    parser.add_argument("--max-age", type=int, default=10)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument(
        "--clean-tracking-dir",
        default=None,
        help="Path to clean_tracking directory (auto-detected if omitted)",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if output already exists")
    args = parser.parse_args()

    bench_dir = Path(__file__).resolve().parent.parent
    if args.clean_tracking_dir:
        clean_dir = Path(args.clean_tracking_dir).resolve()
    else:
        clean_dir = bench_dir.parent.parent / "tracking" / "clean_tracking"
        if not clean_dir.exists():
            clean_dir = bench_dir.parent / "clean_tracking"

    if not clean_dir.exists():
        sys.exit(f"clean_tracking directory not found: {clean_dir}\nPass --clean-tracking-dir explicitly.")

    sys.path.insert(0, str(Path(__file__).parent))
    from kalman_tracker_iou import KalmanTrackerIoU

    tracker_name = f"kalman_{args.source}"
    out_dir = bench_dir / "results" / tracker_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = sorted(
        [d for d in clean_dir.iterdir() if d.is_dir() and d.name not in EXCLUDED],
        key=lambda d: d.name,
    )

    print(f"Source       : {args.source}")
    print(f"Tracker      : KalmanIoU  max_age={args.max_age}  min_hits={args.min_hits}  iou={args.iou_thresh}")
    print(f"Sequences    : {len(sequences)}")
    print(f"Output dir   : {out_dir}")
    print()

    for seq_dir in sequences:
        out_file = out_dir / f"{seq_dir.name}.txt"
        if out_file.exists() and not args.force:
            print(f"  SKIP {seq_dir.name} — output exists. Use --force to re-run.")
            continue

        labels_dir = seq_dir / "labels"
        n_frames = len(list(labels_dir.glob("*.txt")))

        if args.source == "yolo":
            det_csv = bench_dir / "results" / "detections" / f"{seq_dir.name}.csv"
            if not det_csv.exists():
                print(f"  SKIP {seq_dir.name} — detection CSV not found: {det_csv}")
                continue
            frames = load_yolo_detections(det_csv)
        else:
            frames = load_gt_detections(labels_dir)

        tracker = KalmanTrackerIoU(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_thresh=args.iou_thresh,
        )
        rows = run_tracker(frames, n_frames, tracker)
        out_file.write_text("\n".join(rows) + ("\n" if rows else ""))
        print(f"  {seq_dir.name}  ({n_frames} frames, {len(rows)} track rows)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
