#!/usr/bin/env python3
"""
Convert clean_tracking YOLO labels → MOT Challenge format for trackeval.

Usage:
    python 01_prepare_gt.py
    python 01_prepare_gt.py --clean-tracking-dir ../../clean_tracking

Output per sequence:
    gt/{seq}/gt/gt.txt       MOT Challenge CSV
    gt/{seq}/seqinfo.ini     Required by trackeval

MOT format: frame_id, track_id, left, top, width, height, conf, class, visibility
    - frame_id: 1-indexed (parsed from *_frame_NNNNN.txt)
    - track_id: integer from bee_N (bee_0 → 1, bee_1 → 2, ... ; 1-indexed for MOT)
    - left, top, w, h: pixels (image 1280×720, converted from YOLO normalized)
    - conf=1, class=1, visibility=1
"""

import argparse
import configparser
from pathlib import Path

IMG_W = 1280
IMG_H = 720

EXCLUDED = {
    "Camera_43_20240611_143758",   # 6 frames
    "Camera_43_20240619_143401",   # 19 frames
    "Camera_43_20240619_150759",   # 9 frames
}


def parse_frame_id(filename: str) -> int:
    """Extract 1-indexed frame ID from label filename *_frame_NNNNN.txt."""
    stem = Path(filename).stem
    return int(stem.split("_frame_")[-1])


def parse_bee_id(token: str) -> int:
    """Convert bee_N → N+1 (1-indexed MOT track ID)."""
    return int(token.split("_")[-1]) + 1


def convert_sequence(seq_dir: Path, gt_out_dir: Path) -> int:
    """Convert one sequence; returns total annotation count."""
    labels_dir = seq_dir / "labels"
    label_files = sorted(labels_dir.glob("*.txt"), key=lambda p: parse_frame_id(p.name))

    rows = []
    for seq_idx, lf in enumerate(label_files):
        frame_id = seq_idx + 1   # 1-indexed sequential (normalizes absolute frame offsets)
        for line in lf.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            bee_id = parse_bee_id(parts[0])
            cx_n, cy_n, w_n, h_n = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            w_px = w_n * IMG_W
            h_px = h_n * IMG_H
            left = cx_n * IMG_W - w_px / 2
            top = cy_n * IMG_H - h_px / 2
            rows.append(f"{frame_id},{bee_id},{left:.2f},{top:.2f},{w_px:.2f},{h_px:.2f},1,1,1")

    gt_dir = gt_out_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "gt.txt").write_text("\n".join(rows) + ("\n" if rows else ""))

    n_frames = len(label_files)
    cfg = configparser.ConfigParser()
    cfg["Sequence"] = {
        "name": seq_dir.name,
        "imWidth": str(IMG_W),
        "imHeight": str(IMG_H),
        "frameRate": "25",
        "seqLength": str(n_frames),
    }
    with open(gt_out_dir / "seqinfo.ini", "w") as f:
        cfg.write(f)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare MOT-format GT from clean_tracking labels")
    parser.add_argument(
        "--clean-tracking-dir",
        default=None,
        help="Path to clean_tracking directory (default: auto-detect relative to this script)",
    )
    args = parser.parse_args()

    bench_dir = Path(__file__).resolve().parent.parent
    if args.clean_tracking_dir:
        clean_dir = Path(args.clean_tracking_dir).resolve()
    else:
        clean_dir = bench_dir.parent.parent / "data" / "tracking" / "clean_tracking"
        if not clean_dir.exists():
            clean_dir = bench_dir.parent / "clean_tracking"

    if not clean_dir.exists():
        raise SystemExit(f"clean_tracking directory not found: {clean_dir}\nPass --clean-tracking-dir explicitly.")

    gt_base = bench_dir / "gt"
    sequences = sorted(
        [d for d in clean_dir.iterdir() if d.is_dir() and d.name not in EXCLUDED],
        key=lambda d: d.name,
    )

    print(f"clean_tracking dir : {clean_dir}")
    print(f"GT output dir      : {gt_base}")
    print(f"Sequences          : {len(sequences)}")
    print()
    print(f"{'Sequence':<45} {'Frames':>7} {'Annotations':>12}")
    print("-" * 65)

    total_frames = 0
    total_annot = 0
    for seq_dir in sequences:
        labels_dir = seq_dir / "labels"
        n_frames = len(list(labels_dir.glob("*.txt")))
        n_annot = convert_sequence(seq_dir, gt_base / seq_dir.name)
        total_frames += n_frames
        total_annot += n_annot
        print(f"  {seq_dir.name:<43} {n_frames:>7} {n_annot:>12}")

    print("-" * 65)
    print(f"  {'TOTAL':<43} {total_frames:>7} {total_annot:>12}")
    print()
    print("Done. GT files written to:", gt_base)


if __name__ == "__main__":
    main()
