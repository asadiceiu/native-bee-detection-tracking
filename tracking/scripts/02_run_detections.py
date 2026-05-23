#!/usr/bin/env python3
"""
Run YOLO detector on all 13 sequences → per-sequence CSV of detections.

Usage:
    python 02_run_detections.py --weights /path/to/best.pt
    python 02_run_detections.py --weights /path/to/best.pt --conf 0.25 --device 0

Output per sequence:
    results/detections/{seq}.csv
        Columns: frame_id, x1, y1, x2, y2, conf
        frame_id is 1-indexed to match MOT format used by trackeval.
"""

import argparse
import csv
import sys
from pathlib import Path

IMG_W = 1280
IMG_H = 720

EXCLUDED = {
    "Camera_43_20240611_143758",
    "Camera_43_20240619_143401",
    "Camera_43_20240619_150759",
}


def parse_frame_id(filename: str) -> int:
    """Extract 1-indexed frame number from image filename *_frame_NNNNN.{ext}."""
    stem = Path(filename).stem
    return int(stem.split("_frame_")[-1])


def run_sequence(model, image_dir: Path, out_csv: Path, conf_thresh: float):
    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: parse_frame_id(p.name),
    )

    rows = []
    for seq_idx, img_path in enumerate(image_files):
        frame_id = seq_idx + 1   # 1-indexed sequential (normalizes absolute frame offsets)
        results = model(str(img_path), conf=conf_thresh, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu())
            rows.append({
                "frame_id": frame_id,
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
                "conf": round(conf, 4),
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "x1", "y1", "x2", "y2", "conf"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"    {len(image_files)} frames → {len(rows)} detections → {out_csv.name}")


def main():
    parser = argparse.ArgumentParser(description="Run YOLO detector on all 13 tracking sequences")
    parser.add_argument("--weights", required=True, help="Path to YOLO best.pt weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    parser.add_argument("--device", default=None, help="Device override (e.g. 0, cpu)")
    parser.add_argument(
        "--clean-tracking-dir",
        default=None,
        help="Path to clean_tracking directory (auto-detected if omitted)",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if CSV already exists")
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}")

    bench_dir = Path(__file__).resolve().parent.parent
    if args.clean_tracking_dir:
        clean_dir = Path(args.clean_tracking_dir).resolve()
    else:
        clean_dir = bench_dir.parent.parent / "data" / "tracking" / "clean_tracking"
        if not clean_dir.exists():
            clean_dir = bench_dir.parent / "clean_tracking"

    if not clean_dir.exists():
        sys.exit(f"clean_tracking directory not found: {clean_dir}\nPass --clean-tracking-dir explicitly.")

    det_dir = bench_dir / "results" / "detections"
    sequences = sorted(
        [d for d in clean_dir.iterdir() if d.is_dir() and d.name not in EXCLUDED],
        key=lambda d: d.name,
    )

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics --upgrade")

    model = YOLO(str(weights))
    if args.device:
        model.to(args.device)

    print(f"Weights      : {weights}")
    print(f"Sequences    : {len(sequences)}")
    print(f"Conf thresh  : {args.conf}")
    print(f"Output dir   : {det_dir}")
    print()

    for seq_dir in sequences:
        out_csv = det_dir / f"{seq_dir.name}.csv"
        image_dir = seq_dir / "images"

        if not image_dir.exists():
            print(f"  SKIP {seq_dir.name}: images/ dir not found")
            continue

        if out_csv.exists() and not args.force:
            n_frames = len(list(image_dir.iterdir()))
            print(f"  SKIP {seq_dir.name} ({n_frames} frames) — CSV exists. Use --force to re-run.")
            continue

        print(f"  Running: {seq_dir.name}")
        run_sequence(model, image_dir, out_csv, args.conf)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
