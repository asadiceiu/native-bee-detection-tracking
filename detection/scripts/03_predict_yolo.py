#!/usr/bin/env python3
"""
Run inference with a trained YOLO model on the val fold and test set.

Usage:
    python 03_predict_yolo.py --weights ../results/yolov8s_fold2/weights/best.pt --fold 2

Outputs (inside the same result dir as the weights):
    val_predictions.csv
    test_predictions.csv

CSV columns: base_filename, class, cx, cy, width, height, conf  (all pixel coords)
"""

import argparse
import csv
import sys
from pathlib import Path


def run_inference(model, image_paths: list[Path], output_csv: Path, conf_thresh: float):
    """Run model on image_paths and write predictions CSV."""
    rows = []
    for img_path in image_paths:
        results = model(str(img_path), conf=conf_thresh, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()   # [x1, y1, x2, y2] pixels
            conf = float(box.conf[0].cpu())
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            rows.append({
                "base_filename": img_path.name,
                "class": "bee",
                "cx": round(cx, 2),
                "cy": round(cy, 2),
                "width": round(w, 2),
                "height": round(h, 2),
                "conf": round(conf, 4),
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["base_filename", "class", "cx", "cy", "width", "height", "conf"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} detections → {output_csv}")


def load_image_list(txt_path: Path) -> list[Path]:
    with open(txt_path) as f:
        return [Path(line.strip()) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="YOLO inference for val fold and test set")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--conf", type=float, default=0.01, help="Confidence threshold (keep low for post-hoc analysis)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Device override (e.g. cpu, 0)")
    parser.add_argument("--output-dir", default=None, help="Dir to write CSVs (default: same dir as weights)")
    parser.add_argument("--force", action="store_true", help="Re-run inference even if CSVs already exist")
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}")

    bench_dir = Path(__file__).resolve().parent.parent
    fold_lists_dir = bench_dir / "fold_lists"

    val_txt = fold_lists_dir / f"fold_{args.fold}_val.txt"
    test_txt = fold_lists_dir / "test.txt"

    for p in [val_txt, test_txt]:
        if not p.exists():
            sys.exit(f"Image list not found: {p}\nRun 01_make_folds.py first.")

    val_images = load_image_list(val_txt)
    test_images = load_image_list(test_txt)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else weights.parent.parent

    val_csv = output_dir / "val_predictions.csv"
    test_csv = output_dir / "test_predictions.csv"
    if val_csv.exists() and test_csv.exists() and not args.force:
        print(f"SKIP: predictions already exist in {output_dir}. Use --force to re-run.")
        sys.exit(0)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics --upgrade")

    model = YOLO(str(weights))
    if args.device:
        model.to(args.device)

    print(f"Loaded weights: {weights}")
    print(f"Val images : {len(val_images)}")
    print(f"Test images: {len(test_images)}")

    print("Running inference on val fold...")
    run_inference(model, val_images, output_dir / "val_predictions.csv", args.conf)

    print("Running inference on test set...")
    run_inference(model, test_images, output_dir / "test_predictions.csv", args.conf)

    print("Done.")


if __name__ == "__main__":
    main()
