#!/usr/bin/env python3
"""
Train one YOLO model on one fold.

Usage:
    python 02_train_yolo.py --model yolov8s --fold 2 --epochs 100 --batch 16 --device 0

The script reads the pre-generated data YAML from fold_lists/ and trains using
Ultralytics YOLO. Best weights land at:
    results/{model}_fold{N}/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

# ── Model name → checkpoint filename ──────────────────────────────────────
MODEL_CHECKPOINTS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26m": "yolo26m.pt",
}


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for one fold")
    parser.add_argument("--model", required=True, choices=list(MODEL_CHECKPOINTS), help="Model name")
    parser.add_argument("--fold", required=True, type=int, choices=range(5), help="Fold index (0-4)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="0", help="Device: 0, 0,1, cpu, mps")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Results base dir (default: ../results relative to this script)",
    )
    parser.add_argument("--force", action="store_true", help="Re-train even if best.pt already exists")
    args = parser.parse_args()

    bench_dir = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).resolve() if args.output_dir else bench_dir / "results"
    data_yaml = bench_dir / "fold_lists" / f"fold_{args.fold}_data.yaml"
    run_name = f"{args.model}_fold{args.fold}"
    best_weights = output_dir / run_name / "weights" / "best.pt"

    if best_weights.exists() and not args.force:
        print(f"SKIP: {best_weights} already exists. Use --force to re-train.")
        sys.exit(0)

    if not data_yaml.exists():
        sys.exit(
            f"Data YAML not found: {data_yaml}\n"
            "Run 01_make_folds.py first."
        )

    checkpoint = MODEL_CHECKPOINTS[args.model]

    print(f"Training {args.model} on fold {args.fold}")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Data YAML  : {data_yaml}")
    print(f"  Output dir : {output_dir / run_name}")
    print(f"  Epochs     : {args.epochs}  Batch: {args.batch}  Imgsz: {args.imgsz}")
    print(f"  Device     : {args.device}")

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics --upgrade")

    model = YOLO(checkpoint)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(output_dir),
        name=run_name,
        exist_ok=True,
        device=args.device,
        patience=50,
        verbose=True,
    )

    best_weights = output_dir / run_name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\nTraining complete. Best weights: {best_weights}")
    else:
        print(f"\nWarning: best.pt not found at expected path {best_weights}")


if __name__ == "__main__":
    main()
