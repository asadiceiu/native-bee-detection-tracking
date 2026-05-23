#!/usr/bin/env python3
"""
Compute COCO-style metrics for one result directory (val and test splits).

Usage:
    python 05_evaluate.py --result-dir ../results/yolov8s_fold2 --fold 2 --model-name yolov8s

Reads:
    {result_dir}/val_predictions.csv
    {result_dir}/test_predictions.csv

Writes:
    {result_dir}/val_metrics.json
    {result_dir}/test_metrics.json
    {result_dir}/val_pr_curve.png
    {result_dir}/test_pr_curve.png

Ground truth is built directly from clean_yolo/labels/ YOLO .txt files
(not from a CSV), so there is no coordinate-format ambiguity.
Image size is fixed at 1280x720 (all frames in the dataset).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    sys.exit("pycocotools not installed. Run: pip install pycocotools")


IMG_W, IMG_H = 1280, 720
CATEGORY_ID = 1


def load_image_list(txt_path: Path) -> list[str]:
    """Return sorted list of base filenames (e.g. Camera_43_..._frame_00001.jpg)."""
    with open(txt_path) as f:
        return sorted(Path(line.strip()).name for line in f if line.strip())


def build_coco_gt(image_names: list[str], labels_dir: Path) -> dict:
    """
    Build a COCO-format ground-truth dict from YOLO .txt label files.
    YOLO format per line: class cx cy w h  (all normalised to [0,1])
    COCO bbox format     : [x_tl, y_tl, w_px, h_px]  (pixel, top-left)
    """
    images, annotations = [], []
    ann_id = 1

    for img_id, fname in enumerate(image_names, start=1):
        images.append({
            "id": img_id,
            "file_name": fname,
            "width": IMG_W,
            "height": IMG_H,
        })

        label_fname = fname.replace(".jpg", ".txt").replace(".jpeg", ".txt")
        label_path = labels_dir / label_fname
        if not label_path.exists():
            continue

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx_n, cy_n, w_n, h_n = parts
                cx_px = float(cx_n) * IMG_W
                cy_px = float(cy_n) * IMG_H
                w_px = float(w_n) * IMG_W
                h_px = float(h_n) * IMG_H
                x_tl = cx_px - w_px / 2
                y_tl = cy_px - h_px / 2
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CATEGORY_ID,
                    "bbox": [x_tl, y_tl, w_px, h_px],
                    "area": w_px * h_px,
                    "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CATEGORY_ID, "name": "bee", "supercategory": "insect"}],
    }


def build_coco_preds(predictions_csv: Path, image_name_to_id: dict[str, int]) -> list[dict]:
    """
    Build COCO results list from predictions CSV.
    CSV columns: base_filename, class, cx, cy, width, height, conf  (pixel coords)
    """
    results = []
    if not predictions_csv.exists():
        return results

    with open(predictions_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["base_filename"]
            if fname not in image_name_to_id:
                continue
            cx = float(row["cx"])
            cy = float(row["cy"])
            w = float(row["width"])
            h = float(row["height"])
            x_tl = cx - w / 2
            y_tl = cy - h / 2
            results.append({
                "image_id": image_name_to_id[fname],
                "category_id": CATEGORY_ID,
                "bbox": [x_tl, y_tl, w, h],
                "score": float(row["conf"]),
            })
    return results


def run_coco_eval(gt_dict: dict, pred_list: list[dict]):
    """Run COCOeval and return (stats_array, coco_eval_obj)."""
    import tempfile, json as _json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        _json.dump(gt_dict, tf)
        gt_path = tf.name

    coco_gt = COCO(gt_path)

    if not pred_list:
        # No predictions: return zeros
        class _FakeCOCOeval:
            stats = [-1.0] * 12
            eval = {"precision": np.full((10, 101, 1, 4, 3), -1.0)}
            params = type("P", (), {"recThrs": np.linspace(0, 1, 101)})()
        return _FakeCOCOeval()

    import tempfile as _tf2
    with _tf2.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as pf:
        _json.dump(pred_list, pf)
        pred_path = pf.name

    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def save_pr_curve(coco_eval, model_name: str, split: str, output_path: Path):
    try:
        # precision shape: [T, R, K, A, M] → index [0]=IoU=0.5, [:,0]=cat 0, [3]=all area, [2]=maxDets=100
        precision = coco_eval.eval["precision"][0, :, 0, 0, 2]
        recall = coco_eval.params.recThrs
        valid = precision > -1
        precision, recall = precision[valid], recall[valid]
        ap50 = float(coco_eval.stats[1]) if hasattr(coco_eval, "stats") else -1.0

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, "b-", linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve — {model_name} ({split})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.text(0.05, 0.92, f"AP@0.5 = {ap50:.3f}", transform=ax.transAxes,
                fontsize=11, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8))
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  Warning: could not save PR curve: {e}")


def evaluate_split(
    split: str,
    image_names: list[str],
    labels_dir: Path,
    predictions_csv: Path,
    result_dir: Path,
    model_name: str,
    fold: int,
):
    print(f"\n--- Evaluating {split} split ({len(image_names)} images) ---")

    gt_dict = build_coco_gt(image_names, labels_dir)
    n_annotations = len(gt_dict["annotations"])
    image_name_to_id = {img["file_name"]: img["id"] for img in gt_dict["images"]}

    pred_list = build_coco_preds(predictions_csv, image_name_to_id)
    print(f"  GT annotations : {n_annotations}")
    print(f"  Predictions    : {len(pred_list)}")

    coco_eval = run_coco_eval(gt_dict, pred_list)
    stats = coco_eval.stats

    metrics = {
        "model": model_name,
        "fold": fold,
        "split": split,
        "n_frames": len(image_names),
        "n_annotations": n_annotations,
        "n_predictions": len(pred_list),
        "AP": round(float(stats[0]), 6),
        "AP_50": round(float(stats[1]), 6),
        "AP_75": round(float(stats[2]), 6),
        "AP_small": round(float(stats[3]), 6),
        "AP_medium": round(float(stats[4]), 6),
        "AP_large": round(float(stats[5]), 6),
        "AR_1": round(float(stats[6]), 6),
        "AR_10": round(float(stats[7]), 6),
        "AR_100": round(float(stats[8]), 6),
        "AR_small": round(float(stats[9]), 6),
        "AR_medium": round(float(stats[10]), 6),
        "AR_large": round(float(stats[11]), 6),
    }

    metrics_path = result_dir / f"{split}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    pr_path = result_dir / f"{split}_pr_curve.png"
    save_pr_curve(coco_eval, model_name, split, pr_path)
    print(f"  Saved: {pr_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="COCO evaluation for one result directory")
    parser.add_argument("--result-dir", required=True, help="Path to results/{model}_fold{N}/")
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--model-name", required=True, help="e.g. yolov8s or mog2")
    parser.add_argument("--labels-dir", default=None, help="Path to clean_yolo/labels/ (auto-detected)")
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    bench_dir = Path(__file__).resolve().parent.parent
    folds_json = bench_dir / "configs" / "folds.json"

    if not folds_json.exists():
        sys.exit(f"folds.json not found. Run 01_make_folds.py first.")

    with open(folds_json) as f:
        folds_cfg = json.load(f)

    clean_yolo_dir = Path(folds_cfg["clean_yolo_dir"])
    labels_dir = Path(args.labels_dir).resolve() if args.labels_dir else clean_yolo_dir / "labels"

    if not labels_dir.exists():
        sys.exit(f"Labels dir not found: {labels_dir}")

    fold_lists_dir = bench_dir / "fold_lists"
    val_txt = fold_lists_dir / f"fold_{args.fold}_val.txt"
    test_txt = fold_lists_dir / "test.txt"

    for p in [val_txt, test_txt]:
        if not p.exists():
            sys.exit(f"Image list not found: {p}\nRun 01_make_folds.py first.")

    val_images = load_image_list(val_txt)
    test_images = load_image_list(test_txt)

    evaluate_split(
        split="val",
        image_names=val_images,
        labels_dir=labels_dir,
        predictions_csv=result_dir / "val_predictions.csv",
        result_dir=result_dir,
        model_name=args.model_name,
        fold=args.fold,
    )

    evaluate_split(
        split="test",
        image_names=test_images,
        labels_dir=labels_dir,
        predictions_csv=result_dir / "test_predictions.csv",
        result_dir=result_dir,
        model_name=args.model_name,
        fold=args.fold,
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
