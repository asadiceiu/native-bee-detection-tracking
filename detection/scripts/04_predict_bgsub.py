#!/usr/bin/env python3
"""
Run MOG2 or KNN background subtraction on one fold's val videos and the test set.

Usage:
    python 04_predict_bgsub.py --method MOG2 --fold 0

Outputs (in results/{method_lower}_fold{N}/):
    val_predictions.csv
    test_predictions.csv

CSV columns: base_filename, class, cx, cy, width, height, conf  (pixel coords)

Key design notes:
  - A fresh bg-subtractor instance is created per video to prevent cross-video
    background model contamination.
  - All frames of each video are fed to the bg-subtractor for warmup, but
    detections are only recorded for frames present in clean_yolo/images/.
  - For very short videos (e.g. 6 frames) the warmup is insufficient; this
    is expected and noted in the final report.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ── Detection parameters (same as background_subtraction_baseline.py) ──────
MIN_CONTOUR_AREA = 20
MAX_CONTOUR_AREA = 2000
MORPH_KERNEL_SIZE = (3, 3)
GAUSS_BLUR_KERNEL = (5, 5)


def create_subtractor(method: str):
    if method == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    elif method == "KNN":
        return cv2.createBackgroundSubtractorKNN(detectShadows=True)
    else:
        raise ValueError(f"Unknown method: {method}. Choose MOG2 or KNN.")


def preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    return cv2.GaussianBlur(gray, GAUSS_BLUR_KERNEL, 0)


def postprocess_mask(fg_mask: np.ndarray) -> np.ndarray:
    fg_mask = np.where(fg_mask == 127, 0, fg_mask)   # remove shadows
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    return fg_mask


def extract_detections(fg_mask: np.ndarray, frame_num: int, video_name: str) -> list[dict]:
    detections = []
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w / 2
        cy = y + h / 2
        aspect = w / h if h > 0 else 1.0
        aspect_score = max(0.1, min(1.0, 1.0 - abs(aspect - 1.2) / 2.0))
        area_score = max(0.1, min(1.0, area / 500.0))
        conf = round(max(0.1, min(0.9, (aspect_score + area_score) / 2.0)), 3)
        detections.append({
            "base_filename": f"{video_name}_frame_{frame_num:05d}.jpg",
            "class": "bee",
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "width": w,
            "height": h,
            "conf": conf,
        })
    return detections


def process_video(video_path: Path, method: str, annotated_frames: set[str]) -> list[dict]:
    """
    Process one video with a fresh bg-subtractor.
    Records detections only for frames whose filename is in annotated_frames.
    """
    video_name = video_path.stem
    subtractor = create_subtractor(method)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Warning: cannot open {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = []
    frame_num = 0

    for _ in tqdm(range(total), desc=f"  {video_name}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        processed = preprocess(frame)
        fg_mask = subtractor.apply(processed)

        fname = f"{video_name}_frame_{frame_num:05d}.jpg"
        if fname in annotated_frames:
            clean_mask = postprocess_mask(fg_mask)
            detections.extend(extract_detections(clean_mask, frame_num, video_name))

    cap.release()
    print(f"    {video_name}: {len(detections)} detections in annotated frames")
    return detections


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["base_filename", "class", "cx", "cy", "width", "height", "conf"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} detections → {path}")


def main():
    parser = argparse.ArgumentParser(description="Background subtraction predictions for one fold")
    parser.add_argument("--method", required=True, choices=["MOG2", "KNN"])
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Dir containing .avi files (default: auto-detected from folds.json)",
    )
    parser.add_argument("--output-dir", default=None, help="Output dir for CSVs")
    args = parser.parse_args()

    bench_dir = Path(__file__).resolve().parent.parent
    folds_json = bench_dir / "configs" / "folds.json"

    if not folds_json.exists():
        sys.exit(f"folds.json not found: {folds_json}\nRun 01_make_folds.py first.")

    with open(folds_json) as f:
        folds_cfg = json.load(f)

    # ── Paths ──────────────────────────────────────────────────────────────
    clean_yolo_dir = Path(folds_cfg["clean_yolo_dir"])
    images_dir = clean_yolo_dir / "images"

    video_dir = Path(args.video_dir).resolve() if args.video_dir else bench_dir / "data" / "sample-videos"

    method_lower = args.method.lower()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else bench_dir / "results" / f"{method_lower}_fold{args.fold}"
    )

    # ── Determine which videos belong to val fold and test set ─────────────
    val_videos = folds_cfg["fold_val_videos"][str(args.fold)]
    test_videos = folds_cfg["test_videos"]

    # ── Build set of annotated frame filenames (clean_yolo scope) ─────────
    all_clean_frames = set(os.listdir(images_dir))

    def get_annotated_set(video_names: list[str]) -> set[str]:
        result = set()
        for vid in video_names:
            for fname in all_clean_frames:
                if fname.startswith(vid + "_frame_"):
                    result.add(fname)
        return result

    val_annotated = get_annotated_set(val_videos)
    test_annotated = get_annotated_set(test_videos)

    # ── Process videos ─────────────────────────────────────────────────────
    print(f"\nMethod: {args.method}  Fold: {args.fold}")

    print(f"\nVal videos ({len(val_videos)}):")
    val_rows = []
    for vid in val_videos:
        vpath = video_dir / f"{vid}.avi"
        if not vpath.exists():
            print(f"  Warning: video not found: {vpath}")
            continue
        val_rows.extend(process_video(vpath, args.method, val_annotated))
    write_csv(val_rows, output_dir / "val_predictions.csv")

    print(f"\nTest videos ({len(test_videos)}):")
    test_rows = []
    for vid in test_videos:
        vpath = video_dir / f"{vid}.avi"
        if not vpath.exists():
            print(f"  Warning: video not found: {vpath}")
            continue
        test_rows.extend(process_video(vpath, args.method, test_annotated))
    write_csv(test_rows, output_dir / "test_predictions.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
