#!/usr/bin/env python3
"""
Visualize IoU Kalman tracking on a sequence using GT oracle detections.

Reads images from clean_tracking/{seq}/images/ and GT labels from
clean_tracking/{seq}/labels/, runs the IoU Kalman tracker frame-by-frame,
then writes an annotated video.

Usage:
    python visualize_tracking.py Camera_43_20240411_113937
    python visualize_tracking.py Camera_43_20240411_113937 --output /tmp/tracked.mp4
    python visualize_tracking.py Camera_43_20240411_113937 --fps 10 --trail 30
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG_W, IMG_H = 1280, 720

BOX_THICKNESS   = 3       # track bounding box line width
TRAIL_THICKNESS = 4       # base thickness for the most-recent trail segment
TRAIL_ALPHA     = 0.85    # how opaque the trail overlay is
FILL_ALPHA      = 0.18    # semi-transparent fill inside the track box
FONT_SCALE      = 0.65
LABEL_THICKNESS = 2

# Bright, distinct palette — one colour per track ID (cycles)
PALETTE = [
    (  0, 220, 255), (  0, 255, 100), (255,  60,  60), (255, 200,   0),
    (180,   0, 255), (255, 120,   0), ( 60, 180, 255), (255,  60, 200),
    (100, 255,  60), (255, 255,   0), (  0, 160, 255), (220,  60, 255),
]

def color_for(track_id: int):
    return PALETTE[(track_id - 1) % len(PALETTE)]


def parse_frame_id(filename: str) -> int:
    return int(Path(filename).stem.split("_frame_")[-1])


def load_gt_frames(labels_dir: Path) -> dict[int, np.ndarray]:
    """Return {seq_frame_id: (N,4) xyxy pixel boxes}."""
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
            w_px, h_px = w_n * IMG_W, h_n * IMG_H
            x1 = cx_n * IMG_W - w_px / 2
            y1 = cy_n * IMG_H - h_px / 2
            boxes.append([x1, y1, x1 + w_px, y1 + h_px])
        if boxes:
            frames[fid] = np.array(boxes, dtype=np.float32)
    return frames


def draw_box(frame: np.ndarray, x1, y1, x2, y2, color, label: str):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # Semi-transparent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, FILL_ALPHA, frame, 1 - FILL_ALPHA, 0, frame)
    # Solid border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    # Label badge
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LABEL_THICKNESS)
    badge_y0 = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, badge_y0), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), LABEL_THICKNESS, cv2.LINE_AA)


def draw_trail(frame: np.ndarray, centers: list[tuple], color, max_trail: int):
    if len(centers) < 2:
        return
    pts = centers[-max_trail:]
    overlay = frame.copy()
    n = len(pts)
    for i in range(1, n):
        frac = i / n                        # 0 → oldest, 1 → newest
        thickness = max(1, int(TRAIL_THICKNESS * frac))
        cv2.line(overlay, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, TRAIL_ALPHA, frame, 1 - TRAIL_ALPHA, 0, frame)


def main():
    script_dir = Path(__file__).resolve().parent
    bench_dir = script_dir.parent

    parser = argparse.ArgumentParser(description="Visualize IoU Kalman + GT oracle on a sequence")
    parser.add_argument("sequence", help="Sequence name, e.g. Camera_43_20240411_113937")
    parser.add_argument("--clean-tracking-dir", default=None)
    parser.add_argument("--output", default=None, help="Output video path (default: results/videos/{seq}.mp4)")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS (default: 10)")
    parser.add_argument("--trail", type=int, default=25, help="Trail length in frames (default: 25)")
    parser.add_argument("--max-age", type=int, default=10)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    args = parser.parse_args()

    # Locate clean_tracking directory
    if args.clean_tracking_dir:
        clean_dir = Path(args.clean_tracking_dir).resolve()
    else:
        clean_dir = bench_dir.parent.parent / "data" / "tracking" / "clean_tracking"
        if not clean_dir.exists():
            clean_dir = bench_dir.parent / "clean_tracking"
    if not clean_dir.exists():
        sys.exit(f"clean_tracking not found: {clean_dir}")

    seq_dir = clean_dir / args.sequence
    if not seq_dir.exists():
        sys.exit(f"Sequence not found: {seq_dir}")

    labels_dir = seq_dir / "labels"
    images_dir = seq_dir / "images"
    if not images_dir.exists():
        sys.exit(f"images/ directory not found under {seq_dir}")

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = bench_dir / "results" / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.sequence}.mp4"

    # Load GT and images
    gt_frames = load_gt_frames(labels_dir)
    image_files = sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: parse_frame_id(p.name),
    )
    if not image_files:
        sys.exit(f"No images found in {images_dir}")

    n_frames = len(image_files)
    print(f"Sequence : {args.sequence}")
    print(f"Frames   : {n_frames}")
    print(f"GT frames with annotations: {len(gt_frames)}")
    print(f"Output   : {out_path}")
    print()

    # Import tracker (sibling module)
    sys.path.insert(0, str(script_dir))
    from kalman_tracker_iou import KalmanTrackerIoU

    tracker = KalmanTrackerIoU(max_age=args.max_age, min_hits=1, iou_thresh=args.iou_thresh)

    # Video writer — try avc1 (H.264 on macOS/AVFoundation) then MJPG fallback
    for codec, suffix in [("avc1", ".mp4"), ("MJPG", ".avi")]:
        if out_path.suffix.lower() not in {".mp4", ".avi"}:
            out_path = out_path.with_suffix(suffix)
        candidate = out_path.with_suffix(suffix)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(candidate), fourcc, args.fps, (IMG_W, IMG_H))
        if writer.isOpened():
            out_path = candidate
            break
        writer.release()
    else:
        sys.exit("Failed to open VideoWriter — no working codec found (tried avc1/MJPG)")

    # Track centres for trail drawing {track_id: [(cx, cy), ...]}
    trails: dict[int, list] = defaultdict(list)

    for seq_idx, img_path in enumerate(image_files):
        fid = seq_idx + 1
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  Warning: could not read {img_path.name}, skipping")
            continue

        gt_boxes = gt_frames.get(fid, np.empty((0, 4), dtype=np.float32))
        active = tracker.update(gt_boxes)   # [(track_id, [x1,y1,x2,y2]), ...]

        # GT boxes — thin white outline only (shows oracle input)
        for box in gt_boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), 1)

        # Collect centres, draw trails, then draw boxes on top
        for tid, box in active:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            trails[tid].append((cx, cy))

        for tid, box in active:
            draw_trail(frame, trails[tid], color_for(tid), args.trail)

        for tid, box in active:
            draw_box(frame, *box, color_for(tid), f"T{tid}")

        # HUD
        cv2.putText(frame, f"Frame {fid}/{n_frames}  Tracks: {len(active)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame {fid}/{n_frames}  Tracks: {len(active)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)

        writer.write(frame)

        if fid % 50 == 0 or fid == n_frames:
            print(f"  {fid}/{n_frames}  active tracks: {len(active)}")

    writer.release()
    print(f"\nDone — video written to: {out_path}")


if __name__ == "__main__":
    main()
