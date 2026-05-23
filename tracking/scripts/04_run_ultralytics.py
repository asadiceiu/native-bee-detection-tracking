#!/usr/bin/env python3
"""
Run ByteTrack or BoT-SORT on all 13 sequences.

Usage:
    # YOLO detections mode
    python 04_run_ultralytics.py --tracker bytetrack --source yolo --weights /path/to/best.pt
    python 04_run_ultralytics.py --tracker botsort   --source yolo --weights /path/to/best.pt

    # GT oracle mode (no weights needed)
    python 04_run_ultralytics.py --tracker bytetrack --source gt
    python 04_run_ultralytics.py --tracker botsort   --source gt

Output per sequence (MOT format):
    results/{tracker}_{yolo|gt}/{seq}.txt
"""

import argparse
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


def load_gt_boxes_per_frame(labels_dir: Path) -> dict[int, np.ndarray]:
    """Return {seq_frame_id: (N, 4) array of [x1, y1, x2, y2]} from YOLO GT labels.

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


def run_yolo_mode(model, image_dir: Path, tracker_yaml: str, conf: float) -> list[str]:
    """Run model.track() on all images; return MOT-format lines."""
    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: parse_frame_id(p.name),
    )
    rows = []
    for seq_idx, img_path in enumerate(image_files):
        fid = seq_idx + 1   # 1-indexed sequential (normalizes absolute frame offsets)
        results = model.track(
            source=str(img_path),
            tracker=tracker_yaml,
            conf=conf,
            persist=True,
            verbose=False,
        )
        boxes_obj = results[0].boxes
        if boxes_obj is None or boxes_obj.id is None:
            continue
        ids = boxes_obj.id.cpu().numpy().astype(int)
        xyxys = boxes_obj.xyxy.cpu().numpy()
        for track_id, xyxy in zip(ids, xyxys):
            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            rows.append(f"{fid},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1")
    return rows


def run_gt_oracle_mode(labels_dir: Path, n_frames: int, tracker_obj) -> list[str]:
    """Feed GT boxes frame-by-frame to supervision tracker; return MOT-format lines."""
    import supervision as sv

    frames = load_gt_boxes_per_frame(labels_dir)
    rows = []
    for fid in range(1, n_frames + 1):
        boxes = frames.get(fid, np.empty((0, 4)))
        if len(boxes) > 0:
            dets = sv.Detections(
                xyxy=boxes.astype(np.float32),
                confidence=np.ones(len(boxes), dtype=np.float32),
            )
        else:
            dets = sv.Detections.empty()

        tracked = tracker_obj.update_with_detections(dets)

        if tracked.tracker_id is None:
            continue
        for track_id, xyxy in zip(tracked.tracker_id, tracked.xyxy):
            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            rows.append(f"{fid},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1")
    return rows


def make_sv_tracker(tracker_name: str):
    """Instantiate a fresh supervision ByteTrack or BotSort tracker."""
    import supervision as sv
    if tracker_name == "bytetrack":
        return sv.ByteTrack()
    # BotSort was removed/renamed in newer supervision versions
    if hasattr(sv, "BotSort"):
        return sv.BotSort()
    # Fallback: use ByteTrack (equivalent to BoT-SORT without appearance re-ID)
    print("  NOTE: sv.BotSort not available — using sv.ByteTrack as fallback for GT oracle")
    return sv.ByteTrack()


def main():
    parser = argparse.ArgumentParser(description="Run ByteTrack or BoT-SORT on 13 tracking sequences")
    parser.add_argument("--tracker", required=True, choices=["bytetrack", "botsort"])
    parser.add_argument("--source", required=True, choices=["yolo", "gt"])
    parser.add_argument("--weights", default=None, help="Path to YOLO best.pt (required for --source yolo)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO mode")
    parser.add_argument("--device", default=None, help="Device override for YOLO mode (e.g. 0, cpu)")
    parser.add_argument(
        "--clean-tracking-dir",
        default=None,
        help="Path to clean_tracking directory (auto-detected if omitted)",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if output already exists")
    args = parser.parse_args()

    if args.source == "yolo" and args.weights is None:
        sys.exit("--weights is required when --source yolo")

    bench_dir = Path(__file__).resolve().parent.parent
    if args.clean_tracking_dir:
        clean_dir = Path(args.clean_tracking_dir).resolve()
    else:
        clean_dir = bench_dir.parent.parent / "tracking" / "clean_tracking"
        if not clean_dir.exists():
            clean_dir = bench_dir.parent / "clean_tracking"

    if not clean_dir.exists():
        sys.exit(f"clean_tracking directory not found: {clean_dir}\nPass --clean-tracking-dir explicitly.")

    tracker_name = f"{args.tracker}_{args.source}"
    out_dir = bench_dir / "results" / tracker_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = sorted(
        [d for d in clean_dir.iterdir() if d.is_dir() and d.name not in EXCLUDED],
        key=lambda d: d.name,
    )

    print(f"Tracker      : {args.tracker}  source={args.source}")
    print(f"Sequences    : {len(sequences)}")
    print(f"Output dir   : {out_dir}")
    print()

    if args.source == "yolo":
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("ultralytics not installed. Run: pip install ultralytics --upgrade")
        tracker_yaml = f"{args.tracker}.yaml"
        model = YOLO(str(Path(args.weights).resolve()))
        if args.device:
            model.to(args.device)
    else:
        try:
            import supervision
        except ImportError:
            sys.exit("supervision not installed. Run: pip install supervision")

    for seq_dir in sequences:
        out_file = out_dir / f"{seq_dir.name}.txt"
        if out_file.exists() and not args.force:
            print(f"  SKIP {seq_dir.name} — output exists. Use --force to re-run.")
            continue

        labels_dir = seq_dir / "labels"
        n_frames = len(list(labels_dir.glob("*.txt")))
        image_dir = seq_dir / "images"

        print(f"  Running: {seq_dir.name}  ({n_frames} frames)")

        if args.source == "yolo":
            if not image_dir.exists():
                print(f"    SKIP — images/ dir not found")
                continue
            model.predictor = None   # reset persist state between sequences
            rows = run_yolo_mode(model, image_dir, tracker_yaml, args.conf)
        else:
            sv_tracker = make_sv_tracker(args.tracker)
            rows = run_gt_oracle_mode(labels_dir, n_frames, sv_tracker)

        out_file.write_text("\n".join(rows) + ("\n" if rows else ""))
        print(f"    → {len(rows)} track rows")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
