#!/usr/bin/env python3
"""
Generate 5-fold CV splits + fixed test set for the detection benchmark.

Split design:
  - Test set : 3 videos (614 frames), held out permanently
  - CV pool  : 13 videos (2,663 frames), 5-fold split for train/val
  - Each fold trains on ~12 videos, validates on ~1-4 videos

Run once before any training:
    python 01_make_folds.py --clean-yolo-dir ../../detection/clean_yolo
"""

import argparse
import json
import os
from pathlib import Path

# ── Fixed test set (never used during training or val) ─────────────────────
TEST_VIDEOS = [
    "Camera_43_20240420_153908",   # 586 frames, April, 29 bees, challenging
    "Camera_43_20240619_143401",   # 19 frames,  June,  38 bees, very dense
    "Camera_43_20240619_150759",   # 9 frames,   June,  15 bees, dense
]

# ── 5-fold CV pool: val video assignment per fold ──────────────────────────
FOLD_VAL_VIDEOS = {
    0: ["Camera_43_20240411_113937"],                   # 716 frames
    1: ["Camera_43_20240505_130943"],                   # 605 frames
    2: [                                                # 535 frames
        "Camera_43_20240423_124240",
        "Camera_43_20240423_103802",
        "Camera_43_20240502_140512",
    ],
    3: [                                                # 543 frames
        "Camera_43_20240531_133832",
        "Camera_43_20240531_143838",
        "Camera_43_20240412_130142",
        "Camera_43_20240411_130947",
    ],
    4: [                                                # 264 frames
        "Camera_43_20240412_140149",
        "Camera_43_20240505_170358",
        "Camera_43_20240502_150518",
        "Camera_43_20240611_143758",
    ],
}


def get_video_name(filename: str) -> str:
    """Extract video name from image filename."""
    return filename.rsplit("_frame_", 1)[0]


def main():
    _default_data = Path(__file__).resolve().parent.parent.parent.parent / "data" / "detection" / "clean_yolo"
    parser = argparse.ArgumentParser(description="Generate fold splits for detection benchmark")
    parser.add_argument(
        "--clean-yolo-dir",
        default=str(_default_data),
        help=f"Path to clean_yolo/ dir (default: {_default_data})",
    )
    args = parser.parse_args()

    clean_yolo_dir = Path(args.clean_yolo_dir).resolve()
    images_dir = clean_yolo_dir / "images"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    # ── Collect all image filenames, grouped by video ──────────────────────
    all_images = sorted(f for f in os.listdir(images_dir) if f.endswith(".jpg"))
    print(f"Found {len(all_images)} images in {images_dir}")

    video_to_images: dict[str, list[str]] = {}
    for fname in all_images:
        vid = get_video_name(fname)
        video_to_images.setdefault(vid, []).append(fname)

    all_videos = set(video_to_images.keys())
    all_val_videos = {v for vids in FOLD_VAL_VIDEOS.values() for v in vids}
    cv_pool_videos = all_videos - set(TEST_VIDEOS)

    # Sanity checks
    test_missing = set(TEST_VIDEOS) - all_videos
    if test_missing:
        raise ValueError(f"Test videos not found in images dir: {test_missing}")

    val_missing = all_val_videos - all_videos
    if val_missing:
        raise ValueError(f"Fold val videos not found in images dir: {val_missing}")

    unassigned = cv_pool_videos - all_val_videos
    if unassigned:
        raise ValueError(
            f"CV-pool videos not assigned to any fold: {unassigned}\n"
            "Update FOLD_VAL_VIDEOS to include all non-test videos."
        )

    # ── Output paths ───────────────────────────────────────────────────────
    bench_dir = Path(__file__).resolve().parent.parent
    configs_dir = bench_dir / "configs"
    fold_lists_dir = bench_dir / "fold_lists"
    configs_dir.mkdir(exist_ok=True)
    fold_lists_dir.mkdir(exist_ok=True)

    # ── Write folds.json ───────────────────────────────────────────────────
    folds_config = {
        "test_videos": TEST_VIDEOS,
        "fold_val_videos": {str(k): v for k, v in FOLD_VAL_VIDEOS.items()},
        "clean_yolo_dir": str(clean_yolo_dir),
    }
    folds_json_path = configs_dir / "folds.json"
    with open(folds_json_path, "w") as f:
        json.dump(folds_config, f, indent=2)
    print(f"Written: {folds_json_path}")

    # ── Write test.txt ─────────────────────────────────────────────────────
    test_images = []
    for vid in TEST_VIDEOS:
        test_images.extend(
            str(images_dir / fname) for fname in sorted(video_to_images.get(vid, []))
        )
    test_txt = fold_lists_dir / "test.txt"
    with open(test_txt, "w") as f:
        f.write("\n".join(test_images) + "\n")
    print(f"Written: {test_txt}  ({len(test_images)} test images)")

    # ── Write per-fold train/val .txt and data.yaml ────────────────────────
    for fold_idx, val_vids in FOLD_VAL_VIDEOS.items():
        val_images = []
        for vid in val_vids:
            val_images.extend(
                str(images_dir / fname) for fname in sorted(video_to_images.get(vid, []))
            )

        train_images = []
        for vid in sorted(cv_pool_videos - set(val_vids)):
            train_images.extend(
                str(images_dir / fname) for fname in sorted(video_to_images.get(vid, []))
            )

        val_txt = fold_lists_dir / f"fold_{fold_idx}_val.txt"
        train_txt = fold_lists_dir / f"fold_{fold_idx}_train.txt"
        with open(val_txt, "w") as f:
            f.write("\n".join(val_images) + "\n")
        with open(train_txt, "w") as f:
            f.write("\n".join(train_images) + "\n")

        # YOLO data YAML — use text-file lists; YOLO maps /images/ -> /labels/
        yaml_content = (
            f"path: {clean_yolo_dir}\n"
            f"train: {train_txt}\n"
            f"val: {val_txt}\n"
            f"nc: 1\n"
            f"names: ['bee']\n"
        )
        yaml_path = fold_lists_dir / f"fold_{fold_idx}_data.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print(
            f"Fold {fold_idx}: {len(train_images)} train, "
            f"{len(val_images)} val  — {yaml_path.name}"
        )

    print("\nDone. All fold lists and configs written.")
    print(f"  configs/folds.json")
    print(f"  fold_lists/test.txt  ({len(test_images)} images)")
    for i in range(5):
        print(f"  fold_lists/fold_{i}_{{train,val}}.txt + fold_{i}_data.yaml")


if __name__ == "__main__":
    main()
