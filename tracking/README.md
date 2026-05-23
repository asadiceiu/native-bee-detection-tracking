# Tracking Benchmark

Evaluates three multi-object trackers across 13 stingless bee sequences (3,511 frames) in two modes: end-to-end with YOLO detections, and GT oracle (ground-truth boxes fed directly to isolate tracker quality from detector quality).

---

## Experimental Design

### Sequences

13 sequences from `clean_tracking/` are used (3 short clips < 20 frames are excluded):

| Sequence | Frames |
|----------|--------|
| Camera_43_20240411_113937 | 716 |
| Camera_43_20240411_130947 | 100 |
| Camera_43_20240412_130142 | 112 |
| Camera_43_20240412_140149 | 101 |
| Camera_43_20240420_153908 | 656 |
| Camera_43_20240423_103802 | 222 |
| Camera_43_20240423_124240 | 246 |
| Camera_43_20240502_140512 | 97 |
| Camera_43_20240502_150518 | 62 |
| Camera_43_20240505_130943 | 766 |
| Camera_43_20240505_170358 | 95 |
| Camera_43_20240531_133832 | 206 |
| Camera_43_20240531_143838 | 132 |
| **Total** | **3,511** |

### Trackers

| Tracker | Implementation | Association |
|---------|---------------|-------------|
| **Kalman (IoU)** | Custom SORT-style (8-D constant-velocity state) | Hungarian on 1 − IoU, threshold 0.3 |
| **ByteTrack** | Ultralytics `model.track()` / `supervision.ByteTrack` | — |
| **BoT-SORT** | Ultralytics `model.track()` / `supervision.ByteTrack`* | — |

*`supervision` 0.28 does not expose a standalone BoT-SORT class; the GT oracle mode falls back to ByteTrack (appearance re-ID is absent in both cases for this dataset).

### Evaluation Modes

| Mode | Detections fed to tracker | Purpose |
|------|--------------------------|---------|
| **YOLO** | YOLO11-n fold 0 predictions (conf ≥ 0.25) | End-to-end performance |
| **Oracle** | Ground-truth bounding boxes | Isolates tracker quality from detector quality |

### Metrics

Computed per sequence, reported as mean ± std across 13 sequences:

| Metric | Definition |
|--------|-----------|
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1** | Harmonic mean of Precision and Recall |
| **MOTA** | 1 − (FN + FP + IDSW) / GT |
| **IDSW** | Total identity switches (integer, no std) |
| **IDF1** | 2 · IDP · IDR / (IDP + IDR) |

Detection matching uses IoU threshold 0.5.

---

## Directory Layout

```
tracking/
└── scripts/
    ├── kalman_tracker_iou.py      # Standalone SORT-style Kalman tracker (IoU)
    ├── 01_prepare_gt.py           # Convert YOLO labels → MOT Challenge GT format
    ├── 02_run_detections.py       # Run YOLO detector → per-sequence detection CSVs
    ├── 03_run_kalman.py           # Run Kalman tracker (YOLO or GT mode)
    ├── 04_run_ultralytics.py      # Run ByteTrack / BoT-SORT (YOLO or GT mode)
    ├── 05_evaluate.py             # Compute CLEAR + IDF1 metrics
    ├── 06_aggregate.py            # Summary table + bar chart
    ├── visualize_tracking.py      # Write annotated tracking video
    └── run_tracking_benchmark.sh  # End-to-end runner
```

### Generated outputs (not in repo)

```
tracking/
├── gt/                            # Created by 01_prepare_gt.py
│   └── {seq}/
│       ├── gt/gt.txt              # MOT Challenge format
│       └── seqinfo.ini
└── results/
    ├── detections/                # Created by 02_run_detections.py
    │   └── {seq}.csv
    ├── kalman_yolo/               # MOT-format tracker outputs
    ├── kalman_gt/
    ├── bytetrack_yolo/
    ├── bytetrack_gt/
    ├── botsort_yolo/
    ├── botsort_gt/
    └── aggregate/
        ├── tracking_summary.md
        ├── tracking_summary.csv
        └── tracking_bar_chart.png
```

---

## Step-by-Step Usage

### Step 1 — Prepare ground truth (run once)

```bash
python scripts/01_prepare_gt.py \
    --clean-tracking-dir /path/to/data/tracking/clean_tracking
```

Converts YOLO-format labels to MOT Challenge `gt.txt` files under `gt/`.

### Step 2 — Run YOLO detections (YOLO mode only)

```bash
python scripts/02_run_detections.py \
    --weights /path/to/best.pt \
    --device  0               # 0=GPU, mps=Apple Silicon, cpu=CPU
    --conf    0.25
```

Writes `results/detections/{seq}.csv` for all 13 sequences.

### Step 3 — Run trackers

**Kalman (IoU):**

```bash
python scripts/03_run_kalman.py --source yolo   # YOLO mode
python scripts/03_run_kalman.py --source gt     # Oracle mode
```

**ByteTrack and BoT-SORT:**

```bash
# YOLO mode (runs model.track() — weights required)
python scripts/04_run_ultralytics.py --tracker bytetrack --source yolo --weights /path/to/best.pt
python scripts/04_run_ultralytics.py --tracker botsort   --source yolo --weights /path/to/best.pt

# Oracle mode (no weights needed)
python scripts/04_run_ultralytics.py --tracker bytetrack --source gt
python scripts/04_run_ultralytics.py --tracker botsort   --source gt
```

### Step 4 — Evaluate

```bash
python scripts/05_evaluate.py
```

Reads all `results/{tracker}/*.txt` files, evaluates against `gt/`, writes `metrics.json` per tracker.

### Step 5 — Aggregate

```bash
python scripts/06_aggregate.py
```

Writes `results/aggregate/tracking_summary.md` and `tracking_bar_chart.png`.

### Run everything at once

```bash
bash scripts/run_tracking_benchmark.sh \
    --weights /path/to/best.pt \
    --device  0
```

---

## Visualise Tracking

Write an annotated MP4 for any sequence using the IoU Kalman tracker in GT oracle mode:

```bash
python scripts/visualize_tracking.py Camera_43_20240411_113937 \
    --clean-tracking-dir /path/to/data/tracking/clean_tracking \
    --fps   15 \
    --trail 40 \
    --output /tmp/tracked.mp4
```

Each track gets a distinct colour with a fading position trail. Thin white boxes show the raw GT oracle input. Output defaults to `results/videos/{seq}.mp4`.

---

## Kalman Tracker Details

`kalman_tracker_iou.py` is a self-contained SORT-style tracker requiring only `numpy`, `scipy`, and `filterpy`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_age` | 10 | Frames a track survives without a detection match |
| `min_hits` | 1 | Detections required before a track is reported |
| `iou_thresh` | 0.3 | Minimum IoU to accept a detection–track assignment |

**State vector**: `[cx, cy, w, h, vx, vy, vw, vh]` (constant-velocity model).  
**Association**: Hungarian algorithm on `1 − IoU` cost matrix.

Usage as a library:

```python
from kalman_tracker_iou import KalmanTrackerIoU
import numpy as np

tracker = KalmanTrackerIoU(max_age=10, min_hits=1, iou_thresh=0.3)

for frame_boxes in sequence:          # frame_boxes: (N, 4) array of [x1, y1, x2, y2]
    active = tracker.update(frame_boxes)
    for track_id, box in active:
        print(track_id, box)           # box: [x1, y1, x2, y2]
```

---

## Notes

- Frame IDs are normalised to 1-indexed sequential integers (sorted by filename) to handle sequences that start at an arbitrary absolute video frame number.
- The GT format uses `bee_N` class names from the YOLO labels as track IDs; `bee_0` → ID 1, etc.
- ByteTrack and BoT-SORT GT oracle results are identical in this benchmark because `supervision` 0.28 does not include a standalone BoT-SORT implementation (no appearance re-ID).
