# Stingless Bee Detection and Tracking Benchmark

Code for the benchmarks in:

> **[Monitoring Native Eusocial and Semi-social Bees at Hive and Nest Entrances to Quantify Foraging and Guarding Behaviour]**  
> [Asaduz Zaman, Malika Ratnayake, Vanessa Kellermann, Alan Dorin]  
> *International Journal of Computer Vision (IJCV), 2026*

We provide a full pipeline for evaluating stingless bee **detection** and **multi-object tracking** in hive-entrance video, including training scripts, evaluation code, pre-computed results, and a tracking visualisation tool.

---

## Repository Structure

```
.
├── data/               # Dataset (images + annotations)
│   ├── detection/
│   │   └── clean_yolo/     # 3,277 frames, YOLO labels (single-class: bee)
│   └── tracking/
│       └── clean_tracking/ # 16 sequences, YOLO labels with bee identity IDs
│
├── detection/          # Detection benchmark (5-fold CV + fixed test set)
│   ├── scripts/        # Training, inference, evaluation, aggregation
│   ├── slurm/          # HPC array job scripts
│   └── configs/        # Fold split definitions (folds.json)
│
├── tracking/           # Tracking benchmark (13 sequences, 3,511 frames)
│   └── scripts/        # GT preparation, detection, tracking, evaluation, visualisation
│
├── results/            # Pre-computed summary tables
│   ├── detection_results.md
│   └── tracking_results.md
│
└── requirements.txt
```

---

## Dataset

The dataset consists of 3,277 annotated frames from 16 Camera_43 video clips (1,280 × 720 px, 25 fps) recorded at a stingless bee hive entrance.  
Annotations include per-frame bounding boxes and unique bee identities in YOLO format.

The dataset is included in this release under `data/`:

```
data/
├── detection/
│   └── clean_yolo/
│       ├── images/    # 3,277 *.jpg frames
│       └── labels/    # *.txt YOLO labels  (class cx cy w h, normalised)
└── tracking/
    └── clean_tracking/
        └── {sequence_name}/
            ├── images/    # *.jpg frames
            └── labels/    # *.txt YOLO labels with bee_N class IDs
```

---

## Installation

```bash
# Unzip the release archive, then:
cd stingless-bee-benchmark
pip install -r requirements.txt
```

Python 3.10+ recommended. GPU (CUDA or Apple MPS) recommended for YOLO training and inference; CPU works for everything else.

---

## Detection Benchmark

See [`detection/README.md`](detection/README.md) for full instructions.

**Quick start (local smoke test — fold 0, yolov8n, 2 epochs):**

```bash
cd detection
bash scripts/run_local_test.sh
```

**Full benchmark (HPC SLURM):**

```bash
# Submit background-subtraction baselines (CPU, fast)
sbatch slurm/bgsub_array.slurm

# Submit YOLO training + inference (45 GPU tasks: 9 models × 5 folds)
sbatch slurm/train_array.slurm

# After all jobs complete — evaluate and aggregate
bash scripts/run_full_evaluation.sh
```

**Results** (Test AP₅₀, mean ± std across 5 folds):

| Model    | Test AP₅₀          | Test AP        | Test AR₁₀₀    |
|----------|--------------------|---------------|---------------|
| YOLO11-m | **0.919 ± 0.010**  | **0.516 ± 0.005** | **0.596 ± 0.006** |
| YOLO11-n | 0.918 ± 0.014      | 0.509 ± 0.016 | 0.590 ± 0.013 |
| YOLOv8-n | 0.916 ± 0.024      | 0.511 ± 0.023 | 0.590 ± 0.019 |
| YOLOv8-s | 0.912 ± 0.013      | 0.507 ± 0.019 | 0.587 ± 0.013 |
| YOLOv8-m | 0.912 ± 0.015      | 0.492 ± 0.016 | 0.576 ± 0.016 |
| YOLO11-s | 0.910 ± 0.017      | 0.505 ± 0.016 | 0.586 ± 0.012 |
| YOLO26-s | 0.910 ± 0.026      | 0.512 ± 0.015 | 0.588 ± 0.019 |
| YOLO26-n | 0.908 ± 0.020      | 0.506 ± 0.018 | 0.589 ± 0.012 |
| YOLO26-m | 0.905 ± 0.022      | 0.507 ± 0.023 | 0.585 ± 0.017 |
| MOG2     | 0.171 ± 0.000      | 0.039 ± 0.000 | 0.089 ± 0.000 |
| KNN      | 0.108 ± 0.003      | 0.023 ± 0.000 | 0.060 ± 0.001 |

Full table: [`results/detection_results.md`](results/detection_results.md)

---

## Tracking Benchmark

See [`tracking/README.md`](tracking/README.md) for full instructions.

**Quick start (GT oracle, all trackers):**

```bash
cd tracking
python scripts/01_prepare_gt.py
python scripts/03_run_kalman.py --source gt
python scripts/04_run_ultralytics.py --tracker bytetrack --source gt
python scripts/04_run_ultralytics.py --tracker botsort   --source gt
python scripts/05_evaluate.py
python scripts/06_aggregate.py
```

**End-to-end with YOLO detections:**

```bash
python scripts/02_run_detections.py --weights /path/to/best.pt --device 0
python scripts/03_run_kalman.py --source yolo
python scripts/04_run_ultralytics.py --tracker bytetrack --source yolo --weights /path/to/best.pt
python scripts/04_run_ultralytics.py --tracker botsort   --source yolo --weights /path/to/best.pt
python scripts/05_evaluate.py
python scripts/06_aggregate.py
```

**Visualise tracking on a sequence:**

```bash
python scripts/visualize_tracking.py Camera_43_20240411_113937 --fps 15 --trail 40
```

**Results** (mean ± std, 13 sequences, IoU threshold 0.5):

*YOLO detections (end-to-end, backbone: YOLO11-n fold 0):*

| Tracker      | Precision      | Recall         | F1             | MOTA           | IDSW | IDF1           |
|--------------|---------------|---------------|---------------|---------------|------|---------------|
| Kalman (IoU) | 0.849 ± 0.111 | **0.992 ± 0.011** | 0.911 ± 0.066 | 0.753 ± 0.195 | 281  | 0.759 ± 0.153 |
| ByteTrack    | 0.892 ± 0.102 | 0.814 ± 0.185 | 0.839 ± 0.131 | 0.700 ± 0.231 | **64**   | **0.779 ± 0.175** |
| BoT-SORT     | **0.895 ± 0.101** | 0.813 ± 0.184 | **0.841 ± 0.131** | 0.700 ± 0.231 | 79   | 0.721 ± 0.186 |

*GT oracle (tracker quality in isolation):*

| Tracker      | Precision      | Recall         | F1             | MOTA           | IDSW | IDF1           |
|--------------|---------------|---------------|---------------|---------------|------|---------------|
| Kalman (IoU) | 0.998 ± 0.004 | 0.998 ± 0.004 | **0.998 ± 0.004** | **0.958 ± 0.077** | 275  | 0.875 ± 0.127 |
| ByteTrack    | 1.000 ± 0.000 | 0.929 ± 0.094 | 0.960 ± 0.057 | 0.928 ± 0.096 | **4**    | **0.926 ± 0.087** |
| BoT-SORT     | 1.000 ± 0.000 | 0.929 ± 0.094 | 0.960 ± 0.057 | 0.928 ± 0.096 | **4**    | 0.926 ± 0.087 |

Full table: [`results/tracking_results.md`](results/tracking_results.md)

---

## Citation

```bibtex
@article{[citekey],
  title   = {[Paper title]},
  author  = {[Authors]},
  journal = {International Journal of Computer Vision},
  year    = {2025},
}
```

---

## License

Code: MIT License. See `LICENSE`.  
Dataset: See dataset-specific terms upon release.
