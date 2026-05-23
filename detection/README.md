# Detection Benchmark

Evaluates YOLO detectors and background-subtraction baselines for stingless bee detection using 5-fold cross-validation on a training pool and a fixed held-out test set.

---

## Experimental Design

| Split | Videos | Frames | Purpose |
|-------|--------|--------|---------|
| **Test set** (fixed) | 3 | 614 | Never seen during training or model selection |
| **CV pool** | 13 | 2,663 | 5-fold split — each fold: one val group + all others as train |

The fixed test set spans April and June recordings (different bee density and lighting) to ensure temporal generalisability. Val folds are used only for early stopping.

### Models

| Group | Models |
|-------|--------|
| YOLO detectors | `yolov8n`, `yolov8s`, `yolov8m`, `yolo11n`, `yolo11s`, `yolo11m`, `yolo26n`, `yolo26s`, `yolo26m` |
| Background subtraction | MOG2, KNN |

All YOLO models are trained for 100 epochs (patience 50), input size 640, batch 8.  
Each model is trained independently on each of the 5 folds → 5 weight files per model.  
Final metrics: mean ± std across 5 fold models evaluated on the **same fixed test set**.

### Primary Metric

**Test AP₅₀** (PASCAL VOC style, IoU threshold 0.5) — appropriate for small bee objects where strict localisation (AP @ 0.75) is less meaningful than detection presence.

---

## Directory Layout

```
detection/
├── scripts/
│   ├── 01_make_folds.py          # Generate fold splits + YOLO data YAMLs
│   ├── 02_train_yolo.py          # Train one YOLO model on one fold
│   ├── 03_predict_yolo.py        # Run inference → val + test prediction CSVs
│   ├── 04_predict_bgsub.py       # MOG2/KNN → val + test prediction CSVs
│   ├── 05_evaluate.py            # COCO metrics → val_metrics.json + test_metrics.json
│   ├── 06_aggregate_results.py   # mean ± std summary table + bar chart
│   ├── run_full_benchmark.sh     # End-to-end runner (local or HPC interactive)
│   ├── run_full_evaluation.sh    # Evaluation-only runner (after training is done)
│   └── run_local_test.sh         # Smoke test: fold 0, yolov8n, 2 epochs (~5 min)
├── slurm/
│   ├── train_array.slurm         # 45-task array: 9 models × 5 folds
│   └── bgsub_array.slurm         # 10-task array: 2 methods × 5 folds
└── configs/
    └── folds.json                # Authoritative split: test videos + fold assignments
```

### Generated outputs (not in repo)

```
detection/
├── fold_lists/                   # Created by 01_make_folds.py
│   ├── test.txt
│   ├── fold_{0..4}_{train,val}.txt
│   └── fold_{0..4}_data.yaml
└── results/
    ├── {model}_fold{N}/
    │   ├── weights/best.pt
    │   ├── val_predictions.csv
    │   ├── test_predictions.csv
    │   ├── val_metrics.json
    │   └── test_metrics.json
    └── aggregate/
        ├── summary_table.md
        ├── summary_table.csv
        └── comparison_bar_chart.png
```

---

## Step-by-Step Usage

### Step 0 — Generate fold splits (run once)

```bash
python scripts/01_make_folds.py --clean-yolo-dir /path/to/data/detection/clean_yolo
```

This creates `fold_lists/` with absolute paths for this machine. Re-run whenever you move the data.

### Step 1 — Train YOLO (one fold at a time)

```bash
python scripts/02_train_yolo.py \
    --model yolov8n \
    --fold  0 \
    --epochs 100 \
    --batch  8 \
    --device 0          # 0=GPU, mps=Apple Silicon, cpu=CPU
```

Weights saved to `results/yolov8n_fold0/weights/best.pt`.  
Already-finished folds are skipped automatically (checks for `best.pt`).

### Step 2 — Run inference

```bash
python scripts/03_predict_yolo.py \
    --weights results/yolov8n_fold0/weights/best.pt \
    --fold    0 \
    --conf    0.01      # low threshold; evaluator re-thresholds
```

```bash
python scripts/04_predict_bgsub.py --method MOG2 --fold 0
python scripts/04_predict_bgsub.py --method KNN  --fold 0
```

### Step 3 — Evaluate

```bash
python scripts/05_evaluate.py \
    --result-dir results/yolov8n_fold0 \
    --fold       0 \
    --model-name yolov8n
```

Writes `val_metrics.json` and `test_metrics.json` in the result directory.

### Step 4 — Aggregate

```bash
python scripts/06_aggregate_results.py
```

Writes `results/aggregate/summary_table.md` and `comparison_bar_chart.png`.

---

## Running on HPC (SLURM)

```bash
# Step 0 — generate folds on the login node
python scripts/01_make_folds.py --clean-yolo-dir /path/to/clean_yolo

# Step 1+2 — train + infer in parallel (45 GPU jobs)
sbatch slurm/train_array.slurm

# Baselines in parallel (10 CPU jobs)
sbatch slurm/bgsub_array.slurm

# Step 3+4 — after all jobs finish
bash scripts/run_full_evaluation.sh
```

Edit `slurm/train_array.slurm` to set your partition name, module loads, and data paths before submitting.

---

## Local Smoke Test

Runs fold 0, `yolov8n`, 2 epochs only. Expected runtime: 3–8 min on CPU.

```bash
bash scripts/run_local_test.sh
```

Verifies the full pipeline (train → infer → evaluate → aggregate) without committing GPU time.

---

## Notes

- `AP_large = −1` for all models (no large bees in the dataset) — omit from tables.
- Fold 4 has fewer val frames (264 vs ~500+) due to the size distribution of available videos — acknowledged as a dataset constraint.
- MOG2/KNN performance is unreliable for videos shorter than ~30 frames (insufficient warmup frames).
- Prediction CSVs use a low confidence threshold (`--conf 0.01`) so the evaluator can apply any threshold post-hoc via COCO API.
