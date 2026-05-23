#!/usr/bin/env bash
# ── Local smoke test ────────────────────────────────────────────────────────
# Runs the full pipeline on fold 0 with yolov8n (2 epochs) to verify that
# all scripts work correctly before submitting HPC jobs.
#
# Usage (from the detection_benchmark/ root):
#   source ../../venv-yolo/bin/activate
#   bash scripts/run_local_test.sh
#
# Expected runtime: 3–10 min on CPU (or ~1 min with a GPU).
# ───────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
PROJ_ROOT="$(cd "$BENCH_DIR/../.." && pwd)"
CLEAN_YOLO="$PROJ_ROOT/detection/clean_yolo"

echo "====================================================="
echo "  Detection Benchmark — Local Smoke Test"
echo "====================================================="
echo "  bench_dir  : $BENCH_DIR"
echo "  proj_root  : $PROJ_ROOT"
echo "  clean_yolo : $CLEAN_YOLO"
echo ""

cd "$SCRIPT_DIR"

# ── Step 1: Generate fold lists ───────────────────────────────────────────
echo ">>> Step 1/7: Generate fold lists"
python 01_make_folds.py --clean-yolo-dir "$CLEAN_YOLO"
echo ""

# ── Step 2: Train yolov8n on fold 0 (2 epochs, CPU) ──────────────────────
echo ">>> Step 2/7: Train yolov8n (fold 0, 2 epochs)"
python 02_train_yolo.py \
  --model yolov8n \
  --fold 0 \
  --epochs 2 \
  --batch 4 \
  --device mps \
  --output-dir "$BENCH_DIR/results"
echo ""

# ── Step 3: Predict yolov8n on fold 0 val + test ─────────────────────────
echo ">>> Step 3/7: Predict yolov8n (fold 0)"
python 03_predict_yolo.py \
  --weights "$BENCH_DIR/results/yolov8n_fold0/weights/best.pt" \
  --fold 0 \
  --output-dir "$BENCH_DIR/results/yolov8n_fold0" \
  --conf 0.01 \
  --device mps
echo ""

# ── Step 4: MOG2 predictions on fold 0 val + test ────────────────────────
echo ">>> Step 4/7: MOG2 predictions (fold 0)"
python 04_predict_bgsub.py \
  --method MOG2 \
  --fold 0 \
  --video-dir "$PROJ_ROOT/detection/sample-videos" \
  --output-dir "$BENCH_DIR/results/mog2_fold0"
echo ""

# ── Step 5: KNN predictions on fold 0 val + test ─────────────────────────
echo ">>> Step 5/7: KNN predictions (fold 0)"
python 04_predict_bgsub.py \
  --method KNN \
  --fold 0 \
  --video-dir "$PROJ_ROOT/detection/sample-videos" \
  --output-dir "$BENCH_DIR/results/knn_fold0"
echo ""

# ── Step 6: Evaluate all three result dirs ────────────────────────────────
echo ">>> Step 6/7: Evaluate all fold-0 results"
for MODEL_DIR in yolov8n_fold0 mog2_fold0 knn_fold0; do
  # Derive model name from directory (strip _fold0 suffix)
  MODEL_NAME="${MODEL_DIR%_fold0}"
  echo "  Evaluating $MODEL_DIR ..."
  python 05_evaluate.py \
    --result-dir "$BENCH_DIR/results/$MODEL_DIR" \
    --fold 0 \
    --model-name "$MODEL_NAME"
done
echo ""

# ── Step 7: Aggregate (partial — fold 0 only) ─────────────────────────────
echo ">>> Step 7/7: Aggregate results"
python 06_aggregate_results.py
echo ""

echo "====================================================="
echo "  Smoke test complete!"
echo "====================================================="
echo ""
echo "Verification checklist:"
echo "  [ ] fold_lists/ has test.txt + 10 fold .txt files + 5 .yaml files"
echo "  [ ] results/yolov8n_fold0/weights/best.pt exists"
echo "  [ ] results/yolov8n_fold0/{val,test}_predictions.csv non-empty"
echo "  [ ] results/yolov8n_fold0/{val,test}_metrics.json has AP_50 in [0,1]"
echo "  [ ] results/mog2_fold0/{val,test}_predictions.csv non-empty"
echo "  [ ] results/aggregate/summary_table.md renders a valid table"
echo ""
echo "If all checks pass, you are ready to submit jobs on the server."
echo "  sbatch jobs/01_bgsub.job"
echo "  sbatch jobs/02_train.job"
echo ""
echo "See SERVER_GUIDE.md for full server setup instructions."
