#!/usr/bin/env bash
# ── Full benchmark runner — interactive HPC session ─────────────────────────
# Runs all 55 tasks (9 models × 5 folds + 2 baselines × 5 folds) sequentially.
# Already-finished tasks are skipped automatically.
# Safe to interrupt (Ctrl+C) and re-run — resumes from where it left off.
#
# Usage:
#   bash scripts/run_full_benchmark.sh           # GPU device 0 (default)
#   bash scripts/run_full_benchmark.sh cpu       # CPU
#   bash scripts/run_full_benchmark.sh mps       # Apple Silicon
#
# Assumes: virtual environment is already activated.
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
DEVICE="mps"

MODELS=(yolov8n yolov8s yolov8m yolo11n yolo11s yolo11m yolo26n yolo26s yolo26m)
METHODS=(MOG2 KNN)
N_FOLDS=5
N_YOLO=$(( ${#MODELS[@]} * N_FOLDS ))
N_BGSUB=$(( ${#METHODS[@]} * N_FOLDS ))

echo "====================================================================="
echo "  Detection Benchmark — Full Run"
echo "====================================================================="
echo "  bench_dir  : $BENCH_DIR"
echo "  device     : $DEVICE"
echo "  YOLO tasks : ${#MODELS[@]} models × $N_FOLDS folds = $N_YOLO"
echo "  Bgsub tasks: ${#METHODS[@]} methods × $N_FOLDS folds = $N_BGSUB"
echo "  Started at : $(date)"
echo "====================================================================="
echo ""



# ── Step 3: Evaluation ───────────────────────────────────────────────────────
echo ">>> [3/4] Evaluating all result directories"
for DIR in "$BENCH_DIR/results"/*/; do
  [ -d "$DIR" ] || continue
  [ -f "${DIR}val_predictions.csv" ] || continue

  DIRNAME=$(basename "$DIR")
  MODEL_NAME=$(echo "$DIRNAME" | sed 's/_fold[0-9]*$//')
  FOLD_NUM=$(echo  "$DIRNAME" | sed 's/.*_fold//')

  # Skip non-numeric fold suffix (e.g. aggregate/)
  [[ "$FOLD_NUM" =~ ^[0-9]+$ ]] || continue

  if [ -f "${DIR}val_metrics.json" ] && [ -f "${DIR}test_metrics.json" ]; then
    echo "  SKIP  : $DIRNAME"
    continue
  fi

  echo "  Evaluating: $DIRNAME ..."
  python "$SCRIPT_DIR/05_evaluate.py" \
    --result-dir  "$DIR" \
    --fold        "$FOLD_NUM" \
    --model-name  "$MODEL_NAME" \
  || echo "  WARNING: evaluation failed for $DIRNAME"
done
echo ""

# ── Step 4: Aggregate ────────────────────────────────────────────────────────
echo ">>> [4/4] Aggregating results"
python "$SCRIPT_DIR/06_aggregate_results.py"

echo ""
echo "====================================================================="
echo "  All done!  Finished at: $(date)"
echo "  Results: $BENCH_DIR/results/aggregate/summary_table.md"
echo "====================================================================="
