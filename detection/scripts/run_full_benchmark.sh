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
DEVICE="${1:-0}"

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

# ── Step 0: Generate folds (always run to ensure paths match this machine) ────
echo ">>> [0/4] Generating fold splits (regenerates paths for this machine)..."
python "$SCRIPT_DIR/01_make_folds.py"
echo ""

# ── Step 1: Background subtraction (CPU, fast) ───────────────────────────────
echo ">>> [1/4] Background subtraction baselines ($N_BGSUB tasks)"
for METHOD in "${METHODS[@]}"; do
  METHOD_LOWER=$(echo "$METHOD" | tr '[:upper:]' '[:lower:]')
  for (( FOLD=0; FOLD<N_FOLDS; FOLD++ )); do
    OUTPUT="$BENCH_DIR/results/${METHOD_LOWER}_fold${FOLD}/val_predictions.csv"
    if [ -f "$OUTPUT" ]; then
      echo "  SKIP  : $METHOD fold $FOLD"
      continue
    fi
    echo "  Running: $METHOD fold $FOLD  ($(date +%H:%M))"
    python "$SCRIPT_DIR/04_predict_bgsub.py" \
      --method "$METHOD" \
      --fold   "$FOLD" \
      --output-dir "$BENCH_DIR/results/${METHOD_LOWER}_fold${FOLD}" \
    || echo "  WARNING: $METHOD fold $FOLD failed — continuing"
  done
done
echo ""

# ── Step 2: YOLO training + inference ────────────────────────────────────────
DONE=0
echo ">>> [2/4] YOLO training + inference ($N_YOLO tasks)"

for MODEL in "${MODELS[@]}"; do
  for (( FOLD=0; FOLD<N_FOLDS; FOLD++ )); do
    DONE=$(( DONE + 1 ))
    WEIGHTS="$BENCH_DIR/results/${MODEL}_fold${FOLD}/weights/best.pt"

    echo ""
    echo "--- [$DONE/$N_YOLO] $MODEL  fold $FOLD  ($(date +%H:%M:%S)) ---"

    # Training — skips automatically if best.pt already exists
    if ! python "$SCRIPT_DIR/02_train_yolo.py" \
           --model      "$MODEL" \
           --fold       "$FOLD" \
           --epochs     100 \
           --batch      8 \
           --device     "$DEVICE" \
           --output-dir "$BENCH_DIR/results"; then
      echo "  ERROR: training failed for $MODEL fold $FOLD — skipping inference"
      continue
    fi

    # Inference — skips automatically if CSVs already exist
    if [ -f "$WEIGHTS" ]; then
      python "$SCRIPT_DIR/03_predict_yolo.py" \
        --weights    "$WEIGHTS" \
        --fold       "$FOLD" \
        --output-dir "$BENCH_DIR/results/${MODEL}_fold${FOLD}" \
        --conf       0.01 \
        --device     "$DEVICE" \
      || echo "  WARNING: inference failed for $MODEL fold $FOLD"
    else
      echo "  WARNING: $WEIGHTS not found after training"
    fi
  done
done
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
