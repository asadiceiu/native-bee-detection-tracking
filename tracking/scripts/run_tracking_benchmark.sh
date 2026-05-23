#!/usr/bin/env bash
# ── Tracking Benchmark Runner ─────────────────────────────────────────────────
# Runs all steps sequentially. Already-finished steps are skipped automatically.
# Safe to interrupt (Ctrl+C) and re-run — resumes from where it left off.
#
# Usage:
#   bash scripts/run_tracking_benchmark.sh --weights /path/to/best.pt
#   bash scripts/run_tracking_benchmark.sh --weights /path/to/best.pt --device 0
#   bash scripts/run_tracking_benchmark.sh --weights /path/to/best.pt --device cpu
#
# Steps:
#   0  Prepare GT (MOT format from clean_tracking labels)
#   1  Run YOLO detections on all 13 sequences
#   2  Run Kalman tracker (YOLO + GT oracle)
#   3  Run ByteTrack (YOLO + GT oracle)
#   4  Run BoT-SORT (YOLO + GT oracle)
#   5  Evaluate all tracker outputs
#   6  Aggregate results → summary table + bar chart
#
# Assumptions:
#   - Virtual environment is already activated.
#   - Required packages: ultralytics, supervision, filterpy, scipy, matplotlib
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"

WEIGHTS=""
DEVICE="0"
CONF="0.25"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights) WEIGHTS="$2"; shift 2 ;;
    --device)  DEVICE="$2";  shift 2 ;;
    --conf)    CONF="$2";    shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [ -z "$WEIGHTS" ]; then
  echo "ERROR: --weights is required"
  echo "Usage: bash $0 --weights /path/to/best.pt [--device 0] [--conf 0.25]"
  exit 1
fi

echo "====================================================================="
echo "  Tracking Benchmark"
echo "====================================================================="
echo "  bench_dir : $BENCH_DIR"
echo "  weights   : $WEIGHTS"
echo "  device    : $DEVICE"
echo "  conf      : $CONF"
echo "  Started   : $(date)"
echo "====================================================================="
echo ""

# ── Step 0: Prepare GT ───────────────────────────────────────────────────────
echo ">>> [0/6] Preparing MOT-format GT files..."
python "$SCRIPT_DIR/01_prepare_gt.py"
echo ""

# ── Step 1: YOLO detections ──────────────────────────────────────────────────
echo ">>> [1/6] Running YOLO detections on 13 sequences..."
python "$SCRIPT_DIR/02_run_detections.py" \
  --weights "$WEIGHTS" \
  --conf    "$CONF" \
  --device  "$DEVICE" \
|| { echo "ERROR: YOLO detections failed"; exit 1; }
echo ""

# ── Step 2: Kalman tracker ───────────────────────────────────────────────────
echo ">>> [2/6] Running Kalman tracker (YOLO + GT oracle)..."
python "$SCRIPT_DIR/03_run_kalman.py" --source yolo \
|| echo "  WARNING: Kalman (YOLO) failed — continuing"
python "$SCRIPT_DIR/03_run_kalman.py" --source gt \
|| echo "  WARNING: Kalman (GT oracle) failed — continuing"
echo ""

# ── Step 3: ByteTrack ────────────────────────────────────────────────────────
echo ">>> [3/6] Running ByteTrack (YOLO + GT oracle)..."
python "$SCRIPT_DIR/04_run_ultralytics.py" \
  --tracker bytetrack --source yolo --weights "$WEIGHTS" --conf "$CONF" --device "$DEVICE" \
|| echo "  WARNING: ByteTrack (YOLO) failed — continuing"
python "$SCRIPT_DIR/04_run_ultralytics.py" \
  --tracker bytetrack --source gt \
|| echo "  WARNING: ByteTrack (GT oracle) failed — continuing"
echo ""

# ── Step 4: BoT-SORT ─────────────────────────────────────────────────────────
echo ">>> [4/6] Running BoT-SORT (YOLO + GT oracle)..."
python "$SCRIPT_DIR/04_run_ultralytics.py" \
  --tracker botsort --source yolo --weights "$WEIGHTS" --conf "$CONF" --device "$DEVICE" \
|| echo "  WARNING: BoT-SORT (YOLO) failed — continuing"
python "$SCRIPT_DIR/04_run_ultralytics.py" \
  --tracker botsort --source gt \
|| echo "  WARNING: BoT-SORT (GT oracle) failed — continuing"
echo ""

# ── Step 5: Evaluation ───────────────────────────────────────────────────────
echo ">>> [5/6] Evaluating all tracker outputs..."
python "$SCRIPT_DIR/05_evaluate.py" \
|| { echo "ERROR: evaluation failed"; exit 1; }
echo ""

# ── Step 6: Aggregate ────────────────────────────────────────────────────────
echo ">>> [6/6] Aggregating results..."
python "$SCRIPT_DIR/06_aggregate.py"

echo ""
echo "====================================================================="
echo "  All done!  Finished at: $(date)"
echo "  Results: $BENCH_DIR/results/aggregate/tracking_summary.md"
echo "====================================================================="
