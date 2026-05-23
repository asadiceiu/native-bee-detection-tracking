#!/bin/bash
#SBATCH --job-name=bee_det_train
#SBATCH --array=0-44
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --partition=gpu

# ── Task ID → model × fold mapping ────────────────────────────────────────
# task_id = model_idx * 5 + fold_idx
#
# Models (index 0-8):
#   0=yolov8n  1=yolov8s  2=yolov8m
#   3=yolo11n  4=yolo11s  5=yolo11m
#   6=yolo26n  7=yolo26s  8=yolo26m
# Folds (index 0-4): 0,1,2,3,4
# Total tasks: 9 × 5 = 45  (array 0-44)
# ──────────────────────────────────────────────────────────────────────────

MODELS=(yolov8n yolov8s yolov8m yolo11n yolo11s yolo11m yolo26n yolo26s yolo26m)

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
MODEL=${MODELS[$MODEL_IDX]}

echo "======================================================"
echo "  Task $SLURM_ARRAY_TASK_ID : model=$MODEL  fold=$FOLD"
echo "======================================================"

# ── Paths — UPDATE these for your cluster ─────────────────────────────────
# BENCH_DIR should point to detection/detection_benchmark/ on the server.
# Use an absolute path or set it relative to where you submitted the job.
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ_ROOT="$(cd "$BENCH_DIR/../.." && pwd)"

echo "BENCH_DIR  : $BENCH_DIR"
echo "PROJ_ROOT  : $PROJ_ROOT"

# ── Environment ────────────────────────────────────────────────────────────
# Activate the project virtual environment.
# Uncomment the module load line if your cluster uses environment modules.
# module load cuda/12.1 python/3.12

source "$PROJ_ROOT/venv-yolo/bin/activate"

# Upgrade ultralytics to ensure YOLO26 support
pip install ultralytics --upgrade -q

# Create logs dir if not present
mkdir -p "$BENCH_DIR/slurm/logs"

# ── Training ───────────────────────────────────────────────────────────────
echo ""
echo ">>> Training $MODEL on fold $FOLD ..."
python "$BENCH_DIR/scripts/02_train_yolo.py" \
  --model "$MODEL" \
  --fold "$FOLD" \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --output-dir "$BENCH_DIR/results"

# ── Inference (val + test) ─────────────────────────────────────────────────
WEIGHTS="$BENCH_DIR/results/${MODEL}_fold${FOLD}/weights/best.pt"

if [ ! -f "$WEIGHTS" ]; then
  echo "ERROR: weights not found at $WEIGHTS"
  exit 1
fi

echo ""
echo ">>> Running inference with $WEIGHTS ..."
python "$BENCH_DIR/scripts/03_predict_yolo.py" \
  --weights "$WEIGHTS" \
  --fold "$FOLD" \
  --output-dir "$BENCH_DIR/results/${MODEL}_fold${FOLD}" \
  --conf 0.01 \
  --device 0

echo ""
echo ">>> Task $SLURM_ARRAY_TASK_ID complete: $MODEL fold $FOLD"
