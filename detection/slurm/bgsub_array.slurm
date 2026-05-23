#!/bin/bash
#SBATCH --job-name=bee_bgsub
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bgsub_%A_%a.out
#SBATCH --error=logs/bgsub_%A_%a.err
#SBATCH --partition=cpu

# ── Task ID → method × fold mapping ────────────────────────────────────────
# task_id = method_idx * 5 + fold_idx
#   0=MOG2  1=KNN
# Total tasks: 2 × 5 = 10  (array 0-9)
# ──────────────────────────────────────────────────────────────────────────

METHODS=(MOG2 KNN)

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
METHOD=${METHODS[$METHOD_IDX]}
METHOD_LOWER=$(echo "$METHOD" | tr '[:upper:]' '[:lower:]')

echo "======================================================"
echo "  Task $SLURM_ARRAY_TASK_ID : method=$METHOD  fold=$FOLD"
echo "======================================================"

# ── Paths — UPDATE these for your cluster ─────────────────────────────────
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ_ROOT="$(cd "$BENCH_DIR/../.." && pwd)"

echo "BENCH_DIR  : $BENCH_DIR"
echo "PROJ_ROOT  : $PROJ_ROOT"

# ── Environment ────────────────────────────────────────────────────────────
# module load python/3.12
source "$PROJ_ROOT/venv-yolo/bin/activate"

mkdir -p "$BENCH_DIR/slurm/logs"

# ── Background subtraction predictions ────────────────────────────────────
echo ""
echo ">>> Running $METHOD on fold $FOLD (val + test)..."
python "$BENCH_DIR/scripts/04_predict_bgsub.py" \
  --method "$METHOD" \
  --fold "$FOLD" \
  --video-dir "$PROJ_ROOT/detection/sample-videos" \
  --output-dir "$BENCH_DIR/results/${METHOD_LOWER}_fold${FOLD}"

echo ""
echo ">>> Task $SLURM_ARRAY_TASK_ID complete: $METHOD fold $FOLD"
