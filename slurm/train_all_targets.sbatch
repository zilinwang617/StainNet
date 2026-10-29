#!/bin/bash
#SBATCH --job-name=stainnet-all
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:30:00
# --- Cluster-specific: set these for your own cluster ---
# Partition: the submit_*.sh scripts pass PARTITION=<gpu_partition>; when calling
#   sbatch directly, add -p <gpu_partition>. The A100 feature name above may
#   also differ on your cluster.
# SCRATCH_ROOT: node-local scratch used to stage data (below).
# -o/-e are passed by submit_all_targets.sh (date subfolder).

set -eo pipefail

# Project checkout; defaults to the directory sbatch was called from.
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
RUNS_DIR="${RUNS_DIR:-$PROJECT_DIR/runs}"
cd "$PROJECT_DIR"

if command -v module >/dev/null 2>&1; then
    module load "${CONDA_MODULE:-anaconda}"
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-stainnet}"

# Copy data to fast node-local scratch for better I/O performance
# Set SCRATCH_ROOT to your cluster's node-local scratch (default: $TMPDIR or /tmp).
SCRATCH_ROOT="${SCRATCH_ROOT:-${TMPDIR:-/tmp}}"
SCRATCH="$SCRATCH_ROOT/stainnet_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"
cp -rL "$DATA_DIR" "$SCRATCH/data"

trap 'rm -rf "$SCRATCH"' EXIT

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"

RUN_GROUP="${RUN_GROUP:-all_targets}"
RUN_NAME="${TARGET:-Akt}_${SLURM_JOB_ID}"
RUN_DIR="$RUNS_DIR/$RUN_GROUP/$RUN_NAME"

echo "Group:   $RUN_GROUP"
echo "Run:     $RUN_NAME"

# --- Train ---
TRAIN_START=$(date +%s)
python src/train.py \
    --manifest      "$SCRATCH/data/splits/stacked_raw_split.json" \
    --stats         "$SCRATCH/data/stats/stacked_raw_stats_train.json" \
    --batch-size    16 \
    --epochs        50 \
    --lr            1e-3 \
    --lr-patience   5 \
    --num-workers   4 \
    --base-channels 64 \
    --output-dir    "$RUNS_DIR/$RUN_GROUP" \
    --target        "${TARGET:-Akt}" \
    --run-name      "$RUN_NAME" \
    --no-wandb
TRAIN_END=$(date +%s)
echo "Training time: $(( TRAIN_END - TRAIN_START ))s"

# --- Eval ---
echo "Running test-set evaluation..."
EVAL_START=$(date +%s)
python src/evaluate.py \
    --checkpoint  "$RUN_DIR/checkpoints/best.pt" \
    --manifest    "$DATA_DIR/splits/stacked_raw_split.json" \
    --stats       "$DATA_DIR/stats/stacked_raw_stats_train.json" \
    --output-dir  "$RUN_DIR/eval/test" \
    --split       test \
    --batch-size  16 \
    --device      cuda
EVAL_END=$(date +%s)
echo "Eval time: $(( EVAL_END - EVAL_START ))s"
