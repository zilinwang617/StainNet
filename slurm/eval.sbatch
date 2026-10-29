#!/bin/bash
#SBATCH --job-name=stainnet-eval
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.err
# --- Cluster-specific: set the partition for your own cluster ---
# The submit_*.sh scripts pass PARTITION=<cpu_partition>; when calling sbatch
#   directly, add -p <cpu_partition>.

set -eo pipefail

# Project checkout; defaults to the directory sbatch was called from.
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
RUNS_DIR="${RUNS_DIR:-$PROJECT_DIR/runs}"
cd "$PROJECT_DIR"

# Load conda and activate environment
if command -v module >/dev/null 2>&1; then
    module load "${CONDA_MODULE:-anaconda}"
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-stainnet}"

# Defaults: evaluate the bc64 no-aug run for TARGET on the test split.
# Override via:
#   sbatch --export=ALL,TARGET=YAP,RUN_NAME=YAP_bc64_aug,SPLIT=val slurm/eval.sbatch
TARGET="${TARGET:-Akt}"
RUN_NAME="${RUN_NAME:-${TARGET}_bc64}"
SPLIT="${SPLIT:-test}"

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Target:   $TARGET"
echo "Run:      $RUN_NAME"
echo "Split:    $SPLIT"

python src/evaluate.py \
    --checkpoint  "$RUNS_DIR/$RUN_NAME/checkpoints/best.pt" \
    --manifest    "$DATA_DIR/splits/stacked_raw_split.json" \
    --stats       "$DATA_DIR/stats/stacked_raw_stats_train.json" \
    --split       "$SPLIT" \
    --batch-size  16
