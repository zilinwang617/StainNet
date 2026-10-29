#!/bin/bash
# Smoke test: run the full train + eval pipeline for a single target.
# Verify this completes successfully before submitting the full sweep.
# Set for your own cluster: GPU partition to submit to (empty = cluster default).
PARTITION="${PARTITION:-}"
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

sbatch ${PARTITION:+--partition="$PARTITION"} --export=ALL,TARGET=CDK2 slurm/train.sbatch
echo "Submitted test job: CDK2"
echo "Monitor with: squeue -u \$USER"
echo "Logs: logs/train_<jobid>.out"
