#!/bin/bash
# Set for your own cluster: GPU partition to submit to (empty = cluster default).
PARTITION="${PARTITION:-}"
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

TARGETS=("E2F1" "Akt" "YAP")
for TARGET in "${TARGETS[@]}"; do
    sbatch ${PARTITION:+--partition="$PARTITION"} --export=ALL,TARGET=$TARGET slurm/train.sbatch
    echo "Submitted: $TARGET"
done
