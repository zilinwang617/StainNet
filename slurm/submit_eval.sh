#!/bin/bash
# Submit test-set evaluation for the three bc64 no-aug runs.
# Each eval is < 5 min so they should finish quickly.
# Set for your own cluster: CPU partition to submit to (empty = cluster default).
PARTITION="${PARTITION:-}"
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

TARGETS=("E2F1" "Akt" "YAP")
for TARGET in "${TARGETS[@]}"; do
    sbatch ${PARTITION:+--partition="$PARTITION"} --export=ALL,TARGET=$TARGET slurm/eval.sbatch
    echo "Submitted eval: $TARGET"
done
