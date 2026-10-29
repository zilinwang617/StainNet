#!/bin/bash
# Phase B sweep: train bc64 no-aug on 12 representative targets and
# evaluate each on the test set automatically (eval is appended to slurm/train.sbatch).
# Set for your own cluster: GPU partition to submit to (empty = cluster default).
PARTITION="${PARTITION:-}"
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

TARGETS=(
    # Cell cycle — high expected predictability (coupled to cyclin/Rb inputs)
    "CDK2" "CDK4" "p21" "p27"
    # Cell cycle regulators
    "Skp2" "Cdh1" "D1"
    # MAPK signalling
    "ERK" "pERK"
    # Phosphorylated kinase substrates (similar to Akt)
    "pAkt" "pS6"
    # DNA damage
    "pH2AX"
)

for TARGET in "${TARGETS[@]}"; do
    sbatch ${PARTITION:+--partition="$PARTITION"} --export=ALL,TARGET=$TARGET slurm/train.sbatch
    echo "Submitted: $TARGET"
done
