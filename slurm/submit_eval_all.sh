#!/bin/bash
# Re-run test-set evaluation for all 15 bc64 no-aug checkpoints
# using the updated evaluate.py (per-cell intensity + correlations).
# Set for your own cluster: CPU partition to submit to (empty = cluster default).
PARTITION="${PARTITION:-}"
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

TARGETS=(
    # Phase A
    "E2F1" "Akt" "YAP"
    # Phase B — cell cycle
    "CDK2" "CDK4" "p21" "p27"
    # Phase B — cell cycle regulators
    "Skp2" "Cdh1" "D1"
    # Phase B — MAPK signalling
    "ERK" "pERK"
    # Phase B — phosphorylated substrates
    "pAkt" "pS6"
    # Phase B — DNA damage
    "pH2AX"
)

for TARGET in "${TARGETS[@]}"; do
    sbatch ${PARTITION:+--partition="$PARTITION"} --export=ALL,TARGET=$TARGET slurm/eval.sbatch
    echo "Submitted eval: $TARGET"
done
