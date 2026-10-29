#!/bin/bash
# Submit train+eval for all targets on A100 GPUs.
# Logs go to logs/YYYYMMDD/<jobid>.{out,err}
# Runs go to runs/all_targets/<target>_<jobid>/
# Set for your own cluster: GPU partition to submit to (empty = cluster default).
PARTITION="${PARTITION:-}"
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

DATE=$(date +%Y%m%d)
LOG_DIR="logs/$DATE"
mkdir -p "$LOG_DIR"

TARGETS=(
    # Excluded: inputs (DNA_1, PCNA-mTq2, Rb, cycA, cycB1, cycE, pRb)
    #           masks  (nuc, cell, cyto, ring)

    # Baseline proteins
    "ARF" "Akt" "BP1" "Bcl2"
    # CDKs / cell cycle
    "CDK2" "CDK4" "CDK6" "Cdh1" "Cdt1" "D1"
    # DNA content channels
    "DNA_2" "DNA_3" "DNA_4" "DNA_5" "DNA_6" "DNA_7" "DNA_8"
    # Transcription factors / signalling
    "E2F1" "ERK" "Fra1" "GSK3b" "RSK1" "S6" "STAT3" "Skp2" "YAP"
    "bCat" "cFos" "cJun" "cMyc"
    # CKIs / tumour suppressors
    "p16" "p21" "p27" "p38" "p53"
    # Phosphoproteins
    "pAkt" "pChk1" "pERK" "pGSK3b" "pH2AX" "pRSK" "pS6" "pSTAT3"
    # Doubly-phosphorylated / processed forms
    "pp21" "pp27" "pp38" "pp53" "pp65"
)

for TARGET in "${TARGETS[@]}"; do
    JID=$(sbatch ${PARTITION:+--partition="$PARTITION"} \
        --output="$LOG_DIR/%j.out" \
        --error="$LOG_DIR/%j.err" \
        --export=ALL,TARGET=$TARGET,RUN_GROUP=all_targets \
        slurm/train_all_targets.sbatch | awk '{print $4}')
    echo "Submitted $TARGET → job $JID  (logs/$DATE/$JID.out)"
done
