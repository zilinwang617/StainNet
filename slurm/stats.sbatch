#!/bin/bash
#SBATCH --job-name=stainnet-stats
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 1:00:00
#SBATCH -o logs/stats_%j.out
#SBATCH -e logs/stats_%j.err
# --- Cluster-specific: set the partition for your own cluster ---
# Submit with: sbatch -p <cpu_partition> slurm/stats.sbatch

set -eo pipefail

# Project checkout; defaults to the directory sbatch was called from.
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
cd "$PROJECT_DIR"

if command -v module >/dev/null 2>&1; then
    module load "${CONDA_MODULE:-anaconda}"
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-stainnet}"

echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

python src/utilities/compute_dataset_stats.py \
    --manifest  "$DATA_DIR/splits/stacked_raw_split.json" \
    --output-dir "$DATA_DIR/stats"

echo "Done: $(date)"
