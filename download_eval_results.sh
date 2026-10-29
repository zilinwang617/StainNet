#!/bin/bash
# Download summary.json + metrics.csv for all completed runs from the cluster.
# Skips runs where eval/test/ doesn't exist yet (still training).
#
# Usage:
#   CLUSTER=<user>@<cluster> bash download_eval_results.sh             # all_targets group
#   CLUSTER=<user>@<cluster> bash download_eval_results.sh ungrouped   # different group
#
# Local output: local_runs/<group>/<run_name>/

# Set these for your own cluster (or export them before running).
CLUSTER="${CLUSTER:-<user>@<cluster>}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/StainNet}"   # project checkout on the cluster
REMOTE_BASE="$REMOTE_PROJECT_DIR/runs"
GROUP="${1:-all_targets}"

if [[ "$CLUSTER" == *"<"* ]]; then
    echo "Set CLUSTER to your cluster login first, e.g. CLUSTER=<user>@<cluster> bash $0"
    exit 1
fi
LOCAL_BASE="$(dirname "$0")/local_runs/$GROUP"

echo "Downloading eval results for group: $GROUP"
echo "Local destination: $LOCAL_BASE"
echo ""

# Get list of run directories on the cluster
RUN_DIRS=$(ssh "$CLUSTER" "ls -d $REMOTE_BASE/$GROUP/*/eval/test 2>/dev/null")

if [ -z "$RUN_DIRS" ]; then
    echo "No completed eval runs found under $REMOTE_BASE/$GROUP/"
    exit 0
fi

N_OK=0
N_SKIP=0

while IFS= read -r EVAL_DIR; do
    # e.g. ~/StainNet/runs/all_targets/Akt_55603920/eval/test
    RUN_NAME=$(echo "$EVAL_DIR" | awk -F'/' '{print $(NF-2)}')
    LOCAL_DIR="$LOCAL_BASE/$RUN_NAME"
    mkdir -p "$LOCAL_DIR"

    echo -n "$RUN_NAME ... "

    # Download summary.json and metrics.csv only (skip images/)
    if scp -q \
        "$CLUSTER:$EVAL_DIR/summary.json" \
        "$CLUSTER:$EVAL_DIR/metrics.csv" \
        "$LOCAL_DIR/" 2>/dev/null; then
        echo "OK"
        (( N_OK++ ))
    else
        echo "MISSING (eval not done yet?)"
        rmdir "$LOCAL_DIR" 2>/dev/null
        (( N_SKIP++ ))
    fi
done <<< "$RUN_DIRS"

echo ""
echo "Done: $N_OK downloaded, $N_SKIP skipped."
echo "Results in: $LOCAL_BASE"
