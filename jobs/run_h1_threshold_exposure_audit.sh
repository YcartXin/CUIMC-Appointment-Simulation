#!/bin/bash
set -euo pipefail

# Run one baseline-only audit shard.
# Usage:
#   ./jobs/run_h1_threshold_exposure_audit.sh \
#       <shard_index> <shard_count> [workers] [n_seeds]

SHARD_INDEX="${1:?shard_index is required}"
SHARD_COUNT="${2:?shard_count is required}"
WORKERS="${3:-1}"
N_SEEDS="${4:-2}"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
BANK="$REPO/outputs/hypotheses/h1_threshold_exposure_audit_bank.csv"
OUTPUT="${AUDIT_RAW_ROOT:-/scratch/$USER/h1_threshold_exposure_audit}"

cd "$REPO"
test -x "$PYTHON"
test -f "$BANK"

"$PYTHON" -u experiments/h1_threshold_exposure_audit.py run \
    --bank "$BANK" \
    --output-dir "$OUTPUT" \
    --workers "$WORKERS" \
    --n-seeds "$N_SEEDS" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT"

echo "Threshold-audit shard complete: $SHARD_INDEX/$SHARD_COUNT"
