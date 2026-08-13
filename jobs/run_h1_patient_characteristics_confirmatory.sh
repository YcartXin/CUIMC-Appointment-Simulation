#!/bin/bash
set -euo pipefail

# Run one shard of the full patient-characteristics policy search.
#
# Usage:
#   bash jobs/run_h1_patient_characteristics_confirmatory.sh \
#       <shard_index> <shard_count> [workers] [n_seeds]
#
# The average-utilization pass runs first. The weighted-objective pass then
# reuses all compatible raw policy cells and adds only missing refinements.

SHARD_INDEX="${1:?shard_index is required}"
SHARD_COUNT="${2:?shard_count is required}"
WORKERS="${3:-1}"
N_SEEDS="${4:-5}"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
BANK="$REPO/outputs/hypotheses/h1_patient_characteristics_confirmatory_bank.csv"
OUTPUT="${H1PC_CONFIRM_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_confirmatory}"

cd "$REPO"

test -x "$PYTHON" || {
    echo "Python environment not found: $PYTHON" >&2
    exit 1
}

test -f "$BANK" || {
    echo "Full bank not found: $BANK" >&2
    echo "Run: $PYTHON experiments/h1_patient_characteristics_confirmatory_bank.py" >&2
    exit 1
}

COMMON_ARGS=(
    run
    --variant release
    --bank "$BANK"
    --output-dir "$OUTPUT"
    --n-seeds "$N_SEEDS"
    --shard-index "$SHARD_INDEX"
    --shard-count "$SHARD_COUNT"
    --workers "$WORKERS"
)

echo "Full shard $SHARD_INDEX/$SHARD_COUNT: average-utilization objective"
"$PYTHON" -u experiments/h1_short_horizon_reservation.py \
    "${COMMON_ARGS[@]}" \
    --objective average_utilization

echo "Full shard $SHARD_INDEX/$SHARD_COUNT: priority-weighted objective"
"$PYTHON" -u experiments/h1_short_horizon_reservation.py \
    "${COMMON_ARGS[@]}" \
    --objective weighted_utilization

echo "Full patient-characteristics shard complete: $SHARD_INDEX/$SHARD_COUNT"
