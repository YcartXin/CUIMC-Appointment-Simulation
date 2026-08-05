#!/bin/bash
set -euo pipefail

# Run one shard of the controlled patient-characteristics pilot.
#
# Usage:
#   ./jobs/run_h1_patient_characteristics_pilot.sh \
#       <shard_index> <shard_count> [workers] [n_seeds]
#
# Example: four concurrent jobs, eight workers each, five seeds:
#   qsub jobs/run_h1_patient_characteristics_pilot.sh 0 4 8 5
#   qsub jobs/run_h1_patient_characteristics_pilot.sh 1 4 8 5
#   qsub jobs/run_h1_patient_characteristics_pilot.sh 2 4 8 5
#   qsub jobs/run_h1_patient_characteristics_pilot.sh 3 4 8 5

SHARD_INDEX="${1:?shard_index is required}"
SHARD_COUNT="${2:?shard_count is required}"
WORKERS="${3:-8}"
N_SEEDS="${4:-5}"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
BANK="$REPO/outputs/hypotheses/h1_patient_characteristics_pilot_bank.csv"
OUTPUT="${PILOT_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_pilot}"

cd "$REPO"

test -x "$PYTHON" || {
    echo "Python environment not found: $PYTHON" >&2
    exit 1
}

test -f "$BANK" || {
    echo "Pilot bank not found: $BANK" >&2
    echo "Run: $PYTHON experiments/h1_patient_characteristics_pilot_bank.py" >&2
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

echo "Pilot shard $SHARD_INDEX/$SHARD_COUNT: average-utilization refinement"
"$PYTHON" -u experiments/h1_short_horizon_reservation.py \
    "${COMMON_ARGS[@]}" \
    --objective average_utilization

echo "Pilot shard $SHARD_INDEX/$SHARD_COUNT: priority-weighted refinement"
"$PYTHON" -u experiments/h1_short_horizon_reservation.py \
    "${COMMON_ARGS[@]}" \
    --objective weighted_utilization

echo "Pilot shard complete: $SHARD_INDEX/$SHARD_COUNT"
