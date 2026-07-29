#!/bin/bash
set -euo pipefail

# Fast local/grid check before submitting the full pilot.
#
# Usage:
#   ./jobs/smoke_h1_patient_characteristics_pilot.sh [workers]

WORKERS="${1:-4}"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
BANK="$REPO/outputs/hypotheses/h1_patient_characteristics_pilot_bank.csv"
OUTPUT="$REPO/outputs/hypotheses/h1_patient_characteristics_pilot_smoke"

cd "$REPO"

"$PYTHON" experiments/h1_patient_characteristics_pilot_bank.py

rm -rf "$OUTPUT"

"$PYTHON" -u experiments/h1_short_horizon_reservation.py all \
    --variant release \
    --objective average_utilization \
    --bank "$BANK" \
    --output-dir "$OUTPUT" \
    --n-seeds 5 \
    --workers "$WORKERS" \
    --smoke \
    --no-resume

"$PYTHON" -u experiments/h1_short_horizon_reservation.py all \
    --variant release \
    --objective weighted_utilization \
    --bank "$BANK" \
    --output-dir "$OUTPUT" \
    --n-seeds 5 \
    --workers "$WORKERS" \
    --smoke

echo "Smoke test complete: $OUTPUT"
