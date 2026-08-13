#!/bin/bash
set -euo pipefail

WORKERS="${1:-1}"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
BANK="$REPO/outputs/hypotheses/h1_patient_characteristics_confirmatory_bank.csv"
OUTPUT="$REPO/outputs/hypotheses/h1_patient_characteristics_confirmatory_smoke"

cd "$REPO"

"$PYTHON" experiments/h1_patient_characteristics_confirmatory_bank.py
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

echo "Full-design smoke test complete: $OUTPUT"
