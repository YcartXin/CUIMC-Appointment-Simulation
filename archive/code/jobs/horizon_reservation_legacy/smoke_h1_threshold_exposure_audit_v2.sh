#!/bin/bash
set -euo pipefail

WORKERS="${1:-1}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
OUTPUT="$REPO/outputs/hypotheses/h1_threshold_exposure_audit_v2_smoke"
BANK="$REPO/outputs/hypotheses/h1_threshold_exposure_audit_v2_bank.csv"

cd "$REPO"
rm -rf "$OUTPUT"

"$PYTHON" -u experiments/h1_threshold_exposure_audit_v2.py all \
    --bank "$BANK" \
    --output-dir "$OUTPUT" \
    --workers "$WORKERS" \
    --n-seeds 1 \
    --smoke \
    --no-resume

echo "Second smoke audit complete: $OUTPUT/summary"
