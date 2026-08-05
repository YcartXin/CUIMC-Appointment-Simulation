#!/bin/bash
set -euo pipefail

# Fast end-to-end check.
# Usage: ./jobs/smoke_h1_threshold_exposure_audit.sh [workers]

WORKERS="${1:-2}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
OUTPUT="$REPO/outputs/hypotheses/h1_threshold_exposure_audit_smoke"
BANK="$REPO/outputs/hypotheses/h1_threshold_exposure_audit_bank.csv"

cd "$REPO"
rm -rf "$OUTPUT"

"$PYTHON" -u experiments/h1_threshold_exposure_audit.py all \
    --bank "$BANK" \
    --output-dir "$OUTPUT" \
    --workers "$WORKERS" \
    --n-seeds 1 \
    --smoke \
    --no-resume

echo "Smoke audit complete: $OUTPUT/summary"
