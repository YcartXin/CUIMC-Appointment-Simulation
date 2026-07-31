#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
RAW_ROOT="${AUDIT_RAW_ROOT:-/scratch/$USER/h1_threshold_exposure_audit}"
SUMMARY_ROOT="$REPO/full_run_summaries/h1_threshold_exposure_audit"

cd "$REPO"
test -x "$PYTHON"
test -d "$RAW_ROOT/raw"

rm -rf "$SUMMARY_ROOT"
mkdir -p "$SUMMARY_ROOT"

"$PYTHON" -u experiments/h1_threshold_exposure_audit.py summarize \
    --output-dir "$RAW_ROOT" \
    --summary-dir "$SUMMARY_ROOT"

echo "Threshold-audit outputs written to: $SUMMARY_ROOT"
