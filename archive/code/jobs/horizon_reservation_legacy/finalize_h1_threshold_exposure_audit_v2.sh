#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
RAW_ROOT="${AUDIT_V2_RAW_ROOT:-/scratch/$USER/h1_threshold_exposure_audit_v2}"
SUMMARY_ROOT="$REPO/full_run_summaries/h1_threshold_exposure_audit_v2"

cd "$REPO"
test -x "$PYTHON"
test -d "$RAW_ROOT/raw"

rm -rf "$SUMMARY_ROOT"
mkdir -p "$SUMMARY_ROOT"

"$PYTHON" -u experiments/h1_threshold_exposure_audit_v2.py summarize \
    --output-dir "$RAW_ROOT" \
    --summary-dir "$SUMMARY_ROOT"

echo "Second threshold-audit outputs written to: $SUMMARY_ROOT"
