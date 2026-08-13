#!/bin/bash
set -euo pipefail

# Classify and postprocess the full release-only run under both objectives.

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
BANK="$REPO/outputs/hypotheses/h1_patient_characteristics_confirmatory_bank.csv"
RAW_ROOT="${H1PC_CONFIRM_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_confirmatory}"
OUTPUT_ROOT="$REPO/full_run_summaries/h1_patient_characteristics_confirmatory"

cd "$REPO"

test -x "$PYTHON"
test -f "$BANK"
test -d "$RAW_ROOT/release/raw"

"$PYTHON" -u experiments/h1_short_horizon_reservation.py classify \
    --variant release \
    --objective average_utilization \
    --bank "$BANK" \
    --output-dir "$RAW_ROOT"

"$PYTHON" -u experiments/h1_short_horizon_reservation.py classify \
    --variant release \
    --objective weighted_utilization \
    --bank "$BANK" \
    --output-dir "$RAW_ROOT"

"$PYTHON" -u analysis/h1_postprocess_policy_outcomes.py \
    --raw-root "$RAW_ROOT" \
    --bank "$BANK" \
    --output-root "$OUTPUT_ROOT" \
    --variants release \
    --bootstrap-draws 2000 \
    --tolerance 0.005 \
    --no-summary-validation

echo "Full patient-characteristics outputs written to: $OUTPUT_ROOT/release"
