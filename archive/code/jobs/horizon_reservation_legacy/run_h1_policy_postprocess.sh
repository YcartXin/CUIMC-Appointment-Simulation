#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

REPO="$HOME/projects/CUIMC-Appointment-Simulation"
PYTHON="$HOME/.conda/envs/cuimc/bin/python"
RAW_ROOT="/scratch/$USER/h1_short_horizon_reservation_10seed_v2"
BANK="$REPO/outputs/hypotheses/background_scenarios.csv"
SUMMARY_ROOT="$REPO/full_run_summaries"
OUTPUT_ROOT="$REPO/full_run_summaries/h1_policy_outcomes"

cd "$REPO"

test -x "$PYTHON"
test -d "$RAW_ROOT/strict/raw"
test -d "$RAW_ROOT/release/raw"
test -f "$BANK"

exec "$PYTHON" -u analysis/h1_postprocess_policy_outcomes.py \
  --raw-root "$RAW_ROOT" \
  --bank "$BANK" \
  --summary-root "$SUMMARY_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --variants strict release \
  --bootstrap-draws 2000 \
  --tolerance 0.005
