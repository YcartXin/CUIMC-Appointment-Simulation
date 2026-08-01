#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
RAW_ROOT="${H1PC_FULL_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_full}"
BANK="$REPO/outputs/hypotheses/h1_patient_characteristics_full_bank.csv"

cd "$REPO"
"$PYTHON" -u experiments/h1_patient_characteristics_full.py select \
  --bank "$BANK" \
  --output-dir "$RAW_ROOT" \
  --n-seeds "${SEARCH_SEEDS:-5}"
