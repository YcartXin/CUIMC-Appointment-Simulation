#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
RAW_ROOT="${H1PC_FULL_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_full}"
BANK="$REPO/outputs/hypotheses/h1_patient_characteristics_full_bank.csv"
OUTPUT_ROOT="$REPO/full_run_summaries/h1_patient_characteristics_full/release"

cd "$REPO"
"$PYTHON" -u analysis/h1_patient_characteristics_full_postprocess.py \
  --raw-root "$RAW_ROOT" \
  --bank "$BANK" \
  --output-root "$OUTPUT_ROOT" \
  --bootstrap-draws 2000 \
  --tolerance 0.005 \
  --class2-tolerance 0.01 \
  --expected-evaluation-seeds "${EVALUATION_SEEDS:-10}"
