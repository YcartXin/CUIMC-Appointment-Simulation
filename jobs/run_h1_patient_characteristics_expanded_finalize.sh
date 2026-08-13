#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
RAW_ROOT="${H1PC_EXPANDED_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_expanded}"
BANK="$REPO/outputs/hypotheses/h1_patient_characteristics_expanded_bank.csv"
OUTPUT_ROOT="$REPO/full_run_summaries/h1_patient_characteristics_expanded/release"

cd "$REPO"

"$PYTHON" -u analysis/h1_patient_characteristics_expanded_postprocess.py \
  --raw-root "$RAW_ROOT" \
  --bank "$BANK" \
  --output-root "$OUTPUT_ROOT" \
  --bootstrap-draws "${BOOTSTRAP_DRAWS:-2000}" \
  --expected-evaluation-seeds "${EVALUATION_SEEDS:-10}"
