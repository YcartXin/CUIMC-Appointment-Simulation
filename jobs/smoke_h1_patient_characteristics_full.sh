#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
SMOKE_ROOT="${SMOKE_ROOT:-/scratch/$USER/h1_patient_characteristics_full_smoke}"
SMOKE_BANK="$SMOKE_ROOT/smoke_bank.csv"

cd "$REPO"
rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"
"$PYTHON" experiments/h1_patient_characteristics_full_bank.py --smoke --output "$SMOKE_BANK"
"$PYTHON" experiments/h1_patient_characteristics_full.py search \
  --smoke --bank "$SMOKE_BANK" --output-dir "$SMOKE_ROOT" --workers 1 --n-seeds 2
"$PYTHON" experiments/h1_patient_characteristics_full.py select \
  --smoke --n-seeds 2 --bank "$SMOKE_BANK" --output-dir "$SMOKE_ROOT"
"$PYTHON" experiments/h1_patient_characteristics_full.py evaluate \
  --smoke --bank "$SMOKE_BANK" --output-dir "$SMOKE_ROOT" --workers 1 --n-seeds 2
"$PYTHON" analysis/h1_patient_characteristics_full_postprocess.py \
  --raw-root "$SMOKE_ROOT" --bank "$SMOKE_BANK" \
  --output-root "$SMOKE_ROOT/final" --bootstrap-draws 100 \
  --tolerance 0.005 --class2-tolerance 0.01 \
  --expected-evaluation-seeds 2

echo "Smoke test complete: $SMOKE_ROOT/final"
