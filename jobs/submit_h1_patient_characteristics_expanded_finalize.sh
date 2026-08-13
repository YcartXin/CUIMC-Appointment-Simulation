#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
RAW_ROOT="${H1PC_EXPANDED_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_expanded}"
WORKER_SCRIPT="$REPO/jobs/run_h1_patient_characteristics_expanded_finalize.sh"
LOG_DIR="$REPO/grid_logs/h1_patient_characteristics_expanded/finalize"
EXPECTED=5670

FOUND=$(find "$RAW_ROOT/evaluation/raw" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l)
if [[ "$FOUND" -ne "$EXPECTED" ]]; then
  echo "Expected $EXPECTED evaluation shards; found $FOUND. Do not finalize yet." >&2
  exit 1
fi

if [[ ! -f "$RAW_ROOT/selection/selected_cells.csv" ]]; then
  echo "Missing $RAW_ROOT/selection/selected_cells.csv" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export REPO PYTHON H1PC_EXPANDED_RAW_ROOT="$RAW_ROOT"
export EVALUATION_SEEDS="${EVALUATION_SEEDS:-10}"
export BOOTSTRAP_DRAWS="${BOOTSTRAP_DRAWS:-2000}"

cd "$LOG_DIR"
grid_run --grid_submit=batch --grid_ncpus=1 --grid_mem=16G "$WORKER_SCRIPT"
echo "Expanded finalize job submitted. Monitor with qstat."
