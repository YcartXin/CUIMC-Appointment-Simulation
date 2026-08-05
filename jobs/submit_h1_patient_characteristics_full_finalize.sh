#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
RAW_ROOT="${H1PC_FULL_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_full}"
WORKER_SCRIPT="$REPO/jobs/run_h1_patient_characteristics_full_finalize.sh"
LOG_DIR="$REPO/grid_logs/h1_patient_characteristics_full/finalize"
EXPECTED=3780

FOUND=$(find "$RAW_ROOT/evaluation/raw" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l)
if [[ "$FOUND" -ne "$EXPECTED" ]]; then
  echo "Expected $EXPECTED evaluation shards; found $FOUND. Do not finalize yet." >&2
  exit 1
fi
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export REPO PYTHON H1PC_FULL_RAW_ROOT="$RAW_ROOT"
export EVALUATION_SEEDS="${EVALUATION_SEEDS:-10}"
cd "$LOG_DIR"
grid_run --grid_submit=batch --grid_ncpus=1 --grid_mem=16G "$WORKER_SCRIPT"
echo "Finalize job submitted. Monitor with qstat."
