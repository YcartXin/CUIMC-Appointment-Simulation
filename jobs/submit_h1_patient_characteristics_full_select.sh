#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
RAW_ROOT="${H1PC_FULL_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_full}"
WORKER_SCRIPT="$REPO/jobs/run_h1_patient_characteristics_full_select.sh"
LOG_DIR="$REPO/grid_logs/h1_patient_characteristics_full/select"
EXPECTED=3780

FOUND=$(find "$RAW_ROOT/search/raw" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l)
if [[ "$FOUND" -ne "$EXPECTED" ]]; then
  echo "Expected $EXPECTED search shards; found $FOUND. Do not select yet." >&2
  exit 1
fi
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export REPO PYTHON H1PC_FULL_RAW_ROOT="$RAW_ROOT"
export SEARCH_SEEDS="${SEARCH_SEEDS:-5}"
cd "$LOG_DIR"
grid_run --grid_submit=batch --grid_ncpus=1 --grid_mem=8G "$WORKER_SCRIPT"
echo "Selection job submitted. Monitor with qstat."
