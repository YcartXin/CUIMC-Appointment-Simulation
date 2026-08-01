#!/bin/bash
set -euo pipefail

# Usage: ./jobs/submit_h1_patient_characteristics_full_search.sh [shards] [max_concurrent]
SHARD_COUNT="${1:-180}"
MAX_CONCURRENT="${2:-30}"

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
RAW_ROOT="${H1PC_FULL_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_full}"
BANK="$REPO/outputs/hypotheses/h1_patient_characteristics_full_bank.csv"
WORKER_SCRIPT="$REPO/jobs/run_h1_patient_characteristics_full_search.sh"
LOG_DIR="$REPO/grid_logs/h1_patient_characteristics_full/search"

command -v grid_run >/dev/null 2>&1 || { echo "grid_run not found" >&2; exit 1; }
test -x "$PYTHON" || { echo "Python not found: $PYTHON" >&2; exit 1; }
test -x "$WORKER_SCRIPT" || { echo "Worker is not executable: $WORKER_SCRIPT" >&2; exit 1; }

mkdir -p "$RAW_ROOT" "$LOG_DIR"
cd "$REPO"
"$PYTHON" experiments/h1_patient_characteristics_full_bank.py --output "$BANK"
"$PYTHON" -m pytest -q tests/test_h1_patient_characteristics_full.py

export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export REPO PYTHON H1PC_FULL_RAW_ROOT="$RAW_ROOT"
export SHARD_COUNT SEARCH_SEEDS="${SEARCH_SEEDS:-5}" WORKERS="${WORKERS:-1}"

cd "$LOG_DIR"
COMMAND=(
  grid_run
  --grid_submit=batch
  --grid_array="1-${SHARD_COUNT}/${MAX_CONCURRENT}"
  --grid_ncpus=1
  --grid_mem=8G
  --grid_long
)
if [[ -n "${GRID_EMAIL:-}" ]]; then
  COMMAND+=(--grid_email="$GRID_EMAIL")
fi
COMMAND+=("$WORKER_SCRIPT")

printf 'Submitting search array: %q ' "${COMMAND[@]}"; echo
"${COMMAND[@]}"

echo "Search raw output: $RAW_ROOT/search/raw"
echo "Logs: $LOG_DIR"
echo "Monitor with: qstat"
