#!/bin/bash
set -euo pipefail

# Usage: ./jobs/submit_h1_patient_characteristics_full_evaluation.sh [shards] [max_concurrent]
SHARD_COUNT="${1:-180}"
MAX_CONCURRENT="${2:-30}"

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
RAW_ROOT="${H1PC_FULL_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_full}"
WORKER_SCRIPT="$REPO/jobs/run_h1_patient_characteristics_full_evaluation.sh"
LOG_DIR="$REPO/grid_logs/h1_patient_characteristics_full/evaluation"

for FILE in selected_cells.csv constrained_priority_cells.csv; do
  test -f "$RAW_ROOT/selection/$FILE" || {
    echo "Missing $RAW_ROOT/selection/$FILE. Wait for selection to finish." >&2
    exit 1
  }
done
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export REPO PYTHON H1PC_FULL_RAW_ROOT="$RAW_ROOT"
export SHARD_COUNT EVALUATION_SEEDS="${EVALUATION_SEEDS:-10}" WORKERS="${WORKERS:-1}"

cd "$LOG_DIR"
COMMAND=(
  grid_run
  --grid_submit=batch
  --grid_array="1-${SHARD_COUNT}/${MAX_CONCURRENT}"
  --grid_ncpus=1
  --grid_mem=8G
)
if [[ -n "${GRID_EMAIL:-}" ]]; then
  COMMAND+=(--grid_email="$GRID_EMAIL")
fi
COMMAND+=("$WORKER_SCRIPT")
"${COMMAND[@]}"

echo "Evaluation array submitted. Monitor with qstat."
