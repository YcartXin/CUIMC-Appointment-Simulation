#!/bin/bash
set -euo pipefail

# Usage:
#   ./jobs/submit_patient_behavior_factorial.sh search [shards] [max_concurrent]
#   ./jobs/submit_patient_behavior_factorial.sh select
#   ./jobs/submit_patient_behavior_factorial.sh evaluate [shards] [max_concurrent]

STAGE="${1:?Usage: $0 search|select|evaluate [shards] [max_concurrent]}"
SHARD_COUNT="${2:-180}"
MAX_CONCURRENT="${3:-30}"
EXPECTED_BACKGROUNDS=540

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
RAW_ROOT="${PBF_RAW_ROOT:-/scratch/$USER/patient_behavior_factorial}"
BANK="$REPO/outputs/hypotheses/patient_behavior_factorial_bank.csv"
WORKER_SCRIPT="$REPO/jobs/run_patient_behavior_factorial.sh"
LOG_DIR="$REPO/grid_logs/patient_behavior_factorial/$STAGE"

command -v grid_run >/dev/null 2>&1 || { echo "grid_run not found" >&2; exit 1; }
test -x "$PYTHON" || { echo "Python not found: $PYTHON" >&2; exit 1; }
test -x "$WORKER_SCRIPT" || { echo "Worker is not executable: $WORKER_SCRIPT" >&2; exit 1; }
mkdir -p "$RAW_ROOT" "$LOG_DIR"

export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export REPO PYTHON PBF_RAW_ROOT="$RAW_ROOT"
export SEARCH_SEEDS="${SEARCH_SEEDS:-5}" EVALUATION_SEEDS="${EVALUATION_SEEDS:-10}"
export WORKERS="${WORKERS:-1}" SHARD_COUNT

cd "$REPO"

case "$STAGE" in
  search)
    "$PYTHON" experiments/patient_behavior_factorial_bank.py --output "$BANK"
    "$PYTHON" - <<'PY'
import pandas as pd
from pathlib import Path
p = Path("outputs/hypotheses/patient_behavior_factorial_bank.csv")
b = pd.read_csv(p)
assert len(b) == 540
assert b["profile_id"].nunique() == 9
assert b["clinic_context_id"].nunique() == 60
assert set(b["horizon_days"]) == {100}
assert set(b["noshow_high_1"]) == {0.05, 0.15, 0.25}
assert set(b["balk_high_1"]) == {0.10, 0.20, 0.30}
assert set(b["noshow_threshold_1"]) == {5}
assert set(b["balk_threshold_1"]) == {7}
print("Bank validation passed: 540 backgrounds, 9 behavior cells, 60 clinic contexts.")
PY
    export STAGE=search
    cd "$LOG_DIR"
    grid_run \
      --grid_submit=batch \
      --grid_array="1-${SHARD_COUNT}/${MAX_CONCURRENT}" \
      --grid_ncpus=1 \
      --grid_mem=8G \
      "$WORKER_SCRIPT"
    ;;

  select)
    FOUND=$(find "$RAW_ROOT/search/raw" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l)
    if [[ "$FOUND" -ne "$EXPECTED_BACKGROUNDS" ]]; then
      echo "Expected $EXPECTED_BACKGROUNDS search files; found $FOUND. Do not select yet." >&2
      exit 1
    fi
    export STAGE=select
    cd "$LOG_DIR"
    grid_run --grid_submit=batch --grid_ncpus=1 --grid_mem=8G "$WORKER_SCRIPT"
    ;;

  evaluate)
    for FILE in selected_cells.csv constrained_priority_cells.csv; do
      test -f "$RAW_ROOT/selection/$FILE" || {
        echo "Missing $RAW_ROOT/selection/$FILE. Wait for selection to finish." >&2
        exit 1
      }
    done
    export STAGE=evaluate
    cd "$LOG_DIR"
    grid_run \
      --grid_submit=batch \
      --grid_array="1-${SHARD_COUNT}/${MAX_CONCURRENT}" \
      --grid_ncpus=1 \
      --grid_mem=8G \
      "$WORKER_SCRIPT"
    ;;

  *)
    echo "Unknown stage: $STAGE" >&2
    exit 2
    ;;
esac

echo "Submitted $STAGE. Raw root: $RAW_ROOT"
echo "Logs: $LOG_DIR"
echo "Monitor with: qstat"
