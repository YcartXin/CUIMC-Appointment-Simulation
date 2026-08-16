#!/bin/bash
set -euo pipefail

STAGE="${STAGE:?Set STAGE to search, select, or evaluate}"
REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
RAW_ROOT="${PBF_RAW_ROOT:-/scratch/$USER/patient_behavior_factorial}"
BANK="$REPO/outputs/hypotheses/patient_behavior_factorial_bank.csv"
SEARCH_SEEDS="${SEARCH_SEEDS:-5}"
EVALUATION_SEEDS="${EVALUATION_SEEDS:-10}"
WORKERS="${WORKERS:-1}"
SMOKE_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  SMOKE_ARGS+=(--smoke)
fi

cd "$REPO"

case "$STAGE" in
  search)
    : "${SGE_TASK_ID:?search must run as a GRID array task}"
    SHARD_COUNT="${SHARD_COUNT:-${SGE_TASK_LAST:?missing shard count}}"
    SHARD_INDEX=$((SGE_TASK_ID - 1))
    "$PYTHON" -u experiments/patient_behavior_factorial.py search \
      --bank "$BANK" \
      --output-dir "$RAW_ROOT" \
      --workers "$WORKERS" \
      --n-seeds "$SEARCH_SEEDS" \
      --shard-index "$SHARD_INDEX" \
      --shard-count "$SHARD_COUNT" \
      "${SMOKE_ARGS[@]}"
    ;;
  select)
    "$PYTHON" -u experiments/patient_behavior_factorial.py select \
      --bank "$BANK" \
      --output-dir "$RAW_ROOT" \
      --n-seeds "$SEARCH_SEEDS" \
      "${SMOKE_ARGS[@]}"
    ;;
  evaluate)
    : "${SGE_TASK_ID:?evaluate must run as a GRID array task}"
    SHARD_COUNT="${SHARD_COUNT:-${SGE_TASK_LAST:?missing shard count}}"
    SHARD_INDEX=$((SGE_TASK_ID - 1))
    "$PYTHON" -u experiments/patient_behavior_factorial.py evaluate \
      --bank "$BANK" \
      --output-dir "$RAW_ROOT" \
      --workers "$WORKERS" \
      --n-seeds "$EVALUATION_SEEDS" \
      --shard-index "$SHARD_INDEX" \
      --shard-count "$SHARD_COUNT" \
      "${SMOKE_ARGS[@]}"
    ;;
  *)
    echo "Unknown STAGE=$STAGE; use search, select, or evaluate" >&2
    exit 2
    ;;
esac
