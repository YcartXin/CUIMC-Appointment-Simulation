#!/bin/bash
set -euo pipefail

: "${SGE_TASK_ID:?This script must run as a GRID array task}"
SHARD_COUNT="${SHARD_COUNT:-${SGE_TASK_LAST:?SHARD_COUNT or SGE_TASK_LAST is required}}"

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
RAW_ROOT="${H1PC_FULL_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_full}"
BANK="$REPO/outputs/hypotheses/h1_patient_characteristics_full_bank.csv"
SEARCH_SEEDS="${SEARCH_SEEDS:-5}"
WORKERS="${WORKERS:-1}"
SHARD_INDEX=$((SGE_TASK_ID - 1))

cd "$REPO"
"$PYTHON" -u experiments/h1_patient_characteristics_full.py search \
  --bank "$BANK" \
  --output-dir "$RAW_ROOT" \
  --workers "$WORKERS" \
  --n-seeds "$SEARCH_SEEDS" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT"
