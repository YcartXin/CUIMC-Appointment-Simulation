#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
RAW_ROOT="${PBF_RAW_ROOT:-/scratch/$USER/patient_behavior_factorial_3_5}"
BANK="${PBF_BANK:-$REPO/outputs/hypotheses/patient_behavior_factorial_bank.csv}"
SEARCH_SEEDS="${SEARCH_SEEDS:-5}"
WORKERS="${WORKERS:-1}"

: "${SGE_TASK_ID:?This worker must run as a GRID array task}"
SHARD_COUNT="${SHARD_COUNT:-${SGE_TASK_LAST:?Missing shard count}}"
SHARD_INDEX=$((SGE_TASK_ID - 1))

cd "$REPO"

"$PYTHON" -u analysis/patient_behavior_factorial_baseline_access_refine.py   --bank "$BANK"   --output-dir "$RAW_ROOT"   --workers "$WORKERS"   --n-seeds "$SEARCH_SEEDS"   --shard-index "$SHARD_INDEX"   --shard-count "$SHARD_COUNT"
