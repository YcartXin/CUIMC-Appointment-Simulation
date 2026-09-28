#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
BANK="${PBF_BANK:-$REPO/outputs/hypotheses/patient_behavior_factorial_bank.csv}"
ROOT="${PBF_PARETO_ROOT:-/scratch/$USER/patient_behavior_factorial_served_rate_pareto}"
WORKERS="${NSLOTS:-1}"
: "${SGE_TASK_ID:?Run as an SGE array task}"
SHARD_COUNT="${SHARD_COUNT:-${SGE_TASK_LAST:?Missing SHARD_COUNT}}"
SHARD_INDEX=$((SGE_TASK_ID - 1))
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$REPO"
"$PYTHON" -u analysis/patient_behavior_factorial_served_rate_pareto.py refine \
  --bank "$BANK" --output-dir "$ROOT" --workers "$WORKERS" \
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT"
