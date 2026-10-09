#!/usr/bin/env bash
#$ -cwd
#$ -V
set -euo pipefail

STAGE="${STAGE:?Set STAGE=bank|search|catalog|evaluate|postprocess}"
REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
PBF_FOLLOWUP_ROOT="${PBF_FOLLOWUP_ROOT:-/scratch/$USER/patient_behavior_factorial_share80}"
PBF_FOLLOWUP_BANK="${PBF_FOLLOWUP_BANK:-$REPO/outputs/hypotheses/patient_behavior_factorial_share80_bank.csv}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

ARGS=("$STAGE" --bank "$PBF_FOLLOWUP_BANK" --output-dir "$PBF_FOLLOWUP_ROOT" --workers 1)
case "$STAGE" in
  search|evaluate)
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
      ARGS+=(--shard-index 0 --shard-count 1 --dry-run)
    else
      : "${SGE_TASK_ID:?Run this stage as a Grid array}"
      ARGS+=(--shard-index "$((SGE_TASK_ID - 1))" --shard-count "${SHARD_COUNT:-54}")
    fi
    ;;
  bank|catalog|postprocess) ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
"$PYTHON" -u analysis/experiment2_share80_followup.py "${ARGS[@]}"
