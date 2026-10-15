#!/usr/bin/env bash
#$ -cwd
#$ -V
set -euo pipefail
REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
E2_ROOT="${E2_ROOT:-/scratch/$USER/experiment2_frontier}"
E2_SOURCES="${E2_SOURCES:-$REPO/outputs/hypotheses/experiment2_frontier_sources.json}"
STAGE="${STAGE:?Set bank|inventory|search|select|evaluate|postprocess}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
cd "$REPO"
args=("$STAGE" --root "$E2_ROOT" --sources "$E2_SOURCES")
case "$STAGE" in
 search|evaluate)
   if [[ "${DRY_RUN:-0}" == 1 ]]; then args+=(--dry-run)
   else
     : "${SGE_TASK_ID:?Submit stable background IDs from inventory; do not overlap arrays}"
     args+=(--shard "$SGE_TASK_ID")
   fi ;;
 bank|inventory|select|postprocess) ;;
 *) echo "Invalid STAGE=$STAGE" >&2; exit 2 ;;
esac
"$PYTHON" -u analysis/experiment2_frontier.py "${args[@]}"
