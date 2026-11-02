#!/usr/bin/env bash
#$ -cwd
#$ -V
set -euo pipefail
REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
: "${STAGE:?Set catalog|inventory|evaluate|postprocess}"
: "${E2_ROOT:?Expanded frontier root required}"
: "${E2_PARETO_ROOT:?Separate new Pareto root required}"
: "${E2_SOURCES:?Reuse source manifest required}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
cd "$REPO"
args=("$STAGE" --root "$E2_PARETO_ROOT" --frontier-root "$E2_ROOT" --sources "$E2_SOURCES")
if [[ "$STAGE" == evaluate ]]; then
    if [[ "${DRY_RUN:-0}" == 1 ]]; then args+=(--dry-run)
    else args+=(--shard "${SGE_TASK_ID:?Submit the missing Pareto evaluation shards}"); fi
fi
"$PYTHON" -u analysis/experiment2_pareto_completion.py "${args[@]}"
