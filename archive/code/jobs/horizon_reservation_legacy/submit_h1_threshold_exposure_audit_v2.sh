#!/bin/bash
set -euo pipefail

# Usage:
#   bash jobs/submit_h1_threshold_exposure_audit_v2.sh \
#       [number_of_shards] [workers_per_shard] [seeds]
#
# Recommended default: 10 shards, 1 worker each, 2 seeds.

SHARD_COUNT="${1:-10}"
WORKERS="${2:-1}"
N_SEEDS="${3:-2}"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
BANK="$REPO/outputs/hypotheses/h1_threshold_exposure_audit_v2_bank.csv"
WORKER_SCRIPT="$REPO/jobs/run_h1_threshold_exposure_audit_v2.sh"
LOG_DIR="$REPO/grid_logs/h1_threshold_exposure_audit_v2"
AUDIT_V2_RAW_ROOT="${AUDIT_V2_RAW_ROOT:-/scratch/$USER/h1_threshold_exposure_audit_v2}"

cd "$REPO"
command -v qsub >/dev/null 2>&1

test -x "$PYTHON"
test -f "$WORKER_SCRIPT"

mkdir -p "$LOG_DIR" "$AUDIT_V2_RAW_ROOT"
rm -rf "$AUDIT_V2_RAW_ROOT/raw" "$AUDIT_V2_RAW_ROOT/summary"

"$PYTHON" experiments/h1_threshold_exposure_audit_v2.py build --bank "$BANK"

echo "Submitting $SHARD_COUNT second-audit shards"
echo "  Workers per shard: $WORKERS"
echo "  Seeds:             $N_SEEDS"
echo "  Raw output:        $AUDIT_V2_RAW_ROOT"

for ((SHARD_INDEX=0; SHARD_INDEX<SHARD_COUNT; SHARD_INDEX++)); do
    JOB_NAME="h1th2_${SHARD_INDEX}"
    qsub \
        -N "$JOB_NAME" \
        -cwd \
        -V \
        -o "$LOG_DIR" \
        -e "$LOG_DIR" \
        "$WORKER_SCRIPT" \
        "$SHARD_INDEX" \
        "$SHARD_COUNT" \
        "$WORKERS" \
        "$N_SEEDS"
    echo "Submitted $JOB_NAME"
done

echo
echo "Check status with: qstat"
echo "Check errors with:"
echo "  grep -RniE 'traceback|error|exception|killed' \"$LOG_DIR\""
echo "After qstat is empty, run:"
echo "  bash jobs/finalize_h1_threshold_exposure_audit_v2.sh"
