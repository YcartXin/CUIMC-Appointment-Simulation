#!/bin/bash
set -euo pipefail

# Submit the full release-only patient-characteristics run.
#
# Usage:
#   bash jobs/submit_h1_patient_characteristics_confirmatory.sh \
#       [number_of_shards] [workers_per_shard] [seeds]
#
# Recommended:
#   First pass: bash jobs/submit_h1_patient_characteristics_confirmatory.sh 30 1 5
#   Final top-up: rerun the same command with 10 seeds after the first pass
#                 is fully complete. Resume logic reuses the five-seed work.

SHARD_COUNT="${1:-30}"
WORKERS="${2:-1}"
N_SEEDS="${3:-5}"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
H1PC_CONFIRM_RAW_ROOT="${H1PC_CONFIRM_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_confirmatory}"

BANK="$REPO/outputs/hypotheses/h1_patient_characteristics_confirmatory_bank.csv"
WORKER_SCRIPT="$REPO/jobs/run_h1_patient_characteristics_confirmatory.sh"
LOG_DIR="$REPO/grid_logs/h1_patient_characteristics_confirmatory"

cd "$REPO"

command -v qsub >/dev/null 2>&1 || {
    echo "qsub was not found. Run this script from the CBS Research Grid." >&2
    exit 1
}

test -x "$PYTHON" || {
    echo "Python environment not found: $PYTHON" >&2
    exit 1
}

test -f "$WORKER_SCRIPT" || {
    echo "Worker script not found: $WORKER_SCRIPT" >&2
    exit 1
}

mkdir -p "$LOG_DIR" "$H1PC_CONFIRM_RAW_ROOT"

echo "Regenerating and validating the 3,150-background bank..."
"$PYTHON" experiments/h1_patient_characteristics_confirmatory_bank.py

test -f "$BANK" || {
    echo "Bank was not generated: $BANK" >&2
    exit 1
}

echo
echo "Submitting $SHARD_COUNT full-run shards:"
echo "  Workers per shard: $WORKERS"
echo "  Paired seeds:       $N_SEEDS"
echo "  Objectives:         average_utilization, weighted_utilization"
echo "  Variant:            release"
echo "  Raw output:         $H1PC_CONFIRM_RAW_ROOT"
echo "  Grid logs:          $LOG_DIR"
echo

for ((SHARD_INDEX=0; SHARD_INDEX<SHARD_COUNT; SHARD_INDEX++)); do
    JOB_NAME="h1pcf_${SHARD_INDEX}"

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

    echo "Submitted $JOB_NAME (shard $SHARD_INDEX of $SHARD_COUNT)"
done

echo
echo "Check status with: qstat"
echo "Check errors with:"
echo "  grep -RniE 'traceback|error|exception|killed|out of memory' \"$LOG_DIR\""
echo
echo "After qstat is empty, confirm completion messages and run the verifier."
