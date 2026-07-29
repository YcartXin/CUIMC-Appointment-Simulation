#!/bin/bash
set -euo pipefail

# Submit the complete controlled patient-characteristics pilot to the CBS grid.
#
# Usage:
#   ./jobs/submit_h1_patient_characteristics_pilot.sh \
#       [number_of_shards] [workers_per_shard] [seeds]
#
# Default:
#   4 shards, 8 workers per shard, 5 paired seeds
#
# Example:
#   ./jobs/submit_h1_patient_characteristics_pilot.sh 4 8 5

SHARD_COUNT="${1:-4}"
WORKERS="${2:-8}"
N_SEEDS="${3:-5}"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
PILOT_RAW_ROOT="${PILOT_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_pilot}"

BANK="$REPO/outputs/hypotheses/h1_patient_characteristics_pilot_bank.csv"
WORKER_SCRIPT="$REPO/jobs/run_h1_patient_characteristics_pilot.sh"
LOG_DIR="$REPO/grid_logs/h1_patient_characteristics_pilot"

cd "$REPO"

command -v qsub >/dev/null 2>&1 || {
    echo "qsub was not found. Run this script from the CBS Research Grid." >&2
    exit 1
}

test -x "$PYTHON" || {
    echo "Python environment not found: $PYTHON" >&2
    echo "Set PYTHON to the correct interpreter before submitting." >&2
    exit 1
}

test -f "$WORKER_SCRIPT" || {
    echo "Worker script not found: $WORKER_SCRIPT" >&2
    exit 1
}

mkdir -p "$LOG_DIR"
mkdir -p "$PILOT_RAW_ROOT"

echo "Regenerating and validating the 420-background pilot bank..."
"$PYTHON" experiments/h1_patient_characteristics_pilot_bank.py

test -f "$BANK" || {
    echo "Pilot bank was not generated: $BANK" >&2
    exit 1
}

echo
echo "Submitting $SHARD_COUNT pilot shards:"
echo "  Workers per shard: $WORKERS"
echo "  Paired seeds:       $N_SEEDS"
echo "  Raw output:         $PILOT_RAW_ROOT"
echo "  Grid logs:          $LOG_DIR"
echo

for ((SHARD_INDEX=0; SHARD_INDEX<SHARD_COUNT; SHARD_INDEX++)); do
    JOB_NAME="h1pc_pilot_${SHARD_INDEX}"

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
echo "All pilot shards submitted."
echo "Check status with:"
echo "  qstat"
echo
echo "Check logs with:"
echo "  ls -lh \"$LOG_DIR\""
echo "  grep -RniE 'traceback|error|exception|killed' \"$LOG_DIR\""
echo
echo "After qstat is empty and the logs contain no errors, run:"
echo "  ./jobs/finalize_h1_patient_characteristics_pilot.sh"
