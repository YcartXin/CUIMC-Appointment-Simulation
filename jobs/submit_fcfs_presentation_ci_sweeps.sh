#!/bin/bash
set -euo pipefail

# Usage:
#   ./jobs/submit_fcfs_presentation_ci_sweeps.sh design [n_seeds]
#   ./jobs/submit_fcfs_presentation_ci_sweeps.sh run [shards] [max_concurrent] [task_range]
#   ./jobs/submit_fcfs_presentation_ci_sweeps.sh analyze [shards]
#
# Recommended Grid sequence:
#   ./jobs/submit_fcfs_presentation_ci_sweeps.sh design 100
#   ./jobs/submit_fcfs_presentation_ci_sweeps.sh run 40 2 1-2
#   # inspect qstat and the first two logs
#   ./jobs/submit_fcfs_presentation_ci_sweeps.sh run 40 20 1-40
#   ./jobs/submit_fcfs_presentation_ci_sweeps.sh analyze 40

STAGE="${1:?Usage: $0 design|run|analyze [arguments]}"
REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
OUTPUT_DIR="${FCFS_PRESENTATION_CI_ROOT:-/scratch/$USER/fcfs_presentation_ci}"
WORKER="$REPO/jobs/run_fcfs_presentation_ci_sweeps.sh"
LOG_DIR="$REPO/grid_logs/fcfs_presentation_ci"

test -x "$PYTHON" || { echo "Python not found: $PYTHON" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$REPO"

case "$STAGE" in
  design)
    N_SEEDS="${2:-100}"
    "$PYTHON" experiments/fcfs_presentation_ci_sweeps.py \
      --mode design \
      --output-dir "$OUTPUT_DIR" \
      --n-seeds "$N_SEEDS"
    ;;

  run)
    SHARD_COUNT="${2:-40}"
    MAX_CONCURRENT="${3:-20}"
    TASK_RANGE="${4:-1-$SHARD_COUNT}"
    test -f "$OUTPUT_DIR/design.csv" || {
      echo "Missing $OUTPUT_DIR/design.csv. Run the design stage first." >&2
      exit 1
    }
    command -v grid_run >/dev/null 2>&1 || { echo "grid_run not found" >&2; exit 1; }
    test -x "$WORKER" || { echo "Worker is not executable: $WORKER" >&2; exit 1; }
    export REPO PYTHON FCFS_PRESENTATION_CI_ROOT="$OUTPUT_DIR" SHARD_COUNT
    cd "$LOG_DIR"
    grid_run \
      --grid_submit=batch \
      --grid_array="${TASK_RANGE}/${MAX_CONCURRENT}" \
      --grid_ncpus=1 \
      --grid_mem=8G \
      "$WORKER"
    ;;

  analyze)
    SHARD_COUNT="${2:-40}"
    test -d "$OUTPUT_DIR/completed" || {
      echo "Missing completion directory: $OUTPUT_DIR/completed" >&2
      exit 1
    }
    FOUND=$(find "$OUTPUT_DIR/completed" -maxdepth 1 -name "shard_*_of_$(printf '%04d' "$SHARD_COUNT").done" | wc -l)
    if [[ "$FOUND" -ne "$SHARD_COUNT" ]]; then
      echo "Expected $SHARD_COUNT completion markers; found $FOUND. Do not analyze yet." >&2
      exit 1
    fi
    "$PYTHON" experiments/fcfs_presentation_ci_sweeps.py \
      --mode analyze \
      --output-dir "$OUTPUT_DIR"
    ;;

  *)
    echo "Unknown stage: $STAGE" >&2
    exit 2
    ;;
esac

echo "Stage complete: $STAGE"
echo "Output root: $OUTPUT_DIR"
echo "Grid logs: $LOG_DIR"
