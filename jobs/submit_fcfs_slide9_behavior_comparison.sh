#!/bin/bash
set -euo pipefail

# Usage:
#   ./jobs/submit_fcfs_slide9_behavior_comparison.sh audit-design [n_seeds]
#   ./jobs/submit_fcfs_slide9_behavior_comparison.sh audit-run [shards] [max_concurrent] [task_range]
#   ./jobs/submit_fcfs_slide9_behavior_comparison.sh audit-analyze [shards]
#   ./jobs/submit_fcfs_slide9_behavior_comparison.sh full-design [n_seeds]
#   ./jobs/submit_fcfs_slide9_behavior_comparison.sh full-run [shards] [max_concurrent] [task_range]
#   ./jobs/submit_fcfs_slide9_behavior_comparison.sh full-analyze [shards]

STAGE="${1:?Usage: $0 audit-design|audit-run|audit-analyze|full-design|full-run|full-analyze}"
SLIDE9_STUDY="${STAGE%%-*}"
ACTION="${STAGE#*-}"

case "$SLIDE9_STUDY" in
  audit|full) ;;
  *) echo "Invalid stage: $STAGE" >&2; exit 2 ;;
esac
case "$ACTION" in
  design|run|analyze) ;;
  *) echo "Invalid stage: $STAGE" >&2; exit 2 ;;
esac

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
ROOT="${FCFS_SLIDE9_ROOT:-/scratch/$USER/fcfs_slide9_behavior_comparison}"
OUTPUT_DIR="$ROOT/$SLIDE9_STUDY"
WORKER="$REPO/jobs/run_fcfs_slide9_behavior_comparison.sh"
LOG_DIR="$REPO/grid_logs/fcfs_slide9_behavior_comparison/$SLIDE9_STUDY"

test -x "$PYTHON" || { echo "Python not found: $PYTHON" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$REPO"

case "$ACTION" in
  design)
    if [[ "$SLIDE9_STUDY" == "audit" ]]; then
      DEFAULT_SEEDS=10
    else
      DEFAULT_SEEDS=100
    fi
    N_SEEDS="${2:-$DEFAULT_SEEDS}"
    "$PYTHON" experiments/fcfs_slide9_behavior_comparison.py \
      --mode design \
      --study "$SLIDE9_STUDY" \
      --output-dir "$OUTPUT_DIR" \
      --n-seeds "$N_SEEDS"
    ;;

  run)
    if [[ "$SLIDE9_STUDY" == "audit" ]]; then
      DEFAULT_SHARDS=8
      DEFAULT_CONCURRENT=8
    else
      DEFAULT_SHARDS=40
      DEFAULT_CONCURRENT=20
    fi
    SHARD_COUNT="${2:-$DEFAULT_SHARDS}"
    MAX_CONCURRENT="${3:-$DEFAULT_CONCURRENT}"
    TASK_RANGE="${4:-1-$SHARD_COUNT}"
    test -f "$OUTPUT_DIR/design.csv" || {
      echo "Missing $OUTPUT_DIR/design.csv. Run $SLIDE9_STUDY-design first." >&2
      exit 1
    }
    command -v grid_run >/dev/null 2>&1 || { echo "grid_run not found" >&2; exit 1; }
    test -x "$WORKER" || { echo "Worker is not executable: $WORKER" >&2; exit 1; }
    export REPO PYTHON FCFS_SLIDE9_ROOT="$ROOT" SLIDE9_STUDY SHARD_COUNT
    cd "$LOG_DIR"
    grid_run \
      --grid_submit=batch \
      --grid_array="${TASK_RANGE}/${MAX_CONCURRENT}" \
      --grid_ncpus=1 \
      --grid_mem=8G \
      "$WORKER"
    ;;

  analyze)
    if [[ "$SLIDE9_STUDY" == "audit" ]]; then
      DEFAULT_SHARDS=8
    else
      DEFAULT_SHARDS=40
    fi
    SHARD_COUNT="${2:-$DEFAULT_SHARDS}"
    test -d "$OUTPUT_DIR/completed" || {
      echo "Missing completion directory: $OUTPUT_DIR/completed" >&2
      exit 1
    }
    FOUND=$(find "$OUTPUT_DIR/completed" -maxdepth 1 \
      -name "shard_*_of_$(printf '%04d' "$SHARD_COUNT").done" | wc -l)
    if [[ "$FOUND" -ne "$SHARD_COUNT" ]]; then
      echo "Expected $SHARD_COUNT completion markers; found $FOUND. Do not analyze yet." >&2
      exit 1
    fi
    "$PYTHON" experiments/fcfs_slide9_behavior_comparison.py \
      --mode analyze \
      --study "$SLIDE9_STUDY" \
      --output-dir "$OUTPUT_DIR"
    ;;
esac

echo "Stage complete: $STAGE"
echo "Output directory: $OUTPUT_DIR"
echo "Grid logs: $LOG_DIR"
