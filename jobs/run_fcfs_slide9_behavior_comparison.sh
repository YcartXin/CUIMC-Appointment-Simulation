#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$(command -v python)}"
ROOT="${FCFS_SLIDE9_ROOT:-/scratch/$USER/fcfs_slide9_behavior_comparison}"
STUDY="${SLIDE9_STUDY:?SLIDE9_STUDY must be audit or full}"
OUTPUT_DIR="$ROOT/$STUDY"
SHARD_COUNT="${SHARD_COUNT:?SHARD_COUNT must be exported by the submission script}"
SGE_ID="${SGE_TASK_ID:?SGE_TASK_ID is required}"
SHARD_INDEX=$((SGE_ID - 1))

case "$STUDY" in
  audit|full) ;;
  *) echo "Invalid SLIDE9_STUDY: $STUDY" >&2; exit 2 ;;
esac

test -x "$PYTHON" || { echo "Python not found: $PYTHON" >&2; exit 1; }
test -f "$OUTPUT_DIR/design.csv" || { echo "Missing design: $OUTPUT_DIR/design.csv" >&2; exit 1; }

cd "$REPO"
"$PYTHON" -u experiments/fcfs_slide9_behavior_comparison.py \
  --mode run \
  --study "$STUDY" \
  --output-dir "$OUTPUT_DIR" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT"

DONE_DIR="$OUTPUT_DIR/completed"
mkdir -p "$DONE_DIR"
touch "$DONE_DIR/shard_$(printf '%04d' "$SHARD_INDEX")_of_$(printf '%04d' "$SHARD_COUNT").done"
