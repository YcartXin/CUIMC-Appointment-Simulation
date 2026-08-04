#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

# Usage: ./run_h1_average_refine.sh <variant> <shard_index> <shard_count> <workers>
VARIANT="$1"
SHARD_INDEX="$2"
SHARD_COUNT="$3"
WORKERS="$4"

REPO="$HOME/projects/CUIMC-Appointment-Simulation"
PYTHON="$HOME/.conda/envs/cuimc/bin/python"
BANK="$REPO/outputs/hypotheses/background_scenarios.csv"
OUTPUT="/scratch/$USER/h1_short_horizon_reservation_10seed_v2"

cd "$REPO"

test -x "$PYTHON" || {
    echo "Python environment not found: $PYTHON" >&2
    exit 1
}

test -f "$BANK" || {
    echo "Scenario bank not found: $BANK" >&2
    exit 1
}

test -d "$OUTPUT/$VARIANT/raw" || {
    echo "Existing raw output not found: $OUTPUT/$VARIANT/raw" >&2
    exit 1
}

exec "$PYTHON" -u experiments/h1_short_horizon_reservation.py run \
    --variant "$VARIANT" \
    --objective average_utilization \
    --bank "$BANK" \
    --output-dir "$OUTPUT" \
    --n-seeds 10 \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    --workers "$WORKERS"
