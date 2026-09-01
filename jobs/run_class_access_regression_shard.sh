#!/usr/bin/env bash
set -euo pipefail

# Generic helper for an SGE-style array job.
# SGE_TASK_ID is 1-based; the Python experiment uses 0-based shard indices.
SHARD_COUNT="${SHARD_COUNT:-20}"
TASK_ID="${SGE_TASK_ID:?SGE_TASK_ID must be set by the array scheduler}"
SHARD_INDEX=$((TASK_ID - 1))
WORKERS="${NSLOTS:-1}"

# Explicitly use the cuimc Python 3.11 environment on the research grid.
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"

"$PYTHON" scripts/run_class_access_regression_robustness.py \
  --mode run \
  --shard-count "${SHARD_COUNT}" \
  --shard-index "${SHARD_INDEX}" \
  --workers "${WORKERS}"
