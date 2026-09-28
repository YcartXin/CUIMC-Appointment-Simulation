#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
SOURCE_ROOT="${PBF_SOURCE_ROOT:-/scratch/$USER/patient_behavior_factorial_3_5}"
NEW_ROOT="${PBF_PARETO_ROOT:-/scratch/$USER/patient_behavior_factorial_served_rate_pareto}"

cd "$REPO"
SOURCE_RAW="$SOURCE_ROOT/search/raw"
TARGET_RAW="$NEW_ROOT/search/raw"

test -d "$SOURCE_RAW" || { echo "Missing legacy raw bank: $SOURCE_RAW" >&2; exit 1; }
mkdir -p "$TARGET_RAW"

for f in "$SOURCE_RAW"/*.csv; do
  dest="$TARGET_RAW/$(basename "$f")"
  if [[ ! -e "$dest" ]]; then
    cp --reflink=auto -p "$f" "$dest"
  fi
done

echo "Legacy search shards:"
find "$SOURCE_RAW" -maxdepth 1 -type f -name '*.csv' | wc -l
echo "New working search shards:"
find "$TARGET_RAW" -maxdepth 1 -type f -name '*.csv' | wc -l
echo "Working root: $NEW_ROOT"
