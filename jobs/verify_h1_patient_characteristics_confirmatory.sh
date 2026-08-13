#!/bin/bash
set -euo pipefail

# Verify raw-background completeness before finalization, or processed-output
# completeness after finalization.
#
# Usage:
#   bash jobs/verify_h1_patient_characteristics_confirmatory.sh [expected_seeds]

EXPECTED_SEEDS="${1:-10}"

REPO="${REPO:-$HOME/projects/CUIMC-Appointment-Simulation}"
PYTHON="${PYTHON:-$HOME/.conda/envs/cuimc/bin/python}"
BANK="$REPO/outputs/hypotheses/h1_patient_characteristics_confirmatory_bank.csv"
RAW_ROOT="${H1PC_CONFIRM_RAW_ROOT:-/scratch/$USER/h1_patient_characteristics_confirmatory}"
SUMMARY_ROOT="$REPO/full_run_summaries/h1_patient_characteristics_confirmatory/release"

cd "$REPO"

"$PYTHON" - "$BANK" "$RAW_ROOT" "$SUMMARY_ROOT" "$EXPECTED_SEEDS" <<'PY'
from pathlib import Path
import sys
import pandas as pd

bank_path = Path(sys.argv[1])
raw_root = Path(sys.argv[2])
summary_root = Path(sys.argv[3])
expected_seeds = int(sys.argv[4])

bank = pd.read_csv(bank_path)
expected_ids = set(bank["background_id"].astype(str))
raw_dir = raw_root / "release" / "raw"

if not raw_dir.exists():
    raise SystemExit(f"Raw directory does not exist: {raw_dir}")

raw_ids = {path.stem for path in raw_dir.glob("*.csv")}
missing = sorted(expected_ids - raw_ids)
extra = sorted(raw_ids - expected_ids)

print(f"Bank backgrounds: {len(expected_ids):,}")
print(f"Raw background files: {len(raw_ids):,}")
print(f"Missing raw backgrounds: {len(missing):,}")
print(f"Unexpected raw backgrounds: {len(extra):,}")

if missing:
    print("First missing IDs:", missing[:10])
if extra:
    print("First unexpected IDs:", extra[:10])

if missing or extra:
    raise SystemExit("Raw-background completeness check failed.")

if not summary_root.exists():
    print("Processed summary directory is not present yet.")
    print("Raw-background completeness check passed.")
    raise SystemExit(0)

expected_rows = {
    "selected_policy_outcomes.csv": 3150 * 2 * 4,
    "selected_policy_seed_outcomes.csv": 3150 * 2 * 4 * expected_seeds,
    "pairwise_group_deltas.csv": 3150 * 2 * 6,
    "objective_switch_deltas.csv": 3150 * 4,
    "selection_validation.csv": 0,
}

failed = False
for name, expected in expected_rows.items():
    path = summary_root / name
    if not path.exists():
        print(f"{name}: MISSING")
        failed = True
        continue

    frame = pd.read_csv(path)
    actual = len(frame)
    backgrounds = (
        frame["background_id"].nunique()
        if "background_id" in frame.columns
        else 0
    )
    print(
        f"{name}: rows={actual:,}, "
        f"expected={expected:,}, backgrounds={backgrounds:,}"
    )
    if actual != expected:
        failed = True
    if name != "selection_validation.csv" and backgrounds != 3150:
        failed = True

if failed:
    raise SystemExit("Processed-output completeness check failed.")

print("All full-run completeness checks passed.")
PY
