from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_DIR / "outputs" / "hypotheses" / "patient_behavior_factorial_bank.csv"

RHO_VALUES = (1.2, 1.4, 1.7, 2.0, 2.5, 3.0)
CLASS1_SHARE_VALUES = (0.1, 0.3, 0.5, 0.7, 0.9)
CAPACITY_VALUES = (30, 50)
OPEN_HORIZON_DAYS = 100

LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}

# Class 1: thresholds held fixed; only post-threshold probabilities vary.
C1_NOSHOW_THRESHOLD = 5
C1_NOSHOW_PRE = 0.05
C1_NOSHOW_POST = {"low": 0.05, "medium": 0.15, "high": 0.25}

C1_BALK_THRESHOLD = 7
C1_BALK_PRE = 0.05
C1_BALK_POST = {"low": 0.10, "medium": 0.20, "high": 0.30}

# Class 2: fixed, later thresholds so delay sensitivity is less easily triggered.
C2_NOSHOW = (14, 0.05, 0.15)
C2_BALK = (16, 0.05, 0.15)

CANCEL_1 = 0.20
CANCEL_2 = 0.20


def _profiles() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for noshow_level, balk_level in product(LEVEL_ORDER, LEVEL_ORDER):
        rows.append(
            {
                "profile_id": f"NS_{noshow_level.upper()}_BK_{balk_level.upper()}",
                "noshow_level": noshow_level,
                "noshow_level_code": LEVEL_ORDER[noshow_level],
                "balk_level": balk_level,
                "balk_level_code": LEVEL_ORDER[balk_level],
                "cancel_1": CANCEL_1,
                "cancel_2": CANCEL_2,
                "noshow_threshold_1": C1_NOSHOW_THRESHOLD,
                "noshow_low_1": C1_NOSHOW_PRE,
                "noshow_high_1": C1_NOSHOW_POST[noshow_level],
                "balk_threshold_1": C1_BALK_THRESHOLD,
                "balk_low_1": C1_BALK_PRE,
                "balk_high_1": C1_BALK_POST[balk_level],
                "noshow_threshold_2": C2_NOSHOW[0],
                "noshow_low_2": C2_NOSHOW[1],
                "noshow_high_2": C2_NOSHOW[2],
                "balk_threshold_2": C2_BALK[0],
                "balk_low_2": C2_BALK[1],
                "balk_high_2": C2_BALK[2],
            }
        )
    return rows


def _contexts() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, (rho, share, capacity) in enumerate(
        product(RHO_VALUES, CLASS1_SHARE_VALUES, CAPACITY_VALUES), start=1
    ):
        rows.append(
            {
                "clinic_context_id": f"C{number:03d}",
                "rho": rho,
                "class1_share": share,
                "slots_per_day": capacity,
                # Constant 100-day horizon is the effectively-open horizon used
                # by baseline and reservation-only. It is NOT a design factor.
                "horizon_days": OPEN_HORIZON_DAYS,
                "lambda_1": rho * capacity * share,
                "lambda_2": rho * capacity * (1.0 - share),
            }
        )
    return rows


def generate_bank() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for profile in _profiles():
        for context in _contexts():
            rows.append(
                {
                    "background_id": f"PBF_{profile['profile_id']}_{context['clinic_context_id']}",
                    "design_note": "class1_balk_noshow_3x3_factorial",
                    "patient_characteristic": "balk_noshow_factorial",
                    "class2_reference": "fixed_late_thresholds",
                    **profile,
                    **context,
                    "cap_thresholds_to_horizon": False,
                }
            )

    bank = pd.DataFrame(rows)
    if len(bank) != 540:
        raise AssertionError(f"Expected 540 backgrounds, generated {len(bank)}")
    if bank["background_id"].duplicated().any():
        raise ValueError("Generated duplicate background IDs")
    if bank["profile_id"].nunique() != 9:
        raise AssertionError("Expected 9 behavioral profiles")
    if bank["clinic_context_id"].nunique() != 60:
        raise AssertionError("Expected 60 clinic contexts")
    if not (bank["balk_threshold_1"] - bank["noshow_threshold_1"] >= 2).all():
        raise AssertionError("Class 1 balk threshold must be at least 2 days later")

    order = [
        "background_id", "design_note", "patient_characteristic", "class2_reference",
        "profile_id", "noshow_level", "noshow_level_code", "balk_level", "balk_level_code",
        "clinic_context_id", "horizon_days", "rho", "class1_share", "slots_per_day",
        "lambda_1", "lambda_2", "cancel_1", "cancel_2",
        "balk_threshold_1", "balk_low_1", "balk_high_1",
        "balk_threshold_2", "balk_low_2", "balk_high_2",
        "noshow_threshold_1", "noshow_low_1", "noshow_high_1",
        "noshow_threshold_2", "noshow_low_2", "noshow_high_2",
        "cap_thresholds_to_horizon",
    ]
    return bank[order].sort_values(
        ["noshow_level_code", "balk_level_code", "rho", "class1_share", "slots_per_day"],
        kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    bank = generate_bank()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bank.to_csv(args.output, index=False)
    print(f"Wrote {len(bank):,} backgrounds to {args.output}")
    print(f"Behavior cells: {bank['profile_id'].nunique():,}")
    print(f"Clinic contexts: {bank['clinic_context_id'].nunique():,}")


if __name__ == "__main__":
    main()
