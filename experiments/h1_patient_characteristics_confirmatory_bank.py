"""Generate the full controlled patient-characteristics background bank.

This bank is used by the existing four-policy H1 search under the release-only
same-day reservation rule.

Behavioural profiles
--------------------
* no-show sensitivity: 2 Class 2 references x same/mild/strong Class 1
* balking sensitivity: 2 Class 2 references x same/mild/strong Class 1
* cancellation propensity: 1 Class 2 reference x same/mild/strong Class 1
* joint delay sensitivity: 2 Class 2 references x same/mild/strong Class 1

Clinic contexts are fully crossed:
* demand-to-capacity ratio: 1.2, 1.4, 1.7, 2.0, 3.0
* Class 1 share: 0.1, 0.3, 0.5, 0.7, 0.9
* daily capacity: 30, 50
* native booking horizon: 10, 14, 22

The native horizon ladder is designed to increase threshold exposure:
* 10 days activates the main Class 1 thresholds,
* 14 days also activates the moderate Class 2 thresholds,
* 22 days permits all low-sensitivity controls to cross.

This produces:
    21 profiles x 5 demand levels x 5 shares x 2 capacities x 3 horizons
    = 3,150 backgrounds.

The bank sets cap_thresholds_to_horizon=False. A policy that selects a short
booking horizon therefore does not mechanically redefine the patient profile.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_DIR
    / "outputs"
    / "hypotheses"
    / "h1_patient_characteristics_confirmatory_bank.csv"
)

RHO_VALUES = (1.2, 1.4, 1.7, 2.0, 3.0)
CLASS1_SHARE_VALUES = (0.1, 0.3, 0.5, 0.7, 0.9)
CAPACITY_VALUES = (30, 50)
NATIVE_HORIZON_VALUES = (10, 14, 22)

CONTRAST_ORDER = {"same": 0, "mild": 1, "strong": 2}


@dataclass(frozen=True)
class DelayProfile:
    threshold: int
    low: float
    high: float


@dataclass(frozen=True)
class PatientProfile:
    profile_id: str
    characteristic: str
    class2_reference: str
    contrast_level: str
    cancel_1: float
    cancel_2: float
    balk_1: DelayProfile
    balk_2: DelayProfile
    noshow_1: DelayProfile
    noshow_2: DelayProfile

    def __post_init__(self) -> None:
        if self.contrast_level not in CONTRAST_ORDER:
            raise ValueError(f"Unknown contrast level: {self.contrast_level}")

        for value in (self.cancel_1, self.cancel_2):
            if not 0 <= value <= 1:
                raise ValueError("Cancellation probabilities must lie in [0, 1].")

        for rule in (self.balk_1, self.balk_2, self.noshow_1, self.noshow_2):
            if rule.threshold < 0:
                raise ValueError("Thresholds must be nonnegative.")
            if not (0 <= rule.low < rule.high <= 1):
                raise ValueError(
                    "Each delay profile must satisfy 0 <= low < high <= 1."
                )
            if round(rule.low * 100) % 5 != 0:
                raise ValueError("Delay probabilities must be multiples of 5%.")
            if round(rule.high * 100) % 5 != 0:
                raise ValueError("Delay probabilities must be multiples of 5%.")
            if rule.high > 0.25:
                raise ValueError("Delay probabilities must not exceed 25%.")

        if self.balk_1.threshold <= self.noshow_1.threshold:
            raise ValueError(
                f"{self.profile_id}: Class 1 balking threshold must exceed "
                "its no-show threshold."
            )
        if self.balk_2.threshold <= self.noshow_2.threshold:
            raise ValueError(
                f"{self.profile_id}: Class 2 balking threshold must exceed "
                "its no-show threshold."
            )


def _profile(
    *,
    profile_id: str,
    characteristic: str,
    class2_reference: str,
    contrast_level: str,
    cancel_1: float,
    cancel_2: float,
    balk_1: tuple[int, float, float],
    balk_2: tuple[int, float, float],
    noshow_1: tuple[int, float, float],
    noshow_2: tuple[int, float, float],
) -> PatientProfile:
    return PatientProfile(
        profile_id=profile_id,
        characteristic=characteristic,
        class2_reference=class2_reference,
        contrast_level=contrast_level,
        cancel_1=cancel_1,
        cancel_2=cancel_2,
        balk_1=DelayProfile(*balk_1),
        balk_2=DelayProfile(*balk_2),
        noshow_1=DelayProfile(*noshow_1),
        noshow_2=DelayProfile(*noshow_2),
    )


def patient_profiles() -> list[PatientProfile]:
    profiles: list[PatientProfile] = []

    # 1. Class 1 more no-show sensitive.
    # Non-target behaviour: both classes use balking (16, 5%, 15%)
    # and cancellation 20%.
    noshow_references = {
        "low": {
            "class2": (14, 0.05, 0.15),
            "class1": {
                "same": (14, 0.05, 0.15),
                "mild": (5, 0.05, 0.20),
                "strong": (3, 0.05, 0.25),
            },
        },
        "moderate": {
            "class2": (8, 0.05, 0.20),
            "class1": {
                "same": (8, 0.05, 0.20),
                "mild": (5, 0.05, 0.20),
                "strong": (3, 0.05, 0.25),
            },
        },
    }
    fixed_balking = (16, 0.05, 0.15)
    for reference, values in noshow_references.items():
        for contrast, class1_noshow in values["class1"].items():
            profiles.append(
                _profile(
                    profile_id=f"NS_{reference.upper()}_{contrast.upper()}",
                    characteristic="no_show_sensitivity",
                    class2_reference=reference,
                    contrast_level=contrast,
                    cancel_1=0.20,
                    cancel_2=0.20,
                    balk_1=fixed_balking,
                    balk_2=fixed_balking,
                    noshow_1=class1_noshow,
                    noshow_2=values["class2"],
                )
            )

    # 2. Class 1 more balking sensitive.
    # Non-target behaviour: both classes use no-show (3, 5%, 10%)
    # and cancellation 20%.
    balking_references = {
        "low": {
            "class2": (16, 0.05, 0.15),
            "class1": {
                "same": (16, 0.05, 0.15),
                "mild": (6, 0.05, 0.20),
                "strong": (4, 0.05, 0.25),
            },
        },
        "moderate": {
            "class2": (9, 0.05, 0.20),
            "class1": {
                "same": (9, 0.05, 0.20),
                "mild": (6, 0.05, 0.20),
                "strong": (4, 0.05, 0.25),
            },
        },
    }
    fixed_noshow = (3, 0.05, 0.10)
    for reference, values in balking_references.items():
        for contrast, class1_balk in values["class1"].items():
            profiles.append(
                _profile(
                    profile_id=f"BK_{reference.upper()}_{contrast.upper()}",
                    characteristic="balking_sensitivity",
                    class2_reference=reference,
                    contrast_level=contrast,
                    cancel_1=0.20,
                    cancel_2=0.20,
                    balk_1=class1_balk,
                    balk_2=values["class2"],
                    noshow_1=fixed_noshow,
                    noshow_2=fixed_noshow,
                )
            )

    # 3. Class 1 more cancellation prone.
    # Retained unchanged from the controlled 420-background design.
    for contrast, cancel_1 in {
        "same": 0.10,
        "mild": 0.20,
        "strong": 0.30,
    }.items():
        profiles.append(
            _profile(
                profile_id=f"CN_LOW_{contrast.upper()}",
                characteristic="cancellation_propensity",
                class2_reference="low",
                contrast_level=contrast,
                cancel_1=cancel_1,
                cancel_2=0.10,
                balk_1=(16, 0.10, 0.20),
                balk_2=(16, 0.10, 0.20),
                noshow_1=(14, 0.10, 0.20),
                noshow_2=(14, 0.10, 0.20),
            )
        )

    # 4. Class 1 jointly more no-show and balking sensitive.
    joint_references = {
        "low": {
            "class2_noshow": (14, 0.05, 0.15),
            "class2_balk": (16, 0.05, 0.15),
            "class1": {
                "same": {
                    "noshow": (14, 0.05, 0.15),
                    "balk": (16, 0.05, 0.15),
                },
                "mild": {
                    "noshow": (5, 0.05, 0.20),
                    "balk": (6, 0.05, 0.20),
                },
                "strong": {
                    "noshow": (3, 0.05, 0.25),
                    "balk": (4, 0.05, 0.25),
                },
            },
        },
        "moderate": {
            "class2_noshow": (8, 0.05, 0.20),
            "class2_balk": (9, 0.05, 0.20),
            "class1": {
                "same": {
                    "noshow": (8, 0.05, 0.20),
                    "balk": (9, 0.05, 0.20),
                },
                "mild": {
                    "noshow": (5, 0.05, 0.20),
                    "balk": (6, 0.05, 0.20),
                },
                "strong": {
                    "noshow": (3, 0.05, 0.25),
                    "balk": (4, 0.05, 0.25),
                },
            },
        },
    }
    for reference, values in joint_references.items():
        for contrast, class1_values in values["class1"].items():
            profiles.append(
                _profile(
                    profile_id=f"JT_{reference.upper()}_{contrast.upper()}",
                    characteristic="joint_delay_sensitivity",
                    class2_reference=reference,
                    contrast_level=contrast,
                    cancel_1=0.20,
                    cancel_2=0.20,
                    balk_1=class1_values["balk"],
                    balk_2=values["class2_balk"],
                    noshow_1=class1_values["noshow"],
                    noshow_2=values["class2_noshow"],
                )
            )

    return sorted(
        profiles,
        key=lambda profile: (
            profile.characteristic,
            profile.class2_reference,
            CONTRAST_ORDER[profile.contrast_level],
        ),
    )


def clinic_contexts() -> pd.DataFrame:
    """Return the complete 5 x 5 x 2 x 3 clinic-context crossing."""
    rows: list[dict[str, object]] = []
    for context_number, (rho, share, capacity, horizon) in enumerate(
        product(
            RHO_VALUES,
            CLASS1_SHARE_VALUES,
            CAPACITY_VALUES,
            NATIVE_HORIZON_VALUES,
        ),
        start=1,
    ):
        rows.append(
            {
                "clinic_context_id": f"FC{context_number:03d}",
                "rho": float(rho),
                "class1_share": float(share),
                "slots_per_day": int(capacity),
                "horizon_days": int(horizon),
                "clinic_block": "full_factorial_5x5x2x3",
            }
        )
    return pd.DataFrame(rows)


def _profile_record(profile: PatientProfile) -> dict[str, object]:
    return {
        "patient_characteristic": profile.characteristic,
        "class2_reference": profile.class2_reference,
        "contrast_level": profile.contrast_level,
        "profile_id": profile.profile_id,
        "cancel_1": profile.cancel_1,
        "cancel_2": profile.cancel_2,
        "balk_threshold_1": profile.balk_1.threshold,
        "balk_low_1": profile.balk_1.low,
        "balk_high_1": profile.balk_1.high,
        "balk_threshold_2": profile.balk_2.threshold,
        "balk_low_2": profile.balk_2.low,
        "balk_high_2": profile.balk_2.high,
        "noshow_threshold_1": profile.noshow_1.threshold,
        "noshow_low_1": profile.noshow_1.low,
        "noshow_high_1": profile.noshow_1.high,
        "noshow_threshold_2": profile.noshow_2.threshold,
        "noshow_low_2": profile.noshow_2.low,
        "noshow_high_2": profile.noshow_2.high,
    }


def generate_full_bank(
    *,
    include_characteristics: Iterable[str] | None = None,
) -> pd.DataFrame:
    profiles = patient_profiles()

    if include_characteristics is not None:
        include = set(include_characteristics)
        known = {profile.characteristic for profile in profiles}
        unknown = include - known
        if unknown:
            raise ValueError(
                f"Unknown characteristics: {sorted(unknown)}; "
                f"choose from {sorted(known)}"
            )
        profiles = [
            profile
            for profile in profiles
            if profile.characteristic in include
        ]

    contexts = clinic_contexts()
    rows: list[dict[str, object]] = []

    for profile in profiles:
        profile_record = _profile_record(profile)
        for context in contexts.to_dict(orient="records"):
            background_id = (
                f"PCF_{profile.profile_id}_{context['clinic_context_id']}"
            )
            row: dict[str, object] = {
                "background_id": background_id,
                "design_note": "patient_characteristics_full_confirmatory",
                **profile_record,
                **context,
            }
            row["lambda_1"] = (
                float(row["rho"])
                * int(row["slots_per_day"])
                * float(row["class1_share"])
            )
            row["lambda_2"] = (
                float(row["rho"])
                * int(row["slots_per_day"])
                * (1.0 - float(row["class1_share"]))
            )
            row["cap_thresholds_to_horizon"] = False
            rows.append(row)

    bank = pd.DataFrame(rows)

    if bank["background_id"].duplicated().any():
        duplicates = bank.loc[
            bank["background_id"].duplicated(),
            "background_id",
        ].tolist()
        raise ValueError(f"Duplicate background IDs: {duplicates[:5]}")

    column_order = [
        "background_id",
        "design_note",
        "patient_characteristic",
        "class2_reference",
        "contrast_level",
        "profile_id",
        "clinic_context_id",
        "clinic_block",
        "horizon_days",
        "rho",
        "class1_share",
        "slots_per_day",
        "lambda_1",
        "lambda_2",
        "cancel_1",
        "cancel_2",
        "balk_threshold_1",
        "balk_low_1",
        "balk_high_1",
        "balk_threshold_2",
        "balk_low_2",
        "balk_high_2",
        "noshow_threshold_1",
        "noshow_low_1",
        "noshow_high_1",
        "noshow_threshold_2",
        "noshow_low_2",
        "noshow_high_2",
        "cap_thresholds_to_horizon",
    ]

    return (
        bank[column_order]
        .sort_values(
            [
                "patient_characteristic",
                "class2_reference",
                "contrast_level",
                "rho",
                "class1_share",
                "slots_per_day",
                "horizon_days",
            ],
            key=lambda series: (
                series.map(CONTRAST_ORDER)
                if series.name == "contrast_level"
                else series
            ),
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _validate_complete_design(bank: pd.DataFrame) -> None:
    expected_profiles = 21
    expected_contexts = (
        len(RHO_VALUES)
        * len(CLASS1_SHARE_VALUES)
        * len(CAPACITY_VALUES)
        * len(NATIVE_HORIZON_VALUES)
    )
    expected_backgrounds = expected_profiles * expected_contexts

    if bank["profile_id"].nunique() != expected_profiles:
        raise AssertionError(
            f"Expected {expected_profiles} profiles; "
            f"found {bank['profile_id'].nunique()}."
        )
    if bank["clinic_context_id"].nunique() != expected_contexts:
        raise AssertionError(
            f"Expected {expected_contexts} clinic contexts; "
            f"found {bank['clinic_context_id'].nunique()}."
        )
    if len(bank) != expected_backgrounds:
        raise AssertionError(
            f"Expected {expected_backgrounds:,} backgrounds; found {len(bank):,}."
        )

    per_profile = bank.groupby("profile_id")["clinic_context_id"].nunique()
    if not (per_profile == expected_contexts).all():
        raise AssertionError("Every profile must contain all clinic contexts.")

    context_columns = [
        "rho",
        "class1_share",
        "slots_per_day",
        "horizon_days",
    ]
    expected_combinations = set(
        product(
            RHO_VALUES,
            CLASS1_SHARE_VALUES,
            CAPACITY_VALUES,
            NATIVE_HORIZON_VALUES,
        )
    )
    actual_combinations = set(
        map(tuple, clinic_contexts()[context_columns].to_numpy())
    )
    if actual_combinations != expected_combinations:
        raise AssertionError("Clinic contexts are not fully crossed.")

    if not (~bank["cap_thresholds_to_horizon"].astype(bool)).all():
        raise AssertionError("All backgrounds must preserve patient thresholds.")


def _print_design_summary(bank: pd.DataFrame, output: Path) -> None:
    print(f"Full bank: {len(bank):,} backgrounds -> {output}")
    print(f"Patient profiles: {bank['profile_id'].nunique():,}")
    print(f"Clinic contexts: {bank['clinic_context_id'].nunique():,}")
    print("\nBackgrounds by behavioural family:")
    print(
        bank.groupby("patient_characteristic")
        .size()
        .rename("backgrounds")
        .to_string()
    )
    print("\nClinic-grid levels:")
    for column in (
        "rho",
        "class1_share",
        "slots_per_day",
        "horizon_days",
    ):
        values = sorted(bank[column].unique().tolist())
        print(f"  {column}: {values}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--characteristics",
        nargs="+",
        choices=(
            "no_show_sensitivity",
            "balking_sensitivity",
            "cancellation_propensity",
            "joint_delay_sensitivity",
        ),
        default=None,
        help="Optional behavioural-family subset for a reduced run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bank = generate_full_bank(
        include_characteristics=args.characteristics,
    )
    if args.characteristics is None:
        _validate_complete_design(bank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bank.to_csv(args.output, index=False)
    _print_design_summary(bank, args.output)


if __name__ == "__main__":
    main()
