"""Generate the controlled patient-characteristics pilot bank.

The pilot reuses the existing four-policy H1 search, but replaces the broad
Sobol background bank with controlled Class 1 versus Class 2 profiles.

Design
------
Patient profiles:
* no-show sensitivity: 2 Class 2 references x same/mild/strong Class 1
* balking sensitivity: 2 Class 2 references x same/mild/strong Class 1
* cancellation propensity: 1 Class 2 reference x same/mild/strong Class 1
* joint delay sensitivity: 2 Class 2 references x same/mild/strong Class 1

Clinic contexts:
* demand-to-capacity ratio: 1.0, 1.6, 2.0, 3.0
* Class 1 share: 0.1, 0.3, 0.5, 0.7, 0.9
* one balanced capacity/native-horizon block across the 20 rho-share cells

This produces 21 patient profiles x 20 clinic contexts = 420 backgrounds.
Priority-group share is not a separate profile row; it is crossed with every
patient profile and is therefore available as a fifth characteristic axis.

The bank sets cap_thresholds_to_horizon=False. When a policy chooses a horizon
shorter than a patient's threshold, every feasible offer remains in the
pre-threshold region rather than mechanically changing the patient's profile.
Existing banks omit this column and retain the historical capping behavior.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_DIR
    / "outputs"
    / "hypotheses"
    / "h1_patient_characteristics_pilot_bank.csv"
)

RHO_VALUES = (1.0, 1.6, 2.0, 3.0)
CLASS1_SHARE_VALUES = (0.1, 0.3, 0.5, 0.7, 0.9)
CAPACITY_VALUES = (30, 50)
NATIVE_HORIZON_VALUES = (6, 14, 22)

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
    # Balking is held equal and late enough not to pre-empt the no-show contrast.
    noshow_reference_profiles = {
        "low": {
            "class2": (20, 0.05, 0.10),
            "class1": {
                "same": (20, 0.05, 0.10),
                "mild": (14, 0.10, 0.20),
                "strong": (7, 0.20, 0.30),
            },
        },
        "moderate": {
            "class2": (14, 0.10, 0.20),
            "class1": {
                "same": (14, 0.10, 0.20),
                "mild": (7, 0.20, 0.30),
                "strong": (4, 0.30, 0.40),
            },
        },
    }
    for reference, values in noshow_reference_profiles.items():
        for contrast, class1_noshow in values["class1"].items():
            profiles.append(
                _profile(
                    profile_id=f"NS_{reference.upper()}_{contrast.upper()}",
                    characteristic="no_show_sensitivity",
                    class2_reference=reference,
                    contrast_level=contrast,
                    cancel_1=0.20,
                    cancel_2=0.20,
                    balk_1=(22, 0.10, 0.20),
                    balk_2=(22, 0.10, 0.20),
                    noshow_1=class1_noshow,
                    noshow_2=values["class2"],
                )
            )

    # 2. Class 1 more balking sensitive.
    # A low, shared no-show profile keeps the targeted difference in balking.
    balk_reference_profiles = {
        "low": {
            "class2": (22, 0.05, 0.10),
            "class1": {
                "same": (22, 0.05, 0.10),
                "mild": (16, 0.10, 0.20),
                "strong": (9, 0.20, 0.30),
            },
        },
        "moderate": {
            "class2": (16, 0.10, 0.20),
            "class1": {
                "same": (16, 0.10, 0.20),
                "mild": (9, 0.20, 0.30),
                "strong": (5, 0.30, 0.40),
            },
        },
    }
    for reference, values in balk_reference_profiles.items():
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
                    noshow_1=(4, 0.05, 0.10),
                    noshow_2=(4, 0.05, 0.10),
                )
            )

    # 3. Class 1 more cancellation prone.
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
    joint_reference_profiles = {
        "low": {
            "class2_noshow": (20, 0.05, 0.10),
            "class2_balk": (22, 0.05, 0.10),
            "class1": {
                "same": {
                    "noshow": (20, 0.05, 0.10),
                    "balk": (22, 0.05, 0.10),
                },
                "mild": {
                    "noshow": (14, 0.10, 0.20),
                    "balk": (16, 0.10, 0.20),
                },
                "strong": {
                    "noshow": (7, 0.20, 0.30),
                    "balk": (9, 0.20, 0.30),
                },
            },
        },
        "moderate": {
            "class2_noshow": (14, 0.10, 0.20),
            "class2_balk": (16, 0.10, 0.20),
            "class1": {
                "same": {
                    "noshow": (14, 0.10, 0.20),
                    "balk": (16, 0.10, 0.20),
                },
                "mild": {
                    "noshow": (7, 0.20, 0.30),
                    "balk": (9, 0.20, 0.30),
                },
                "strong": {
                    "noshow": (4, 0.30, 0.40),
                    "balk": (5, 0.30, 0.40),
                },
            },
        },
    }
    for reference, values in joint_reference_profiles.items():
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

    profiles.sort(
        key=lambda p: (
            p.characteristic,
            p.class2_reference,
            CONTRAST_ORDER[p.contrast_level],
        )
    )
    return profiles


def clinic_contexts() -> pd.DataFrame:
    """Return one balanced capacity/horizon block over all rho-share pairs."""
    rows = []
    context_number = 0
    for rho_index, rho in enumerate(RHO_VALUES):
        for share_index, share in enumerate(CLASS1_SHARE_VALUES):
            context_number += 1
            capacity = CAPACITY_VALUES[(rho_index + share_index) % 2]
            horizon = NATIVE_HORIZON_VALUES[(2 * rho_index + share_index) % 3]
            rows.append(
                {
                    "clinic_context_id": f"C{context_number:02d}",
                    "rho": rho,
                    "class1_share": share,
                    "slots_per_day": capacity,
                    "horizon_days": horizon,
                    "clinic_block": "compact_balanced_block_1",
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


def generate_pilot_bank(
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
            profile for profile in profiles if profile.characteristic in include
        ]

    contexts = clinic_contexts()
    rows: list[dict[str, object]] = []
    for profile in profiles:
        profile_record = _profile_record(profile)
        for context in contexts.to_dict(orient="records"):
            background_id = (
                f"PCP_{profile.profile_id}_{context['clinic_context_id']}"
            )
            row = {
                "background_id": background_id,
                "design_note": "patient_characteristics_pilot",
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
                * (1 - float(row["class1_share"]))
            )
            row["cap_thresholds_to_horizon"] = False
            rows.append(row)

    bank = pd.DataFrame(rows)
    if bank["background_id"].duplicated().any():
        duplicates = bank.loc[
            bank["background_id"].duplicated(), "background_id"
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
    return bank[column_order].sort_values(
        [
            "patient_characteristic",
            "class2_reference",
            "contrast_level",
            "rho",
            "class1_share",
        ],
        key=lambda series: (
            series.map(CONTRAST_ORDER)
            if series.name == "contrast_level"
            else series
        ),
        kind="stable",
    ).reset_index(drop=True)


def _print_design_summary(bank: pd.DataFrame, output: Path) -> None:
    print(f"Pilot bank: {len(bank):,} backgrounds -> {output}")
    print(
        bank.groupby(
            ["patient_characteristic", "class2_reference", "contrast_level"]
        )
        .size()
        .rename("backgrounds")
        .to_string()
    )
    print("\nClinic context coverage:")
    print(
        bank[
            [
                "clinic_context_id",
                "rho",
                "class1_share",
                "slots_per_day",
                "horizon_days",
            ]
        ]
        .drop_duplicates()
        .sort_values("clinic_context_id")
        .to_string(index=False)
    )
    print("\nCapacity counts across the 20 clinic contexts:")
    print(
        bank[
            ["clinic_context_id", "slots_per_day"]
        ]
        .drop_duplicates()["slots_per_day"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print("\nNative-horizon counts across the 20 clinic contexts:")
    print(
        bank[
            ["clinic_context_id", "horizon_days"]
        ]
        .drop_duplicates()["horizon_days"]
        .value_counts()
        .sort_index()
        .to_string()
    )


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
        help=(
            "Optional subset for a smaller first run. Priority-group share "
            "is crossed with every selected characteristic."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bank = generate_pilot_bank(
        include_characteristics=args.characteristics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bank.to_csv(args.output, index=False)
    _print_design_summary(bank, args.output)


if __name__ == "__main__":
    main()
