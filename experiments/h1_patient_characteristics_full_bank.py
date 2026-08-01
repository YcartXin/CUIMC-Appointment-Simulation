"""Generate the confirmatory controlled patient-characteristics bank.

The design fully crosses 21 behavioural profiles with 180 clinic contexts:
6 demand ratios x 5 Class-1 shares x 2 capacities x 3 native horizons.
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
    REPO_DIR / "outputs" / "hypotheses" / "h1_patient_characteristics_full_bank.csv"
)

RHO_VALUES = (1.2, 1.4, 1.7, 2.0, 2.5, 3.0)
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
            raise ValueError(f"Unknown contrast: {self.contrast_level}")
        for value in (self.cancel_1, self.cancel_2):
            if not 0 <= value <= 1:
                raise ValueError("Cancellation probabilities must lie in [0, 1].")
        for rule in (self.balk_1, self.balk_2, self.noshow_1, self.noshow_2):
            if rule.threshold < 0 or not (0 <= rule.low < rule.high <= 1):
                raise ValueError(f"Invalid threshold rule: {rule}")
            if rule.high > 0.25:
                raise ValueError(f"Delay-dependent probabilities may not exceed 0.25: {rule}")
        if self.balk_1.threshold <= self.noshow_1.threshold:
            raise ValueError(f"{self.profile_id}: Class 1 balk threshold must exceed no-show threshold")
        if self.balk_2.threshold <= self.noshow_2.threshold:
            raise ValueError(f"{self.profile_id}: Class 2 balk threshold must exceed no-show threshold")


def _delay(values: tuple[int, float, float]) -> DelayProfile:
    return DelayProfile(*values)


def _profile(
    profile_id: str,
    characteristic: str,
    class2_reference: str,
    contrast_level: str,
    *,
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
        balk_1=_delay(balk_1),
        balk_2=_delay(balk_2),
        noshow_1=_delay(noshow_1),
        noshow_2=_delay(noshow_2),
    )


def patient_profiles() -> list[PatientProfile]:
    profiles: list[PatientProfile] = []

    # A. No-show sensitivity; balking and cancellation held fixed.
    noshow_refs = {
        "low": (14, 0.05, 0.15),
        "moderate": (8, 0.05, 0.20),
    }
    c1_noshow = {
        "mild": (5, 0.05, 0.20),
        "strong": (3, 0.05, 0.25),
    }
    for reference, class2_rule in noshow_refs.items():
        for contrast in ("same", "mild", "strong"):
            class1_rule = class2_rule if contrast == "same" else c1_noshow[contrast]
            profiles.append(
                _profile(
                    f"NS_{reference.upper()}_{contrast.upper()}",
                    "no_show_sensitivity",
                    reference,
                    contrast,
                    cancel_1=0.20,
                    cancel_2=0.20,
                    balk_1=(16, 0.05, 0.15),
                    balk_2=(16, 0.05, 0.15),
                    noshow_1=class1_rule,
                    noshow_2=class2_rule,
                )
            )

    # B. Balking sensitivity; no-show and cancellation held fixed.
    balk_refs = {
        "low": (16, 0.05, 0.15),
        "moderate": (9, 0.05, 0.20),
    }
    c1_balk = {
        "mild": (6, 0.05, 0.20),
        "strong": (4, 0.05, 0.25),
    }
    for reference, class2_rule in balk_refs.items():
        for contrast in ("same", "mild", "strong"):
            class1_rule = class2_rule if contrast == "same" else c1_balk[contrast]
            profiles.append(
                _profile(
                    f"BK_{reference.upper()}_{contrast.upper()}",
                    "balking_sensitivity",
                    reference,
                    contrast,
                    cancel_1=0.20,
                    cancel_2=0.20,
                    balk_1=class1_rule,
                    balk_2=class2_rule,
                    noshow_1=(3, 0.05, 0.10),
                    noshow_2=(3, 0.05, 0.10),
                )
            )

    # C. Cancellation propensity. Shared low-reference delay rules are explicit.
    for contrast, cancel_1 in (("same", 0.10), ("mild", 0.20), ("strong", 0.30)):
        profiles.append(
            _profile(
                f"CN_LOW_{contrast.upper()}",
                "cancellation_propensity",
                "low",
                contrast,
                cancel_1=cancel_1,
                cancel_2=0.10,
                balk_1=(16, 0.05, 0.15),
                balk_2=(16, 0.05, 0.15),
                noshow_1=(14, 0.05, 0.15),
                noshow_2=(14, 0.05, 0.15),
            )
        )

    # D. Joint no-show and balking sensitivity.
    joint_refs = {
        "low": {
            "noshow": (14, 0.05, 0.15),
            "balk": (16, 0.05, 0.15),
        },
        "moderate": {
            "noshow": (8, 0.05, 0.20),
            "balk": (9, 0.05, 0.20),
        },
    }
    c1_joint = {
        "mild": {"noshow": (5, 0.05, 0.20), "balk": (6, 0.05, 0.20)},
        "strong": {"noshow": (3, 0.05, 0.25), "balk": (4, 0.05, 0.25)},
    }
    for reference, class2_rules in joint_refs.items():
        for contrast in ("same", "mild", "strong"):
            class1_rules = class2_rules if contrast == "same" else c1_joint[contrast]
            profiles.append(
                _profile(
                    f"JT_{reference.upper()}_{contrast.upper()}",
                    "joint_delay_sensitivity",
                    reference,
                    contrast,
                    cancel_1=0.20,
                    cancel_2=0.20,
                    balk_1=class1_rules["balk"],
                    balk_2=class2_rules["balk"],
                    noshow_1=class1_rules["noshow"],
                    noshow_2=class2_rules["noshow"],
                )
            )

    profiles.sort(
        key=lambda p: (
            p.characteristic,
            p.class2_reference,
            CONTRAST_ORDER[p.contrast_level],
        )
    )
    if len(profiles) != 21:
        raise AssertionError(f"Expected 21 profiles, generated {len(profiles)}")
    return profiles


def clinic_contexts() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for number, (rho, share, capacity, horizon) in enumerate(
        product(RHO_VALUES, CLASS1_SHARE_VALUES, CAPACITY_VALUES, NATIVE_HORIZON_VALUES),
        start=1,
    ):
        rows.append(
            {
                "clinic_context_id": f"C{number:03d}",
                "rho": rho,
                "class1_share": share,
                "slots_per_day": capacity,
                "horizon_days": horizon,
                "clinic_block": "fully_crossed_confirmatory",
            }
        )
    contexts = pd.DataFrame(rows)
    if len(contexts) != 180:
        raise AssertionError(f"Expected 180 contexts, generated {len(contexts)}")
    return contexts


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


def generate_bank(
    *,
    include_characteristics: Iterable[str] | None = None,
    smoke: bool = False,
) -> pd.DataFrame:
    profiles = patient_profiles()
    if include_characteristics is not None:
        requested = set(include_characteristics)
        known = {profile.characteristic for profile in profiles}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown characteristics: {sorted(unknown)}")
        profiles = [profile for profile in profiles if profile.characteristic in requested]

    contexts = clinic_contexts()
    if smoke:
        # One matched same/strong pair in two different clinic contexts.
        profiles = [
            profile
            for profile in profiles
            if profile.profile_id in {"NS_LOW_SAME", "NS_LOW_STRONG"}
        ]
        contexts = contexts.iloc[[0, -1]].copy()

    rows: list[dict[str, object]] = []
    for profile in profiles:
        profile_record = _profile_record(profile)
        for context in contexts.to_dict(orient="records"):
            row: dict[str, object] = {
                "background_id": f"PCF_{profile.profile_id}_{context['clinic_context_id']}",
                "design_note": "patient_characteristics_full_confirmatory",
                **profile_record,
                **context,
            }
            row["lambda_1"] = float(row["rho"]) * int(row["slots_per_day"]) * float(row["class1_share"])
            row["lambda_2"] = float(row["rho"]) * int(row["slots_per_day"]) * (1.0 - float(row["class1_share"]))
            row["cap_thresholds_to_horizon"] = False
            rows.append(row)

    bank = pd.DataFrame(rows)
    if bank["background_id"].duplicated().any():
        raise ValueError("Generated duplicate background IDs")
    if not smoke and include_characteristics is None and len(bank) != 3780:
        raise AssertionError(f"Expected 3,780 backgrounds, generated {len(bank)}")

    order = [
        "background_id", "design_note", "patient_characteristic", "class2_reference",
        "contrast_level", "profile_id", "clinic_context_id", "clinic_block",
        "horizon_days", "rho", "class1_share", "slots_per_day", "lambda_1", "lambda_2",
        "cancel_1", "cancel_2", "balk_threshold_1", "balk_low_1", "balk_high_1",
        "balk_threshold_2", "balk_low_2", "balk_high_2", "noshow_threshold_1",
        "noshow_low_1", "noshow_high_1", "noshow_threshold_2", "noshow_low_2",
        "noshow_high_2", "cap_thresholds_to_horizon",
    ]
    return bank[order].sort_values(
        [
            "patient_characteristic", "class2_reference", "contrast_level",
            "rho", "class1_share", "slots_per_day", "horizon_days",
        ],
        key=lambda series: series.map(CONTRAST_ORDER) if series.name == "contrast_level" else series,
        kind="stable",
    ).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--characteristics",
        nargs="+",
        choices=(
            "no_show_sensitivity", "balking_sensitivity",
            "cancellation_propensity", "joint_delay_sensitivity",
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bank = generate_bank(
        include_characteristics=args.characteristics,
        smoke=args.smoke,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bank.to_csv(args.output, index=False)
    print(f"Wrote {len(bank):,} backgrounds to {args.output}")
    print(f"Profiles: {bank['profile_id'].nunique():,}")
    print(f"Clinic contexts: {bank['clinic_context_id'].nunique():,}")


if __name__ == "__main__":
    main()
