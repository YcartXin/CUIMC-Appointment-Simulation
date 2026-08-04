"""Baseline-only threshold-exposure audit for controlled patient profiles.

This audit is deliberately smaller than the policy optimization. It checks
whether offered appointment delays actually fall on both sides of the proposed
balking and no-show thresholds before the complete policy run is launched.

Design
------
Profiles:
* 6 no-show sensitivity profiles
* 6 balking sensitivity profiles
* 6 joint delay-sensitivity profiles

Clinic contexts (fully crossed):
* demand-to-capacity ratio: 1.2, 1.7, 3.0
* Class 1 share: 0.1, 0.5, 0.9
* daily capacity: 30, 50
* native booking horizon: 6, 14, 22

This gives 18 profiles x 54 clinic contexts = 972 backgrounds. The default
uses two seeds, for 1,944 baseline simulations.

The audit records:
* share of real offers above each class's balking threshold
* share of accepted bookings above each class's no-show threshold
* counts on both sides of each threshold
* offered- and accepted-delay quantiles
* realized balking, no-show, cancellation, no-offer, and served rates

Thresholds are not capped to the active horizon. A context is separately
marked as structurally crossable when horizon_days - 1 exceeds the threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from simulation.engine import ClinicAppointmentSimulation  # noqa: E402
from simulation.model import (  # noqa: E402
    Booking,
    PatientClassParams,
    SimulationConfig,
    ThresholdRule,
)

RHO_VALUES = (1.2, 1.7, 3.0)
CLASS1_SHARE_VALUES = (0.1, 0.5, 0.9)
CAPACITY_VALUES = (30, 50)
NATIVE_HORIZON_VALUES = (6, 14, 22)
SEEDS = tuple(range(1000, 1020))

BURN_IN_DAYS = 30
MEASURE_DAYS = 365
# The policy search can test horizons up to 26 days. Using 26 here ensures
# that measurement-window bookings have time to resolve during this audit.
COOLDOWN_DAYS = 26

DEFAULT_BANK = (
    REPO_DIR
    / "outputs"
    / "hypotheses"
    / "h1_threshold_exposure_audit_bank.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_DIR
    / "outputs"
    / "hypotheses"
    / "h1_threshold_exposure_audit"
)

CONTRAST_ORDER = {"same": 0, "mild": 1, "strong": 2}


@dataclass(frozen=True)
class DelayProfile:
    threshold: int
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.threshold < 0:
            raise ValueError("Thresholds must be nonnegative.")
        if not 0 <= self.low < self.high <= 1:
            raise ValueError("Each profile must satisfy 0 <= low < high <= 1.")


@dataclass(frozen=True)
class AuditProfile:
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
) -> AuditProfile:
    return AuditProfile(
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


def patient_profiles() -> list[AuditProfile]:
    """Return the proposed no-show, balking, and joint profiles."""

    profiles: list[AuditProfile] = []

    noshow_references = {
        "low": {
            "class2": (14, 0.05, 0.15),
            "class1": {
                "same": (14, 0.05, 0.15),
                "mild": (10, 0.05, 0.20),
                "strong": (7, 0.05, 0.25),
            },
        },
        "moderate": {
            "class2": (7, 0.05, 0.20),
            "class1": {
                "same": (7, 0.05, 0.20),
                "mild": (5, 0.05, 0.25),
                "strong": (3, 0.05, 0.25),
            },
        },
    }
    fixed_balk = (16, 0.05, 0.15)
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
                    balk_1=fixed_balk,
                    balk_2=fixed_balk,
                    noshow_1=class1_noshow,
                    noshow_2=values["class2"],
                )
            )

    balk_references = {
        "low": {
            "class2": (16, 0.05, 0.15),
            "class1": {
                "same": (16, 0.05, 0.15),
                "mild": (12, 0.05, 0.20),
                "strong": (9, 0.05, 0.25),
            },
        },
        "moderate": {
            "class2": (9, 0.05, 0.20),
            "class1": {
                "same": (9, 0.05, 0.20),
                "mild": (6, 0.05, 0.25),
                "strong": (4, 0.05, 0.25),
            },
        },
    }
    # This fixed no-show rule is deliberately below every proposed balking
    # threshold. The prior suggestion of a 14-day fixed no-show threshold
    # would violate the repository's profile-ordering validation for the
    # 4-, 6-, 9-, and 12-day balking profiles.
    fixed_noshow = (3, 0.05, 0.10)
    for reference, values in balk_references.items():
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
                    "noshow": (10, 0.05, 0.20),
                    "balk": (12, 0.05, 0.20),
                },
                "strong": {
                    "noshow": (7, 0.05, 0.25),
                    "balk": (9, 0.05, 0.25),
                },
            },
        },
        "moderate": {
            "class2_noshow": (7, 0.05, 0.20),
            "class2_balk": (9, 0.05, 0.20),
            "class1": {
                "same": {
                    "noshow": (7, 0.05, 0.20),
                    "balk": (9, 0.05, 0.20),
                },
                "mild": {
                    "noshow": (5, 0.05, 0.25),
                    "balk": (6, 0.05, 0.25),
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
    rows: list[dict[str, Any]] = []
    context_number = 0
    for rho in RHO_VALUES:
        for share in CLASS1_SHARE_VALUES:
            for capacity in CAPACITY_VALUES:
                for horizon in NATIVE_HORIZON_VALUES:
                    context_number += 1
                    rows.append(
                        {
                            "clinic_context_id": f"A{context_number:03d}",
                            "rho": rho,
                            "class1_share": share,
                            "slots_per_day": capacity,
                            "horizon_days": horizon,
                            "lambda_1": rho * capacity * share,
                            "lambda_2": rho * capacity * (1 - share),
                        }
                    )
    return pd.DataFrame(rows)


def _profile_record(profile: AuditProfile) -> dict[str, Any]:
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


def generate_bank() -> pd.DataFrame:
    contexts = clinic_contexts()
    rows: list[dict[str, Any]] = []
    for profile in patient_profiles():
        profile_record = _profile_record(profile)
        for context in contexts.to_dict(orient="records"):
            rows.append(
                {
                    "background_id": (
                        f"AUD_{profile.profile_id}_{context['clinic_context_id']}"
                    ),
                    "design_note": "threshold_exposure_audit",
                    **profile_record,
                    **context,
                    "cap_thresholds_to_horizon": False,
                }
            )
    bank = pd.DataFrame(rows)
    if len(bank) != 972:
        raise AssertionError(f"Expected 972 backgrounds, found {len(bank)}")
    if bank["background_id"].duplicated().any():
        raise ValueError("Audit background IDs must be unique.")
    return bank.sort_values(
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
    ).reset_index(drop=True)


def write_bank(path: Path) -> pd.DataFrame:
    bank = generate_bank()
    path.parent.mkdir(parents=True, exist_ok=True)
    bank.to_csv(path, index=False)
    print(f"Audit bank: {len(bank):,} backgrounds -> {path}")
    return bank


class ThresholdExposureSimulation(ClinicAppointmentSimulation):
    """Clinic simulation with measurement-window offered-delay histograms."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        self.audit_offered_delay_counts: dict[int, Counter[int]] = {
            class_id: Counter() for class_id in config.classes
        }

    def process_daily_arrivals(
        self,
        ordered_arrivals: list[int],
        track_patients: bool,
    ) -> None:
        if any(
            params.standby_prob != 0.0
            for params in self.config.classes.values()
        ):
            raise ValueError("The threshold audit assumes standby is disabled.")

        if track_patients:
            for class_id in ordered_arrivals:
                self.class_metrics[class_id].arrivals += 1

        for class_id in ordered_arrivals:
            params = self.config.classes[class_id]
            metrics = self.class_metrics[class_id]
            offer = self.find_earliest_open_day(class_id)

            if offer is None:
                if track_patients:
                    metrics.no_offer += 1
                continue

            offered_day, reserved_slot = offer
            tau = offered_day

            if track_patients:
                self.audit_offered_delay_counts[class_id][tau] += 1

            if self.rng.random() < params.balk_prob(tau):
                if track_patients:
                    metrics.total_offered_booking_delay += tau
                    metrics.balked += 1
                continue

            self.calendar[offered_day].append(
                Booking(
                    patient_class=class_id,
                    booking_delay=tau,
                    tracked=track_patients,
                    reserved_slot=reserved_slot,
                )
            )
            if track_patients:
                metrics.total_offered_booking_delay += tau
                metrics.booked += 1
                metrics.total_booking_delay += tau
                metrics.accepted_delay_counts[tau] = (
                    metrics.accepted_delay_counts.get(tau, 0) + 1
                )


def _safe_rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _hist_quantile(counts: Mapping[int, int], q: float) -> float:
    total = int(sum(int(value) for value in counts.values()))
    if total <= 0:
        return math.nan
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1].")

    sorted_items = sorted((int(delay), int(count)) for delay, count in counts.items())

    def value_at_rank(rank: int) -> int:
        cumulative = 0
        for delay, count in sorted_items:
            cumulative += count
            if rank < cumulative:
                return delay
        return sorted_items[-1][0]

    position = q * (total - 1)
    lower_rank = int(math.floor(position))
    upper_rank = int(math.ceil(position))
    lower_value = value_at_rank(lower_rank)
    upper_value = value_at_rank(upper_rank)
    if lower_rank == upper_rank:
        return float(lower_value)
    weight = position - lower_rank
    return lower_value + weight * (upper_value - lower_value)


def _hist_mean(counts: Mapping[int, int]) -> float:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return math.nan
    return sum(int(delay) * int(count) for delay, count in counts.items()) / total


def _serialize_hist(counts: Mapping[int, int]) -> str:
    return json.dumps(
        {str(int(delay)): int(count) for delay, count in sorted(counts.items())},
        separators=(",", ":"),
    )


def _class_row(
    *,
    task: Mapping[str, Any],
    simulation: ThresholdExposureSimulation,
    result: Any,
    class_id: int,
) -> dict[str, Any]:
    metrics = result.class_metrics[class_id]
    offered_counts = simulation.audit_offered_delay_counts[class_id]
    accepted_counts = Counter(
        {int(delay): int(count) for delay, count in metrics.accepted_delay_counts.items()}
    )

    balk_threshold = int(task[f"balk_threshold_{class_id}"])
    noshow_threshold = int(task[f"noshow_threshold_{class_id}"])
    offered_total = sum(offered_counts.values())
    accepted_total = sum(accepted_counts.values())
    offered_above = sum(
        count for delay, count in offered_counts.items() if delay > balk_threshold
    )
    accepted_above = sum(
        count for delay, count in accepted_counts.items() if delay > noshow_threshold
    )
    resolved_bookings = metrics.served + metrics.no_show

    row = {
        key: task[key]
        for key in (
            "background_id",
            "patient_characteristic",
            "class2_reference",
            "contrast_level",
            "profile_id",
            "clinic_context_id",
            "rho",
            "class1_share",
            "slots_per_day",
            "horizon_days",
            "lambda_1",
            "lambda_2",
            "seed",
        )
    }
    row.update(
        {
            "class_id": class_id,
            "cancel_prob": float(task[f"cancel_{class_id}"]),
            "balk_threshold": balk_threshold,
            "balk_low": float(task[f"balk_low_{class_id}"]),
            "balk_high": float(task[f"balk_high_{class_id}"]),
            "noshow_threshold": noshow_threshold,
            "noshow_low": float(task[f"noshow_low_{class_id}"]),
            "noshow_high": float(task[f"noshow_high_{class_id}"]),
            "balk_structurally_crossable": (
                int(task["horizon_days"]) - 1 > balk_threshold
            ),
            "noshow_structurally_crossable": (
                int(task["horizon_days"]) - 1 > noshow_threshold
            ),
            "arrivals": metrics.arrivals,
            "offers": offered_total,
            "offers_at_or_below_balk_threshold": offered_total - offered_above,
            "offers_above_balk_threshold": offered_above,
            "share_offers_above_balk_threshold": _safe_rate(
                offered_above, offered_total
            ),
            "accepted_bookings": accepted_total,
            "accepted_at_or_below_noshow_threshold": accepted_total - accepted_above,
            "accepted_above_noshow_threshold": accepted_above,
            "share_accepted_above_noshow_threshold": _safe_rate(
                accepted_above, accepted_total
            ),
            "balked": metrics.balked,
            "canceled": metrics.canceled,
            "no_show": metrics.no_show,
            "served": metrics.served,
            "no_offer": metrics.no_offer,
            "balk_rate_among_offers": _safe_rate(metrics.balked, offered_total),
            "cancellation_rate_among_bookings": _safe_rate(
                metrics.canceled, accepted_total
            ),
            "no_show_rate_among_resolved_bookings": _safe_rate(
                metrics.no_show, resolved_bookings
            ),
            "served_rate": _safe_rate(metrics.served, metrics.arrivals),
            "no_offer_rate": _safe_rate(metrics.no_offer, metrics.arrivals),
            "mean_offered_delay": _hist_mean(offered_counts),
            "median_offered_delay": _hist_quantile(offered_counts, 0.50),
            "p75_offered_delay": _hist_quantile(offered_counts, 0.75),
            "p90_offered_delay": _hist_quantile(offered_counts, 0.90),
            "mean_accepted_delay": _hist_mean(accepted_counts),
            "median_accepted_delay": _hist_quantile(accepted_counts, 0.50),
            "p75_accepted_delay": _hist_quantile(accepted_counts, 0.75),
            "p90_accepted_delay": _hist_quantile(accepted_counts, 0.90),
            "offered_delay_counts_json": _serialize_hist(offered_counts),
            "accepted_delay_counts_json": _serialize_hist(accepted_counts),
        }
    )
    return row


def run_one(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    classes = {
        class_id: PatientClassParams(
            class_id=class_id,
            lambda_per_day=float(task[f"lambda_{class_id}"]),
            cancel_prob=float(task[f"cancel_{class_id}"]),
            balk_prob=ThresholdRule(
                threshold=int(task[f"balk_threshold_{class_id}"]),
                low=float(task[f"balk_low_{class_id}"]),
                high=float(task[f"balk_high_{class_id}"]),
            ),
            no_show_prob=ThresholdRule(
                threshold=int(task[f"noshow_threshold_{class_id}"]),
                low=float(task[f"noshow_low_{class_id}"]),
                high=float(task[f"noshow_high_{class_id}"]),
            ),
        )
        for class_id in (1, 2)
    }
    config = SimulationConfig(
        slots_per_day=int(task["slots_per_day"]),
        horizon_days=int(task["horizon_days"]),
        burn_in_days=int(task["burn_in_days"]),
        measure_days=int(task["measure_days"]),
        cooldown_days=int(task["cooldown_days"]),
        classes=classes,
        seed=int(task["seed"]),
        reserved_class_id=None,
        reserved_slots_per_day=0,
        reserved_window_days=None,
        same_day_cancellation_enabled=True,
        release_unused_reservation_same_day=False,
    )
    simulation = ThresholdExposureSimulation(config)
    result = simulation.run()
    return [
        _class_row(
            task=task,
            simulation=simulation,
            result=result,
            class_id=class_id,
        )
        for class_id in (1, 2)
    ]


def _task_from_row(row: pd.Series, *, seed: int, smoke: bool) -> dict[str, Any]:
    task = row.to_dict()
    task.update(
        {
            "seed": int(seed),
            "burn_in_days": 5 if smoke else BURN_IN_DAYS,
            "measure_days": 30 if smoke else MEASURE_DAYS,
            "cooldown_days": 26 if not smoke else max(5, int(row["horizon_days"])),
        }
    )
    return task


def _completed_pairs(raw_path: Path) -> set[tuple[str, int]]:
    if not raw_path.exists():
        return set()
    existing = pd.read_csv(
        raw_path,
        usecols=["background_id", "seed", "class_id"],
    )
    counts = (
        existing.drop_duplicates()
        .groupby(["background_id", "seed"])["class_id"]
        .nunique()
    )
    return {
        (str(background_id), int(seed))
        for (background_id, seed), count in counts.items()
        if count == 2
    }


def _append_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_audit(
    *,
    bank_path: Path,
    output_dir: Path,
    workers: int,
    n_seeds: int,
    shard_index: int,
    shard_count: int,
    smoke: bool,
    resume: bool,
) -> None:
    if not 1 <= n_seeds <= len(SEEDS):
        raise ValueError(f"n_seeds must be between 1 and {len(SEEDS)}")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count.")

    bank = pd.read_csv(bank_path) if bank_path.exists() else write_bank(bank_path)
    if smoke:
        smoke_profiles = ["NS_LOW_MILD", "BK_LOW_MILD", "JT_MODERATE_STRONG"]
        bank = bank[
            bank["profile_id"].isin(smoke_profiles)
            & bank["rho"].isin([1.2, 3.0])
            & bank["class1_share"].isin([0.1, 0.9])
            & (bank["slots_per_day"] == 30)
            & bank["horizon_days"].isin([6, 22])
        ].copy()

    ordered = bank.sort_values("background_id", kind="stable").reset_index(drop=True)
    shard_bank = ordered[ordered.index % shard_count == shard_index]
    raw_path = (
        output_dir
        / "raw"
        / f"shard_{shard_index:03d}_of_{shard_count:03d}.csv"
    )
    if raw_path.exists() and not resume:
        raw_path.unlink()

    completed = _completed_pairs(raw_path) if resume else set()
    tasks: list[dict[str, Any]] = []
    for _, row in shard_bank.iterrows():
        for seed in SEEDS[:n_seeds]:
            key = (str(row["background_id"]), int(seed))
            if key not in completed:
                tasks.append(_task_from_row(row, seed=seed, smoke=smoke))

    print(
        f"Audit shard {shard_index}/{shard_count}: "
        f"{len(shard_bank):,} backgrounds, {len(tasks):,} new runs"
    )
    if not tasks:
        print("Nothing to run.")
        return

    buffer: list[dict[str, Any]] = []
    executor = None
    try:
        if workers <= 1:
            iterator = map(run_one, tasks)
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            iterator = executor.map(run_one, tasks, chunksize=2)

        for index, pair_rows in enumerate(iterator, start=1):
            buffer.extend(pair_rows)
            if len(buffer) >= 100:
                _append_rows(raw_path, buffer)
                buffer.clear()
            if index % 100 == 0 or index == len(tasks):
                print(f"Completed {index:,}/{len(tasks):,} new runs")
        _append_rows(raw_path, buffer)
    finally:
        if executor is not None:
            executor.shutdown()

    print(f"Audit shard complete: {raw_path}")


def _parse_hist(value: Any) -> Counter[int]:
    if pd.isna(value):
        return Counter()
    parsed = json.loads(str(value))
    return Counter({int(delay): int(count) for delay, count in parsed.items()})


def _merge_histograms(values: Iterable[Any]) -> Counter[int]:
    merged: Counter[int] = Counter()
    for value in values:
        merged.update(_parse_hist(value))
    return merged


def _aggregate_rows(group: pd.DataFrame, group_cols: list[str]) -> dict[str, Any]:
    first = group.iloc[0]
    offered = _merge_histograms(group["offered_delay_counts_json"])
    accepted = _merge_histograms(group["accepted_delay_counts_json"])
    balk_threshold = int(first["balk_threshold"])
    noshow_threshold = int(first["noshow_threshold"])
    offered_total = sum(offered.values())
    accepted_total = sum(accepted.values())
    offered_above = sum(
        count for delay, count in offered.items() if delay > balk_threshold
    )
    accepted_above = sum(
        count for delay, count in accepted.items() if delay > noshow_threshold
    )

    row = {column: first[column] for column in group_cols}
    row.update(
        {
            "seeds": group["seed"].nunique(),
            "balk_threshold": balk_threshold,
            "balk_low": float(first["balk_low"]),
            "balk_high": float(first["balk_high"]),
            "noshow_threshold": noshow_threshold,
            "noshow_low": float(first["noshow_low"]),
            "noshow_high": float(first["noshow_high"]),
            "contexts": group["background_id"].nunique(),
            "balk_structurally_crossable": bool(
                group["balk_structurally_crossable"].astype(bool).all()
            ),
            "noshow_structurally_crossable": bool(
                group["noshow_structurally_crossable"].astype(bool).all()
            ),
            "balk_crossable_context_share": group[
                "balk_structurally_crossable"
            ].astype(bool).groupby(group["background_id"]).max().mean(),
            "noshow_crossable_context_share": group[
                "noshow_structurally_crossable"
            ].astype(bool).groupby(group["background_id"]).max().mean(),
            "offers": offered_total,
            "offers_at_or_below_balk_threshold": offered_total - offered_above,
            "offers_above_balk_threshold": offered_above,
            "share_offers_above_balk_threshold": _safe_rate(
                offered_above, offered_total
            ),
            "accepted_bookings": accepted_total,
            "accepted_at_or_below_noshow_threshold": accepted_total - accepted_above,
            "accepted_above_noshow_threshold": accepted_above,
            "share_accepted_above_noshow_threshold": _safe_rate(
                accepted_above, accepted_total
            ),
            "arrivals": int(group["arrivals"].sum()),
            "balked": int(group["balked"].sum()),
            "canceled": int(group["canceled"].sum()),
            "no_show": int(group["no_show"].sum()),
            "served": int(group["served"].sum()),
            "no_offer": int(group["no_offer"].sum()),
            "balk_rate_among_offers": _safe_rate(
                group["balked"].sum(), offered_total
            ),
            "cancellation_rate_among_bookings": _safe_rate(
                group["canceled"].sum(), accepted_total
            ),
            "no_show_rate_among_resolved_bookings": _safe_rate(
                group["no_show"].sum(),
                group["no_show"].sum() + group["served"].sum(),
            ),
            "served_rate": _safe_rate(
                group["served"].sum(), group["arrivals"].sum()
            ),
            "no_offer_rate": _safe_rate(
                group["no_offer"].sum(), group["arrivals"].sum()
            ),
            "mean_offered_delay": _hist_mean(offered),
            "median_offered_delay": _hist_quantile(offered, 0.50),
            "p75_offered_delay": _hist_quantile(offered, 0.75),
            "p90_offered_delay": _hist_quantile(offered, 0.90),
            "mean_accepted_delay": _hist_mean(accepted),
            "median_accepted_delay": _hist_quantile(accepted, 0.50),
            "p75_accepted_delay": _hist_quantile(accepted, 0.75),
            "p90_accepted_delay": _hist_quantile(accepted, 0.90),
            "offered_delay_counts_json": _serialize_hist(offered),
            "accepted_delay_counts_json": _serialize_hist(accepted),
        }
    )
    return row


def _aggregate_frame(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = [
        _aggregate_rows(group, group_cols)
        for _, group in raw.groupby(group_cols, sort=False, dropna=False)
    ]
    return pd.DataFrame(rows)


def _status(crossable_share: float, realized_crossing: float, total: int) -> str:
    if total < 100:
        return "too few observations"
    if crossable_share < 0.50:
        return "too few contexts can cross"
    if realized_crossing < 0.05:
        return "too little realized crossing"
    if realized_crossing < 0.10:
        return "low realized crossing"
    if realized_crossing <= 0.70:
        return "informative"
    if realized_crossing <= 0.90:
        return "high realized crossing"
    return "mostly above threshold"


def _audit_flags(context: pd.DataFrame) -> pd.DataFrame:
    class1 = context[context["class_id"] == 1].copy()
    flag_rows: list[dict[str, Any]] = []

    for _, row in class1.iterrows():
        mechanisms: tuple[str, ...]
        if row["patient_characteristic"] == "no_show_sensitivity":
            mechanisms = ("no_show",)
        elif row["patient_characteristic"] == "balking_sensitivity":
            mechanisms = ("balking",)
        else:
            mechanisms = ("balking", "no_show")

        for mechanism in mechanisms:
            if mechanism == "balking":
                crossable = bool(row["balk_structurally_crossable"])
                above = int(row["offers_above_balk_threshold"])
                total = int(row["offers"])
                threshold = int(row["balk_threshold"])
            else:
                crossable = bool(row["noshow_structurally_crossable"])
                above = int(row["accepted_above_noshow_threshold"])
                total = int(row["accepted_bookings"])
                threshold = int(row["noshow_threshold"])

            flag_rows.append(
                {
                    "background_id": row["background_id"],
                    "profile_id": row["profile_id"],
                    "patient_characteristic": row["patient_characteristic"],
                    "class2_reference": row["class2_reference"],
                    "contrast_level": row["contrast_level"],
                    "rho": row["rho"],
                    "class1_share": row["class1_share"],
                    "slots_per_day": row["slots_per_day"],
                    "horizon_days": row["horizon_days"],
                    "mechanism": mechanism,
                    "threshold": threshold,
                    "structurally_crossable": crossable,
                    "above_threshold": above,
                    "total_observations": total,
                    "crossing_share": _safe_rate(above, total),
                }
            )

    flags = pd.DataFrame(flag_rows)
    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "profile_id",
        "patient_characteristic",
        "class2_reference",
        "contrast_level",
        "rho",
        "mechanism",
        "threshold",
    ]
    for keys, group in flags.groupby(group_cols, sort=False, dropna=False):
        record = dict(zip(group_cols, keys))
        total = int(group["total_observations"].sum())
        above = int(group["above_threshold"].sum())
        crossable_share = float(group["structurally_crossable"].mean())
        realized = _safe_rate(above, total)
        record.update(
            {
                "contexts": group["background_id"].nunique(),
                "crossable_context_share": crossable_share,
                "above_threshold": above,
                "total_observations": total,
                "crossing_share": realized,
                "status": _status(crossable_share, realized, total),
            }
        )
        summary_rows.append(record)
    return pd.DataFrame(summary_rows)


def _profile_recommendations(flags: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "profile_id",
        "patient_characteristic",
        "class2_reference",
        "contrast_level",
        "mechanism",
        "threshold",
    ]
    for keys, group in flags.groupby(group_cols, sort=False, dropna=False):
        informative_levels = int((group["status"] == "informative").sum())
        if informative_levels >= 2:
            recommendation = "retain"
        elif informative_levels == 1:
            recommendation = "review after context-level inspection"
        else:
            recommendation = "revise threshold or horizon coverage"
        record = dict(zip(group_cols, keys))
        record.update(
            {
                "informative_demand_levels": informative_levels,
                "demand_levels_tested": group["rho"].nunique(),
                "recommendation": recommendation,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _write_markdown_summary(
    *,
    raw: pd.DataFrame,
    flags: pd.DataFrame,
    recommendations: pd.DataFrame,
    path: Path,
) -> None:
    status_counts = flags["status"].value_counts().sort_index()
    recommendation_counts = recommendations["recommendation"].value_counts()
    lines = [
        "# Threshold Exposure Audit",
        "",
        f"- Seed-level simulations: {raw[['background_id', 'seed']].drop_duplicates().shape[0]:,}",
        f"- Backgrounds represented: {raw['background_id'].nunique():,}",
        f"- Profiles represented: {raw['profile_id'].nunique():,}",
        f"- Seeds represented: {raw['seed'].nunique():,}",
        "",
        "## Demand-level status counts",
        "",
        "| Status | Rows |",
        "|---|---:|",
    ]
    lines.extend(f"| {status} | {count} |" for status, count in status_counts.items())
    lines.extend(
        [
            "",
            "## Profile recommendations",
            "",
            "| Recommendation | Profile-mechanism rows |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| {recommendation} | {count} |"
        for recommendation, count in recommendation_counts.items()
    )
    lines.extend(
        [
            "",
            "Interpret the recommendations together with `threshold_exposure_flags.csv`.",
            "A low-reference threshold may be retained as a negative-control condition even",
            "when fewer than half of the horizon contexts can structurally cross it.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_audit(*, output_dir: Path, summary_dir: Path) -> None:
    raw_files = sorted((output_dir / "raw").glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No audit raw files found under {output_dir / 'raw'}")

    raw = pd.concat((pd.read_csv(path) for path in raw_files), ignore_index=True)
    raw = raw.drop_duplicates(
        subset=["background_id", "seed", "class_id"], keep="last"
    )
    summary_dir.mkdir(parents=True, exist_ok=True)

    context_cols = [
        "background_id",
        "patient_characteristic",
        "class2_reference",
        "contrast_level",
        "profile_id",
        "clinic_context_id",
        "rho",
        "class1_share",
        "slots_per_day",
        "horizon_days",
        "lambda_1",
        "lambda_2",
        "class_id",
    ]
    context = _aggregate_frame(raw, context_cols)
    context.to_csv(summary_dir / "context_class_exposure.csv", index=False)

    profile_demand_cols = [
        "patient_characteristic",
        "class2_reference",
        "contrast_level",
        "profile_id",
        "rho",
        "class_id",
    ]
    profile_demand = _aggregate_frame(raw, profile_demand_cols)
    profile_demand.to_csv(summary_dir / "profile_demand_exposure.csv", index=False)

    profile_cols = [
        "patient_characteristic",
        "class2_reference",
        "contrast_level",
        "profile_id",
        "class_id",
    ]
    profile = _aggregate_frame(raw, profile_cols)
    profile.to_csv(summary_dir / "profile_exposure.csv", index=False)

    flags = _audit_flags(context)
    flags.to_csv(summary_dir / "threshold_exposure_flags.csv", index=False)
    recommendations = _profile_recommendations(flags)
    recommendations.to_csv(summary_dir / "profile_recommendations.csv", index=False)
    _write_markdown_summary(
        raw=raw,
        flags=flags,
        recommendations=recommendations,
        path=summary_dir / "audit_summary.md",
    )

    print(f"Audit summaries written to: {summary_dir}")
    print("Recommendation counts:")
    print(recommendations["recommendation"].value_counts().to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=["build", "run", "summarize", "all"])
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary_dir = args.summary_dir or (args.output_dir / "summary")

    if args.command in {"build", "all"}:
        write_bank(args.bank)

    if args.command in {"run", "all"}:
        if not args.bank.exists():
            write_bank(args.bank)
        run_audit(
            bank_path=args.bank,
            output_dir=args.output_dir,
            workers=args.workers,
            n_seeds=args.n_seeds,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            smoke=args.smoke,
            resume=not args.no_resume,
        )

    if args.command in {"summarize", "all"}:
        if args.shard_count != 1 and args.command == "all":
            print("Skipping summary during a sharded run; summarize after all shards finish.")
        else:
            summarize_audit(output_dir=args.output_dir, summary_dir=summary_dir)


if __name__ == "__main__":
    main()
