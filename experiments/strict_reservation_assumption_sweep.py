"""
Diagnostic sweep for the strict Class-1 reservation policy.

The experiment deliberately does not optimize or rank reservation quantities.
It checks implementation invariants and evaluates pre-specified business
assumptions against paired FCFS comparisons.

Standard cardinality:
    15 regimes x 8 Q values x 4 demand values x 3 compositions x 20 seeds
    = 28,800 unique simulation tasks.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from simulation.engine import ClinicAppointmentSimulation
from simulation.model import (
    Booking,
    PatientClassParams,
    SimulationConfig,
    SimulationResults,
    ThresholdRule,
)


DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "strict_reservation_assumption_sweep"
DEFAULT_REPORT_DIR = (
    REPO_DIR / "docs" / "reports" / "reservation" / "assumption_diagnostics"
)

SLOTS_PER_DAY = 32
HORIZON_DAYS = 14
STANDARD_BURN_IN_DAYS = 30
STANDARD_MEASURE_DAYS = 365
STANDARD_COOLDOWN_DAYS = 14
STANDARD_Q_VALUES = (0, 1, 4, 8, 16, 24, 31, 32)
STANDARD_TOTAL_DEMANDS = (24, 32, 50, 100)
STANDARD_CLASS_1_SHARES = (0.25, 0.50, 0.75)
STANDARD_SEEDS = tuple(range(61001, 61021))
THRESHOLD_PAIRS = ((9, 9), (5, 9), (9, 5), (12, 12), (5, 12))
BALK_HIGH_VALUES = (0.3, 0.5, 0.7)

BALK_LOW = 0.0
NO_SHOW_THRESHOLD = 6
NO_SHOW_LOW = 0.0
NO_SHOW_HIGH = 0.3
CANCEL_PROB = 0.1
TOLERANCE = 0.005
BOOTSTRAP_DRAWS = 4000
SHARD_SCHEMA_VERSION = 1

HARD_CHECK_NAMES = (
    "configuration_probabilities",
    "class_accounting",
    "total_accounting",
    "delay_accounting",
    "slot_utilization_accounting",
    "capacity_total",
    "capacity_reserved",
    "capacity_general",
    "reserved_ownership",
    "earliest_admissible_offer",
    "zero_unresolved_with_cooldown",
    "deterministic_replay_canary",
    "q0_exact_fcfs_equivalence",
    "q_s_class2_exclusion",
)


@dataclass(frozen=True)
class Regime:
    regime_id: str
    class_1_threshold: int
    class_2_threshold: int
    balk_high: float


@dataclass(frozen=True)
class SweepProfile:
    name: str
    regimes: tuple[Regime, ...]
    q_values: tuple[int, ...]
    total_demands: tuple[int, ...]
    class_1_shares: tuple[float, ...]
    seeds: tuple[int, ...]
    burn_in_days: int
    measure_days: int
    cooldown_days: int


@dataclass(frozen=True)
class SweepTask:
    profile: str
    regime_id: str
    class_1_threshold: int
    class_2_threshold: int
    balk_high: float
    q: int
    total_demand: int
    class_1_share: float
    seed: int
    burn_in_days: int
    measure_days: int
    cooldown_days: int
    task_id: str


@dataclass
class OracleAudit:
    calls: int = 0
    tracked_offer_count: int = 0
    tracked_offer_delay_sum: int = 0
    tracked_offer_max_delay: int = -1
    violation_counts: Counter[str] | None = None
    samples: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.violation_counts is None:
            self.violation_counts = Counter()
        if self.samples is None:
            self.samples = []

    def record(self, code: str, **details: Any) -> None:
        assert self.violation_counts is not None
        assert self.samples is not None
        self.violation_counts[code] += 1
        if len(self.samples) < 12:
            self.samples.append({"code": code, **details})


def historical_regimes() -> tuple[Regime, ...]:
    """Return the 15 historical threshold/balking regimes."""
    return tuple(
        Regime(
            regime_id=f"t{c1:02d}_{c2:02d}_b{int(round(high * 10)):02d}",
            class_1_threshold=c1,
            class_2_threshold=c2,
            balk_high=high,
        )
        for c1, c2 in THRESHOLD_PAIRS
        for high in BALK_HIGH_VALUES
    )


def profile_grid(profile: str) -> SweepProfile:
    """Return an immutable standard or smoke grid."""
    regimes = historical_regimes()
    if profile == "standard":
        return SweepProfile(
            name=profile,
            regimes=regimes,
            q_values=STANDARD_Q_VALUES,
            total_demands=STANDARD_TOTAL_DEMANDS,
            class_1_shares=STANDARD_CLASS_1_SHARES,
            seeds=STANDARD_SEEDS,
            burn_in_days=STANDARD_BURN_IN_DAYS,
            measure_days=STANDARD_MEASURE_DAYS,
            cooldown_days=STANDARD_COOLDOWN_DAYS,
        )
    if profile == "smoke":
        return SweepProfile(
            name=profile,
            regimes=(regimes[0], regimes[2], regimes[-1]),
            q_values=(0, 4, 32),
            total_demands=(24, 50),
            class_1_shares=(0.25, 0.50),
            seeds=(61001, 61002),
            burn_in_days=2,
            measure_days=7,
            cooldown_days=HORIZON_DAYS,
        )
    raise ValueError(f"Unknown profile: {profile!r}")


def task_identifier(
    *,
    profile: str,
    regime_id: str,
    q: int,
    total_demand: int,
    class_1_share: float,
    seed: int,
    burn_in_days: int,
    measure_days: int,
    cooldown_days: int,
) -> str:
    """Build a stable, readable, collision-resistant task ID."""
    share = int(round(class_1_share * 100))
    stem = (
        f"{profile}__{regime_id}__q{q:02d}__d{total_demand:03d}"
        f"__s{share:02d}__seed{seed}"
    )
    payload = (
        f"{stem}|{burn_in_days}|{measure_days}|{cooldown_days}|"
        f"{SLOTS_PER_DAY}|{HORIZON_DAYS}"
    )
    digest = hashlib.blake2s(payload.encode("ascii"), digest_size=5).hexdigest()
    return f"{stem}__{digest}"


def build_tasks(profile: str | SweepProfile = "standard") -> list[SweepTask]:
    """Expand a profile into uniquely identified tasks."""
    grid = profile_grid(profile) if isinstance(profile, str) else profile
    tasks: list[SweepTask] = []
    for regime in grid.regimes:
        for q in grid.q_values:
            for demand in grid.total_demands:
                for share in grid.class_1_shares:
                    for seed in grid.seeds:
                        identifier = task_identifier(
                            profile=grid.name,
                            regime_id=regime.regime_id,
                            q=q,
                            total_demand=demand,
                            class_1_share=share,
                            seed=seed,
                            burn_in_days=grid.burn_in_days,
                            measure_days=grid.measure_days,
                            cooldown_days=grid.cooldown_days,
                        )
                        tasks.append(
                            SweepTask(
                                profile=grid.name,
                                regime_id=regime.regime_id,
                                class_1_threshold=regime.class_1_threshold,
                                class_2_threshold=regime.class_2_threshold,
                                balk_high=regime.balk_high,
                                q=q,
                                total_demand=demand,
                                class_1_share=share,
                                seed=seed,
                                burn_in_days=grid.burn_in_days,
                                measure_days=grid.measure_days,
                                cooldown_days=grid.cooldown_days,
                                task_id=identifier,
                            )
                        )
    identifiers = [task.task_id for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("Task identifiers are not unique.")
    return tasks


def expected_cardinality(profile: str | SweepProfile) -> int:
    grid = profile_grid(profile) if isinstance(profile, str) else profile
    return (
        len(grid.regimes)
        * len(grid.q_values)
        * len(grid.total_demands)
        * len(grid.class_1_shares)
        * len(grid.seeds)
    )


def build_config(task: SweepTask, *, explicit_q0_owner: bool = False) -> SimulationConfig:
    """Build the exact simulation configuration for one task."""
    lambda_1 = task.total_demand * task.class_1_share
    lambda_2 = task.total_demand - lambda_1
    classes = {
        1: PatientClassParams(
            class_id=1,
            lambda_per_day=lambda_1,
            balk_prob=ThresholdRule(
                threshold=task.class_1_threshold,
                low=BALK_LOW,
                high=task.balk_high,
            ),
            cancel_prob=CANCEL_PROB,
            no_show_prob=ThresholdRule(
                threshold=NO_SHOW_THRESHOLD,
                low=NO_SHOW_LOW,
                high=NO_SHOW_HIGH,
            ),
            value=1.0,
        ),
        2: PatientClassParams(
            class_id=2,
            lambda_per_day=lambda_2,
            balk_prob=ThresholdRule(
                threshold=task.class_2_threshold,
                low=BALK_LOW,
                high=task.balk_high,
            ),
            cancel_prob=CANCEL_PROB,
            no_show_prob=ThresholdRule(
                threshold=NO_SHOW_THRESHOLD,
                low=NO_SHOW_LOW,
                high=NO_SHOW_HIGH,
            ),
            value=1.0,
        ),
    }
    owner = 1 if task.q > 0 or explicit_q0_owner else None
    return SimulationConfig(
        slots_per_day=SLOTS_PER_DAY,
        horizon_days=HORIZON_DAYS,
        burn_in_days=task.burn_in_days,
        measure_days=task.measure_days,
        cooldown_days=task.cooldown_days,
        classes=classes,
        seed=task.seed,
        reserved_class_id=owner,
        reserved_slots_per_day=task.q,
    )


def strict_reservation_oracle(
    calendar: Sequence[Sequence[Booking]],
    config: SimulationConfig,
    class_id: int,
) -> Optional[tuple[int, bool]]:
    """
    Independent oracle for the strict-reservation offer.

    It intentionally does not call or share helpers with the engine's
    ``find_earliest_open_day`` implementation.
    """
    class_horizon = config.classes[class_id].horizon_days
    horizon = min(
        len(calendar),
        config.horizon_days if class_horizon is None else class_horizon,
    )
    q = config.reserved_slots_per_day
    if q == 0 or config.reserved_class_id is None:
        for day in range(horizon):
            if sum(1 for _ in calendar[day]) < config.slots_per_day:
                return day, False
        return None

    general_capacity = config.slots_per_day - q
    for day in range(horizon):
        reserved_used = 0
        general_used = 0
        for booking in calendar[day]:
            if booking.reserved_slot:
                reserved_used += 1
            else:
                general_used += 1
        if class_id == config.reserved_class_id and reserved_used < q:
            return day, True
        if general_used < general_capacity:
            return day, False
    return None


def capacity_violations(
    calendar: Sequence[Sequence[Booking]],
    config: SimulationConfig,
) -> Counter[str]:
    """Return current total/pool/ownership capacity violations."""
    violations: Counter[str] = Counter()
    q = config.reserved_slots_per_day
    general_capacity = config.slots_per_day - q
    for bookings in calendar:
        reserved_used = sum(bool(booking.reserved_slot) for booking in bookings)
        general_used = len(bookings) - reserved_used
        if len(bookings) > config.slots_per_day:
            violations["capacity_total"] += 1
        if reserved_used > q:
            violations["capacity_reserved"] += 1
        if general_used > general_capacity:
            violations["capacity_general"] += 1
        for booking in bookings:
            if booking.reserved_slot and booking.patient_class != config.reserved_class_id:
                violations["reserved_ownership"] += 1
    return violations


class StrictReservationDiagnosticSimulation(ClinicAppointmentSimulation):
    """Read-only instrumentation around the production simulation engine."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        self.oracle_audit = OracleAudit()
        self._audit_tracked_offers = False

    def _audit_calendar(self) -> None:
        for code, count in capacity_violations(self.calendar, self.config).items():
            for _ in range(count):
                self.oracle_audit.record(code)

    def find_earliest_open_day(self, class_id: int) -> Optional[tuple[int, bool]]:
        expected = strict_reservation_oracle(self.calendar, self.config, class_id)
        actual = super().find_earliest_open_day(class_id)
        self.oracle_audit.calls += 1
        if actual != expected:
            self.oracle_audit.record(
                "earliest_admissible_offer",
                class_id=class_id,
                expected=expected,
                actual=actual,
            )
        if actual is not None:
            day, reserved = actual
            if self._audit_tracked_offers:
                self.oracle_audit.tracked_offer_count += 1
                self.oracle_audit.tracked_offer_delay_sum += day
                self.oracle_audit.tracked_offer_max_delay = max(
                    self.oracle_audit.tracked_offer_max_delay,
                    day,
                )
            if reserved and class_id != self.config.reserved_class_id:
                self.oracle_audit.record(
                    "reserved_ownership",
                    class_id=class_id,
                    day=day,
                )
        self._audit_calendar()
        return actual

    def process_daily_arrivals(
        self,
        ordered_arrivals: list[int],
        track_patients: bool,
    ) -> None:
        self._audit_tracked_offers = track_patients
        try:
            super().process_daily_arrivals(ordered_arrivals, track_patients)
            self._audit_calendar()
        finally:
            self._audit_tracked_offers = False


def configuration_checks(
    config: SimulationConfig,
    task: SweepTask | None = None,
) -> dict[str, bool]:
    """Validate fixed values and probability-rule boundary behavior."""
    probabilities_valid = True
    for params in config.classes.values():
        if not isinstance(params.balk_prob, ThresholdRule) or not isinstance(
            params.no_show_prob, ThresholdRule
        ):
            probabilities_valid = False
            continue
        threshold = params.balk_prob.threshold
        probabilities_valid &= math.isclose(params.balk_prob(threshold), BALK_LOW)
        probabilities_valid &= math.isclose(
            params.balk_prob(threshold + 1), params.balk_prob.high
        )
        probabilities_valid &= math.isclose(
            params.no_show_prob(NO_SHOW_THRESHOLD), NO_SHOW_LOW
        )
        probabilities_valid &= math.isclose(
            params.no_show_prob(NO_SHOW_THRESHOLD + 1), NO_SHOW_HIGH
        )
        probabilities_valid &= all(
            0.0 <= probability <= 1.0
            for probability in (
                params.balk_prob(0),
                params.balk_prob(HORIZON_DAYS),
                params.no_show_prob(0),
                params.no_show_prob(HORIZON_DAYS),
                params.cancel_prob,
            )
        )
    fixed_valid = (
        config.slots_per_day == SLOTS_PER_DAY
        and config.horizon_days == HORIZON_DAYS
        and config.cooldown_days >= config.horizon_days
        and 0 <= config.reserved_slots_per_day <= config.slots_per_day
        and set(config.classes) == {1, 2}
        and all(params.lambda_per_day >= 0 for params in config.classes.values())
        and (
            config.reserved_slots_per_day == 0
            or config.reserved_class_id == 1
        )
    )
    if task is not None:
        fixed_valid &= config.reserved_slots_per_day == task.q
        fixed_valid &= config.seed == task.seed
        fixed_valid &= config.burn_in_days == task.burn_in_days
        fixed_valid &= config.measure_days == task.measure_days
        fixed_valid &= config.cooldown_days == task.cooldown_days
        fixed_valid &= math.isclose(
            sum(params.lambda_per_day for params in config.classes.values()),
            task.total_demand,
        )
        fixed_valid &= math.isclose(
            config.classes[1].lambda_per_day,
            task.total_demand * task.class_1_share,
        )
        fixed_valid &= (
            config.classes[1].balk_prob.threshold == task.class_1_threshold
        )
        fixed_valid &= (
            config.classes[2].balk_prob.threshold == task.class_2_threshold
        )
        fixed_valid &= all(
            math.isclose(params.balk_prob.high, task.balk_high)
            for params in config.classes.values()
        )
    return {
        "configuration_probabilities": bool(probabilities_valid and fixed_valid),
    }


def class_accounting_checks(results: SimulationResults) -> dict[str, bool]:
    """Check class-level and aggregate flow conservation."""
    class_ok = True
    delay_ok = True
    totals = Counter()
    for metrics in results.class_metrics.values():
        unresolved = (
            metrics.booked - metrics.canceled - metrics.no_show - metrics.served
        )
        class_ok &= unresolved >= 0
        class_ok &= metrics.arrivals == metrics.offered + metrics.no_offer
        class_ok &= metrics.offered == metrics.booked + metrics.balked
        class_ok &= metrics.booked == (
            metrics.canceled + metrics.no_show + metrics.served + unresolved
        )
        delay_count = sum(metrics.accepted_delay_counts.values())
        delay_sum = sum(
            delay * count for delay, count in metrics.accepted_delay_counts.items()
        )
        delay_ok &= delay_count == metrics.booked
        delay_ok &= math.isclose(delay_sum, metrics.total_booking_delay)
        delay_ok &= metrics.total_offered_booking_delay >= 0
        delay_ok &= metrics.total_booking_delay >= 0
        delay_ok &= all(
            0 <= delay < HORIZON_DAYS
            for delay in metrics.accepted_delay_counts
        )
        delay_ok &= (
            metrics.total_booking_delay
            <= metrics.booked * (HORIZON_DAYS - 1)
        )
        delay_ok &= (
            metrics.total_offered_booking_delay
            <= metrics.offered * (HORIZON_DAYS - 1)
        )
        for name in (
            "arrivals",
            "booked",
            "balked",
            "no_offer",
            "canceled",
            "no_show",
            "served",
        ):
            totals[name] += getattr(metrics, name)
        totals["unresolved"] += unresolved

    total_ok = totals["arrivals"] == (
        totals["served"]
        + totals["balked"]
        + totals["no_offer"]
        + totals["canceled"]
        + totals["no_show"]
        + totals["unresolved"]
    )
    total_ok &= totals["booked"] == (
        totals["served"]
        + totals["canceled"]
        + totals["no_show"]
        + totals["unresolved"]
    )
    slots = results.slot_metrics
    slot_ok = (
        slots.booked_slots == slots.served_slots + slots.no_show_slots
        and 0 <= slots.booked_slots <= results.total_slots
        and 0 <= slots.served_slots <= slots.booked_slots
        and 0 <= slots.no_show_slots <= slots.booked_slots
        and slots.measured_days > 0
        and math.isclose(
            results.booked_slot_utilization,
            _safe_rate(slots.booked_slots, results.total_slots),
        )
        and math.isclose(
            results.average_utilization,
            _safe_rate(slots.daily_utilization_sum, slots.measured_days),
        )
    )
    return {
        "class_accounting": bool(class_ok),
        "total_accounting": bool(total_ok),
        "delay_accounting": bool(delay_ok),
        "slot_utilization_accounting": bool(slot_ok),
        "zero_unresolved_with_cooldown": totals["unresolved"] == 0,
    }


def check_accounting(results: SimulationResults) -> dict[str, bool]:
    """Public accounting-check alias suitable for focused tests."""
    return class_accounting_checks(results)


def result_fingerprint(results: SimulationResults) -> str:
    """Serialize all stable result state used by deterministic canaries."""
    payload: dict[str, Any] = {
        "classes": {},
        "slots": asdict(results.slot_metrics),
        "total_slots": results.total_slots,
        "total_value": results.total_value,
        "daily_summary_states": results.daily_summary_states,
        "final_full_state": results.final_full_state,
    }
    for class_id, metrics in sorted(results.class_metrics.items()):
        data = asdict(metrics)
        data["accepted_delay_counts"] = sorted(
            metrics.accepted_delay_counts.items()
        )
        payload["classes"][class_id] = data
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_replay_canary(task: SweepTask) -> bool:
    """Select two deterministic replay canaries per profile."""
    return (
        task.q == 4
        and math.isclose(task.class_1_share, 0.5)
        and task.seed == 61001
        and (
            (
                task.regime_id == historical_regimes()[0].regime_id
                and task.total_demand == 24
            )
            or (
                task.regime_id == historical_regimes()[-1].regime_id
                and task.total_demand == 50
            )
        )
    )


def is_q0_canary(task: SweepTask) -> bool:
    """Select two exact Q=0/FCFS equivalence canaries per profile."""
    return (
        task.q == 0
        and math.isclose(task.class_1_share, 0.5)
        and task.seed == 61001
        and (
            (
                task.regime_id == historical_regimes()[0].regime_id
                and task.total_demand == 24
            )
            or (
                task.regime_id == historical_regimes()[-1].regime_id
                and task.total_demand == 50
            )
        )
    )


def _safe_rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _result_row(
    task: SweepTask,
    results: SimulationResults,
    audit: OracleAudit,
    hard_checks: Mapping[str, bool],
) -> dict[str, Any]:
    c1 = results.class_metrics[1]
    c2 = results.class_metrics[2]
    all_metrics = (c1, c2)
    total_arrivals = sum(m.arrivals for m in all_metrics)
    total_booked = sum(m.booked for m in all_metrics)
    total_balked = sum(m.balked for m in all_metrics)
    total_no_offer = sum(m.no_offer for m in all_metrics)
    total_canceled = sum(m.canceled for m in all_metrics)
    total_no_show = sum(m.no_show for m in all_metrics)
    total_served = sum(m.served for m in all_metrics)
    total_offered = total_booked + total_balked
    total_offered_delay = sum(m.total_offered_booking_delay for m in all_metrics)
    total_accepted_delay = sum(m.total_booking_delay for m in all_metrics)
    unresolved = (
        total_booked - total_canceled - total_no_show - total_served
    )
    violations = audit.violation_counts or Counter()
    row: dict[str, Any] = {
        "shard_schema_version": SHARD_SCHEMA_VERSION,
        **asdict(task),
        "lambda_1": task.total_demand * task.class_1_share,
        "lambda_2": task.total_demand * (1.0 - task.class_1_share),
        "slots_per_day": SLOTS_PER_DAY,
        "horizon_days": HORIZON_DAYS,
        "total_arrivals": total_arrivals,
        "total_offered": total_offered,
        "total_booked": total_booked,
        "total_balked": total_balked,
        "total_no_offer": total_no_offer,
        "total_canceled": total_canceled,
        "total_no_show": total_no_show,
        "total_served": total_served,
        "total_unresolved": unresolved,
        "overall_served_rate": _safe_rate(total_served, total_arrivals),
        "overall_no_offer_rate": _safe_rate(total_no_offer, total_arrivals),
        "overall_balk_rate": _safe_rate(total_balked, total_offered),
        "mean_offered_wait": _safe_rate(total_offered_delay, total_offered),
        "mean_accepted_wait": _safe_rate(total_accepted_delay, total_booked),
        "booked_slot_utilization": results.booked_slot_utilization,
        "served_slot_utilization": results.average_utilization,
        "class_1_arrivals": c1.arrivals,
        "class_1_offered": c1.offered,
        "class_1_booked": c1.booked,
        "class_1_balked": c1.balked,
        "class_1_no_offer": c1.no_offer,
        "class_1_canceled": c1.canceled,
        "class_1_no_show": c1.no_show,
        "class_1_served": c1.served,
        "class_1_served_rate": c1.percent_serviced,
        "class_1_no_offer_rate": _safe_rate(c1.no_offer, c1.arrivals),
        "class_1_mean_offered_wait": (
            c1.total_offered_booking_delay / c1.offered
            if c1.offered
            else math.nan
        ),
        "class_2_arrivals": c2.arrivals,
        "class_2_offered": c2.offered,
        "class_2_booked": c2.booked,
        "class_2_balked": c2.balked,
        "class_2_no_offer": c2.no_offer,
        "class_2_canceled": c2.canceled,
        "class_2_no_show": c2.no_show,
        "class_2_served": c2.served,
        "class_2_served_rate": c2.percent_serviced,
        "class_2_no_offer_rate": _safe_rate(c2.no_offer, c2.arrivals),
        "class_2_mean_offered_wait": (
            c2.total_offered_booking_delay / c2.offered
            if c2.offered
            else math.nan
        ),
        "additive_equal_weight_objective": (
            c1.percent_serviced + c2.percent_serviced
        ),
        "pooled_objective": _safe_rate(total_served, total_arrivals),
        "oracle_calls": audit.calls,
        "audited_tracked_offer_count": audit.tracked_offer_count,
        "audited_tracked_offer_delay_sum": audit.tracked_offer_delay_sum,
        "audited_tracked_offer_max_delay": audit.tracked_offer_max_delay,
        "oracle_violation_count": sum(violations.values()),
        "oracle_violation_counts": json.dumps(
            dict(sorted(violations.items())), separators=(",", ":")
        ),
        "oracle_violation_samples": json.dumps(
            audit.samples or [], separators=(",", ":")
        ),
        "hard_check_failure_count": sum(not value for value in hard_checks.values()),
        "is_deterministic_replay_canary": is_replay_canary(task),
        "is_q0_equivalence_canary": is_q0_canary(task),
    }
    for name in HARD_CHECK_NAMES:
        row[f"check_{name}"] = bool(hard_checks.get(name, True))
    return row


def run_task(task: SweepTask) -> dict[str, Any]:
    """Run one task and return one compact, shard-ready record."""
    config = build_config(task)
    sim = StrictReservationDiagnosticSimulation(config)
    results = sim.run()

    hard_checks: dict[str, bool] = {}
    hard_checks.update(configuration_checks(config, task))
    hard_checks.update(class_accounting_checks(results))
    hard_checks["delay_accounting"] &= (
        sim.oracle_audit.tracked_offer_count
        == sum(metrics.offered for metrics in results.class_metrics.values())
        and sim.oracle_audit.tracked_offer_delay_sum
        == sum(
            metrics.total_offered_booking_delay
            for metrics in results.class_metrics.values()
        )
        and sim.oracle_audit.tracked_offer_max_delay < HORIZON_DAYS
    )
    violations = sim.oracle_audit.violation_counts or Counter()
    for name in (
        "capacity_total",
        "capacity_reserved",
        "capacity_general",
        "reserved_ownership",
        "earliest_admissible_offer",
    ):
        hard_checks[name] = violations[name] == 0

    hard_checks["deterministic_replay_canary"] = True
    if is_replay_canary(task):
        replay = StrictReservationDiagnosticSimulation(config).run()
        hard_checks["deterministic_replay_canary"] = (
            result_fingerprint(results) == result_fingerprint(replay)
        )

    hard_checks["q0_exact_fcfs_equivalence"] = True
    if task.q == 0:
        plain_fcfs = ClinicAppointmentSimulation(build_config(task)).run()
        explicit_owner_q0 = ClinicAppointmentSimulation(
            build_config(task, explicit_q0_owner=True)
        ).run()
        hard_checks["q0_exact_fcfs_equivalence"] = (
            result_fingerprint(plain_fcfs)
            == result_fingerprint(explicit_owner_q0)
            == result_fingerprint(results)
        )

    hard_checks["q_s_class2_exclusion"] = (
        task.q != SLOTS_PER_DAY
        or (
            results.class_metrics[2].offered == 0
            and results.class_metrics[2].booked == 0
            and results.class_metrics[2].balked == 0
            and results.class_metrics[2].served == 0
            and results.class_metrics[2].no_offer
            == results.class_metrics[2].arrivals
        )
    )
    return _result_row(task, results, sim.oracle_audit, hard_checks)


def shard_path(output_dir: Path, task: SweepTask) -> Path:
    return output_dir / "shards" / task.profile / f"{task.task_id}.csv.gz"


def atomic_write_csv_gz(path: Path, row: Mapping[str, Any]) -> None:
    """Atomically write a one-row compressed CSV shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_shard(path: Path) -> dict[str, str]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in shard {path}, found {len(rows)}.")
    return rows[0]


def valid_completed_task_ids(
    output_dir: Path,
    tasks: Iterable[SweepTask],
) -> set[str]:
    """Return task IDs with readable shards containing the matching ID."""
    completed: set[str] = set()
    for task in tasks:
        path = shard_path(output_dir, task)
        if not path.exists():
            continue
        try:
            row = read_shard(path)
        except (OSError, EOFError, csv.Error, ValueError):
            continue
        if (
            row.get("task_id") == task.task_id
            and row.get("shard_schema_version") == str(SHARD_SCHEMA_VERSION)
        ):
            completed.add(task.task_id)
    return completed


def resume_pending_tasks(
    output_dir: Path,
    tasks: Sequence[SweepTask],
) -> tuple[list[SweepTask], set[str]]:
    """Split a task list into pending and valid completed IDs."""
    completed = valid_completed_task_ids(output_dir, tasks)
    return [task for task in tasks if task.task_id not in completed], completed


def _run_and_write(task: SweepTask, output_dir: str) -> tuple[str, int]:
    row = run_task(task)
    atomic_write_csv_gz(shard_path(Path(output_dir), task), row)
    return task.task_id, int(row["hard_check_failure_count"])


def load_task_rows(output_dir: Path, tasks: Sequence[SweepTask]) -> pd.DataFrame:
    """Load valid rows for the requested task set in deterministic order."""
    rows: list[dict[str, str]] = []
    for task in tasks:
        path = shard_path(output_dir, task)
        if path.exists():
            row = read_shard(path)
            if (
                row.get("task_id") == task.task_id
                and row.get("shard_schema_version") == str(SHARD_SCHEMA_VERSION)
            ):
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    numeric_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "profile",
            "regime_id",
            "task_id",
            "oracle_violation_counts",
            "oracle_violation_samples",
        }
        and not column.startswith("check_")
    ]
    for column in numeric_columns:
        try:
            frame[column] = pd.to_numeric(frame[column])
        except (TypeError, ValueError):
            pass
    boolean_columns = [
        column
        for column in frame
        if column.startswith("check_") or column.startswith("is_")
    ]
    for column in boolean_columns:
        frame[column] = frame[column].astype(str).str.lower().eq("true")
    return frame


def paired_bootstrap_ci(
    differences: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 20260621,
) -> tuple[float, float, float]:
    """Return mean and percentile CI from paired differences."""
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return mean, float(low), float(high)


def _status_noninferiority(low: float, high: float, boundary: float) -> str:
    if low >= boundary:
        return "supported"
    if high < boundary:
        return "contradicted"
    return "inconclusive"


def _status_nonsuperiority(low: float, high: float, boundary: float) -> str:
    if high <= boundary:
        return "supported"
    if low > boundary:
        return "contradicted"
    return "inconclusive"


def _status_equivalence(low: float, high: float, tolerance: float) -> str:
    if low >= -tolerance and high <= tolerance:
        return "supported"
    if high < -tolerance or low > tolerance:
        return "contradicted"
    return "inconclusive"


def _bootstrap_record(
    *,
    assumption: str,
    metric: str,
    comparison: str,
    differences: Sequence[float],
    status_kind: str,
    tolerance: float = TOLERANCE,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    seed_payload = f"{assumption}|{metric}|{comparison}"
    bootstrap_seed = int.from_bytes(
        hashlib.blake2s(seed_payload.encode("utf-8"), digest_size=4).digest(),
        "big",
    )
    mean, low, high = paired_bootstrap_ci(
        differences,
        seed=bootstrap_seed,
    )
    if status_kind == "noninferiority":
        status = _status_noninferiority(low, high, -tolerance)
    elif status_kind == "nonsuperiority":
        status = _status_nonsuperiority(low, high, tolerance)
    elif status_kind == "equivalence":
        status = _status_equivalence(low, high, tolerance)
    else:
        raise ValueError(f"Unknown status kind: {status_kind}")
    return {
        "assumption": assumption,
        "metric": metric,
        "comparison": comparison,
        "n_pairs": len(differences),
        "mean_difference": mean,
        "ci95_low": low,
        "ci95_high": high,
        "tolerance": tolerance,
        "status": status,
        **(metadata or {}),
    }


PAIR_KEYS = [
    "regime_id",
    "class_1_threshold",
    "class_2_threshold",
    "balk_high",
    "total_demand",
    "class_1_share",
    "seed",
]


def paired_vs_fcfs(frame: pd.DataFrame) -> pd.DataFrame:
    """Pair every Q>0 run to Q=0 within the same scenario and seed."""
    fcfs_columns = PAIR_KEYS + [
        "class_1_served_rate",
        "class_2_served_rate",
        "overall_served_rate",
        "overall_no_offer_rate",
        "mean_offered_wait",
        "additive_equal_weight_objective",
        "pooled_objective",
    ]
    fcfs = frame.loc[frame["q"] == 0, fcfs_columns].copy()
    fcfs = fcfs.rename(
        columns={
            column: f"{column}_fcfs"
            for column in fcfs_columns
            if column not in PAIR_KEYS
        }
    )
    strict = frame.loc[frame["q"] > 0].copy()
    paired = strict.merge(fcfs, on=PAIR_KEYS, how="left", validate="many_to_one")
    for metric in (
        "class_1_served_rate",
        "class_2_served_rate",
        "overall_served_rate",
        "overall_no_offer_rate",
        "mean_offered_wait",
        "additive_equal_weight_objective",
        "pooled_objective",
    ):
        paired[f"delta_{metric}"] = paired[metric] - paired[f"{metric}_fcfs"]
    return paired


def business_assumption_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the specified assumptions without ranking Q values."""
    records: list[dict[str, Any]] = []
    paired = paired_vs_fcfs(frame)
    cell_keys = [
        "regime_id",
        "q",
        "total_demand",
        "class_1_share",
    ]
    for keys, group in paired.groupby(cell_keys, sort=True):
        metadata = dict(zip(cell_keys, keys))
        comparison = (
            f"Q={metadata['q']} minus FCFS; demand={metadata['total_demand']}; "
            f"C1 share={metadata['class_1_share']:.2f}; "
            f"regime={metadata['regime_id']}"
        )
        records.append(
            _bootstrap_record(
                assumption="C1 no material reduction vs FCFS",
                metric="class_1_served_rate",
                comparison=comparison,
                differences=group["delta_class_1_served_rate"].to_numpy(),
                status_kind="noninferiority",
                metadata=metadata,
            )
        )
        records.append(
            _bootstrap_record(
                assumption="C2 no material improvement vs FCFS",
                metric="class_2_served_rate",
                comparison=comparison,
                differences=group["delta_class_2_served_rate"].to_numpy(),
                status_kind="nonsuperiority",
                metadata=metadata,
            )
        )

    symmetric_fcfs = frame[
        (frame["q"] == 0)
        & (frame["class_1_threshold"] == frame["class_2_threshold"])
        & np.isclose(frame["class_1_share"], 0.5)
    ].copy()
    symmetric_fcfs["class_rate_gap"] = (
        symmetric_fcfs["class_1_served_rate"]
        - symmetric_fcfs["class_2_served_rate"]
    )
    symmetric_keys = ["regime_id", "total_demand"]
    for keys, group in symmetric_fcfs.groupby(symmetric_keys, sort=True):
        metadata = dict(zip(symmetric_keys, keys))
        records.append(
            _bootstrap_record(
                assumption="Symmetric FCFS has similar class served rates",
                metric="class_1_minus_class_2_served_rate",
                comparison=(
                    f"FCFS; demand={metadata['total_demand']}; "
                    f"regime={metadata['regime_id']}"
                ),
                differences=group["class_rate_gap"].to_numpy(),
                status_kind="equivalence",
                metadata=metadata,
            )
        )

    heavy = frame[
        (frame["class_1_threshold"] == frame["class_2_threshold"])
        & np.isclose(frame["class_1_share"], 0.5)
        & (frame["total_demand"] >= 50)
        & (frame["balk_high"].isin([0.3, 0.7]))
    ].copy()
    heavy_keys = [
        "class_1_threshold",
        "class_2_threshold",
        "q",
        "total_demand",
        "seed",
    ]
    low = heavy[heavy["balk_high"] == 0.3][
        heavy_keys + ["overall_served_rate"]
    ].rename(columns={"overall_served_rate": "rate_low"})
    high = heavy[heavy["balk_high"] == 0.7][
        heavy_keys + ["overall_served_rate"]
    ].rename(columns={"overall_served_rate": "rate_high"})
    heavy_pairs = high.merge(low, on=heavy_keys, validate="one_to_one")
    heavy_pairs["difference"] = heavy_pairs["rate_high"] - heavy_pairs["rate_low"]
    for keys, group in heavy_pairs.groupby(
        ["class_1_threshold", "q", "total_demand"], sort=True
    ):
        threshold, q, demand = keys
        records.append(
            _bootstrap_record(
                assumption="Balk 0.3 vs 0.7 has little heavy symmetric effect",
                metric="overall_served_rate",
                comparison=(
                    f"balk_high 0.7 minus 0.3; symmetric threshold={threshold}; "
                    f"demand={demand}; Q={q}"
                ),
                differences=group["difference"].to_numpy(),
                status_kind="equivalence",
                metadata={
                    "class_1_threshold": threshold,
                    "class_2_threshold": threshold,
                    "total_demand": demand,
                    "q": q,
                    "class_1_share": 0.5,
                },
            )
        )

    equal_demand = frame[np.isclose(frame["class_1_share"], 0.5)].copy()
    objective_columns = [
        "regime_id",
        "total_demand",
        "seed",
        "q",
        "additive_equal_weight_objective",
        "pooled_objective",
    ]
    left = equal_demand[objective_columns].rename(
        columns={
            "q": "q_left",
            "additive_equal_weight_objective": "additive_left",
            "pooled_objective": "pooled_left",
        }
    )
    right = equal_demand[objective_columns].rename(
        columns={
            "q": "q_right",
            "additive_equal_weight_objective": "additive_right",
            "pooled_objective": "pooled_right",
        }
    )
    objective_pairs = left.merge(
        right,
        on=["regime_id", "total_demand", "seed"],
        validate="many_to_many",
    )
    objective_pairs = objective_pairs[
        objective_pairs["q_left"] < objective_pairs["q_right"]
    ].copy()
    objective_pairs["additive_order"] = np.sign(
        objective_pairs["additive_left"] - objective_pairs["additive_right"]
    )
    objective_pairs["pooled_order"] = np.sign(
        objective_pairs["pooled_left"] - objective_pairs["pooled_right"]
    )
    objective_pairs["ordering_disagreement"] = (
        objective_pairs["additive_order"] != objective_pairs["pooled_order"]
    ).astype(float)
    seed_disagreement = (
        objective_pairs.groupby(
            ["regime_id", "total_demand", "seed"],
            as_index=False,
        )["ordering_disagreement"]
        .mean()
    )
    for keys, group in seed_disagreement.groupby(
        ["regime_id", "total_demand"], sort=True
    ):
        regime_id, demand = keys
        records.append(
            _bootstrap_record(
                assumption="Additive and pooled objective ordering is consistent",
                metric="pairwise_ordering_disagreement_rate",
                comparison=(
                    f"equal demand; all pairwise Q comparisons; "
                    f"demand={demand}; regime={regime_id}"
                ),
                differences=group["ordering_disagreement"].to_numpy(),
                status_kind="nonsuperiority",
                metadata={
                    "regime_id": regime_id,
                    "total_demand": demand,
                    "class_1_share": 0.5,
                },
            )
        )
    return pd.DataFrame(records)


def composition_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag lower offered waits that coincide with increased no-offer rates."""
    paired = paired_vs_fcfs(frame)
    keys = ["regime_id", "q", "total_demand", "class_1_share"]
    records: list[dict[str, Any]] = []
    for values, group in paired.groupby(keys, sort=True):
        metadata = dict(zip(keys, values))
        wait_mean, wait_low, wait_high = paired_bootstrap_ci(
            group["delta_mean_offered_wait"].to_numpy(),
            seed=1701,
        )
        no_offer_mean, no_offer_low, no_offer_high = paired_bootstrap_ci(
            group["delta_overall_no_offer_rate"].to_numpy(),
            seed=1702,
        )
        records.append(
            {
                **metadata,
                "n_pairs": len(group),
                "offered_wait_difference": wait_mean,
                "offered_wait_ci95_low": wait_low,
                "offered_wait_ci95_high": wait_high,
                "no_offer_rate_difference": no_offer_mean,
                "no_offer_ci95_low": no_offer_low,
                "no_offer_ci95_high": no_offer_high,
                "composition_flag": (
                    wait_high < -TOLERANCE and no_offer_low > TOLERANCE
                ),
            }
        )
    return pd.DataFrame(records)


def hard_violation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for name in HARD_CHECK_NAMES:
        applicable = frame
        if name == "deterministic_replay_canary":
            applicable = frame[frame["is_deterministic_replay_canary"]]
        elif name == "q0_exact_fcfs_equivalence":
            applicable = frame[frame["q"] == 0]
        elif name == "q_s_class2_exclusion":
            applicable = frame[frame["q"] == SLOTS_PER_DAY]
        column = f"check_{name}"
        failed = (
            int((~applicable[column]).sum())
            if column in applicable
            else len(applicable)
        )
        records.append(
            {
                "check": name,
                "tasks_evaluated": len(applicable),
                "failed_tasks": failed,
                "passed_tasks": len(applicable) - failed,
            }
        )
    return pd.DataFrame(records)


def cell_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "class_1_served_rate",
        "class_2_served_rate",
        "overall_served_rate",
        "overall_no_offer_rate",
        "mean_offered_wait",
        "booked_slot_utilization",
        "served_slot_utilization",
    ]
    keys = [
        "regime_id",
        "class_1_threshold",
        "class_2_threshold",
        "balk_high",
        "q",
        "total_demand",
        "class_1_share",
    ]
    return (
        frame.groupby(keys, as_index=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    flattened = frame.copy()
    flattened.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in flattened.columns
    ]
    return flattened


def plot_violation_counts(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#b3261e" if value else "#5f6b66" for value in summary["failed_tasks"]]
    ax.barh(summary["check"], summary["failed_tasks"], color=colors)
    ax.set_xlabel("Tasks failing check")
    ax.set_ylabel("")
    ax.set_title("Strict-reservation diagnostic violations")
    ax.grid(axis="x", alpha=0.25)
    if not summary["failed_tasks"].any():
        ax.set_xlim(0, 1)
        ax.text(
            0.5,
            0.5,
            "No hard-check failures",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_regions(assumptions: pd.DataFrame, path: Path) -> None:
    selected = assumptions[
        assumptions["assumption"].isin(
            [
                "C1 no material reduction vs FCFS",
                "C2 no material improvement vs FCFS",
            ]
        )
    ].copy()
    status_value = {"supported": 0, "inconclusive": 1, "contradicted": 2}
    selected["status_value"] = selected["status"].map(status_value)
    regions = (
        selected.groupby(["assumption", "total_demand", "q"], as_index=False)
        .agg(
            status_value=("status_value", "max"),
            non_supported_cells=(
                "status_value",
                lambda values: int((values > 0).sum()),
            ),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (assumption, group) in zip(
        axes,
        regions.groupby("assumption", sort=True),
    ):
        ax.scatter(
            group["total_demand"],
            group["q"],
            c=group["status_value"],
            s=55 + 35 * np.sqrt(group["non_supported_cells"]),
            cmap=matplotlib.colors.ListedColormap(
                ["#2f6f4e", "#d5a72d", "#b3261e"]
            ),
            vmin=0,
            vmax=2,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.4,
        )
        for row in group.itertuples(index=False):
            if row.non_supported_cells:
                ax.annotate(
                    str(row.non_supported_cells),
                    (row.total_demand, row.q),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
        ax.set_title(assumption.replace(" vs FCFS", "\nvs FCFS"), fontsize=10)
        ax.set_xlabel("Total daily demand")
        ax.set_ylabel("Reserved slots Q")
        ax.set_xticks(sorted(group["total_demand"].unique()))
        ax.set_yticks(sorted(group["q"].unique()))
        ax.grid(alpha=0.2)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color,
            label=label,
        )
        for label, color in (
            ("supported", "#2f6f4e"),
            ("inconclusive", "#d5a72d"),
            ("contradicted", "#b3261e"),
        )
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Business-assumption parameter regions\n"
        "Color is the worst status; labels count non-supported scenario cells",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def atomic_write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def dataframe_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional pandas dependencies."""
    columns = [str(column) for column in frame.columns]

    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    lines.extend(
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_report(
    frame: pd.DataFrame,
    *,
    profile: str,
    report_dir: Path = DEFAULT_REPORT_DIR,
    expected_tasks: int | None = None,
) -> dict[str, Path]:
    """Write markdown, compact CSVs, and the two required figures."""
    report_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = report_dir / "tables"
    figures_dir = report_dir / "figures"
    prefix = "" if profile == "standard" else f"{profile}_"

    violations = hard_violation_summary(frame)
    assumptions = business_assumption_summary(frame)
    composition = composition_flags(frame)
    cells = _flatten_columns(cell_summary(frame))

    violation_path = tables_dir / f"{prefix}violation_summary.csv"
    assumption_path = tables_dir / f"{prefix}business_assumptions.csv"
    composition_path = tables_dir / f"{prefix}composition_flags.csv"
    cell_path = tables_dir / f"{prefix}cell_summary.csv"
    plot_violation_path = figures_dir / f"{prefix}violation_counts.png"
    plot_region_path = figures_dir / f"{prefix}parameter_regions.png"
    markdown_path = report_dir / f"{prefix}assumption_report.md"

    atomic_write_dataframe(violations, violation_path)
    atomic_write_dataframe(assumptions, assumption_path)
    atomic_write_dataframe(composition, composition_path)
    atomic_write_dataframe(cells, cell_path)
    plot_violation_counts(violations, plot_violation_path)
    plot_parameter_regions(assumptions, plot_region_path)

    complete = expected_tasks is None or len(frame) == expected_tasks
    status_counts = assumptions["status"].value_counts().to_dict()
    assumption_breakdown = (
        assumptions.groupby(["assumption", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["supported", "inconclusive", "contradicted"], fill_value=0)
        .reset_index()
    )
    contradicted = assumptions[assumptions["status"] == "contradicted"]
    contradicted_summary = (
        contradicted.groupby("assumption", as_index=False)
        .agg(
            cells=("status", "size"),
            demand_values=(
                "total_demand",
                lambda values: ", ".join(
                    str(int(value))
                    for value in sorted(pd.Series(values).dropna().unique())
                ),
            ),
            regimes=("regime_id", lambda values: pd.Series(values).dropna().nunique()),
            minimum_effect=("mean_difference", "min"),
            maximum_effect=("mean_difference", "max"),
        )
        if not contradicted.empty
        else pd.DataFrame(
            columns=[
                "assumption",
                "cells",
                "demand_values",
                "regimes",
                "minimum_effect",
                "maximum_effect",
            ]
        )
    )
    for column in ("minimum_effect", "maximum_effect"):
        if column in contradicted_summary:
            contradicted_summary[column] = contradicted_summary[column].round(4)
    hard_failures = int(violations["failed_tasks"].sum())
    composition_count = int(composition["composition_flag"].sum())
    lines = [
        "# Strict Reservation Assumption Diagnostics",
        "",
        f"- Profile: `{profile}`",
        f"- Tasks analyzed: {len(frame):,}"
        + (f" of {expected_tasks:,}" if expected_tasks is not None else ""),
        f"- Sweep complete: `{complete}`",
        f"- Hard-check failed task-count sum: {hard_failures:,}",
        f"- Paired bootstrap draws: {BOOTSTRAP_DRAWS:,}",
        f"- Material-effect tolerance: {TOLERANCE:.3f}",
        "",
        "This report is diagnostic. It does not rank reservation quantities, "
        "recommend a policy, or assume monotonicity in Q.",
        "",
        "## Hard Checks",
        "",
        dataframe_markdown(violations),
        "",
        "![Violation counts](figures/"
        + plot_violation_path.name
        + ")",
        "",
        "## Business Assumptions",
        "",
        "Verdicts use paired seed-level bootstrap 95% confidence intervals. "
        "`supported` means the full interval satisfies the stated tolerance; "
        "`contradicted` means the full interval crosses the material boundary "
        "in the opposite direction; otherwise the result is `inconclusive`.",
        "",
        "| Status | Cells |",
        "|---|---:|",
        *[
            f"| {status} | {status_counts.get(status, 0):,} |"
            for status in ("supported", "inconclusive", "contradicted")
        ],
        "",
        "### Results By Assumption",
        "",
        dataframe_markdown(assumption_breakdown),
        "",
        "### Contradicted Assumptions",
        "",
        (
            dataframe_markdown(contradicted_summary)
            if not contradicted_summary.empty
            else "No business assumption was contradicted."
        ),
        "",
        "For the objective-ordering check, equal demand means equal configured "
        "arrival rates. Realized Poisson arrivals can still differ between "
        "classes within a seed, so the additive and pooled formulas are not "
        "guaranteed to be proportional.",
        "",
        "![Parameter regions](figures/" + plot_region_path.name + ")",
        "",
        "## Composition Diagnostic",
        "",
        f"{composition_count:,} cells show a confidently lower offered wait "
        "together with a confidently higher no-offer rate. Such cells are "
        "flagged as denominator-composition diagnostics, not policy benefits.",
        "",
        "## Files",
        "",
        f"- Hard-check summary: `tables/{violation_path.name}`",
        f"- Business assumptions: `tables/{assumption_path.name}`",
        f"- Composition flags: `tables/{composition_path.name}`",
        f"- Cell summary: `tables/{cell_path.name}`",
        "",
    ]
    fd, name = tempfile.mkstemp(prefix=f".{markdown_path.name}.", dir=report_dir)
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text("\n".join(lines), encoding="utf-8")
        os.replace(temporary, markdown_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "markdown": markdown_path,
        "violations": violation_path,
        "assumptions": assumption_path,
        "composition": composition_path,
        "cells": cell_path,
        "violation_plot": plot_violation_path,
        "region_plot": plot_region_path,
    }


def run_sweep(
    *,
    profile: str,
    output_dir: Path,
    workers: int,
    resume: bool,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Path]:
    tasks = build_tasks(profile)
    if resume:
        pending, completed = resume_pending_tasks(output_dir, tasks)
    else:
        pending, completed = list(tasks), set()
    print(
        f"Profile {profile}: {len(tasks):,} tasks; "
        f"{len(completed):,} resumable; {len(pending):,} pending."
    )

    failures = 0
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_and_write, task, str(output_dir)): task
                for task in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                task_id, task_failures = future.result()
                failures += task_failures
                if index == 1 or index % max(1, len(pending) // 20) == 0:
                    print(
                        f"Completed {index:,}/{len(pending):,} pending tasks "
                        f"(latest {task_id}; hard failures seen {failures:,})."
                    )

    frame = load_task_rows(output_dir, tasks)
    if len(frame) != len(tasks):
        raise RuntimeError(
            f"Only {len(frame):,} of {len(tasks):,} task shards are valid."
        )
    return write_report(
        frame,
        profile=profile,
        report_dir=report_dir,
        expected_tasks=len(tasks),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "standard"),
        default="smoke",
        help="Sweep size and simulation duration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print grid cardinality and exit without creating output.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid atomic task shards and run only missing tasks.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="ProcessPoolExecutor worker count.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root for compressed task shards.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive.")
    grid = profile_grid(args.profile)
    cardinality = expected_cardinality(grid)
    print(
        f"{args.profile} cardinality: {cardinality:,} unique tasks "
        f"({len(grid.regimes)} regimes x {len(grid.q_values)} Q x "
        f"{len(grid.total_demands)} demand x "
        f"{len(grid.class_1_shares)} shares x {len(grid.seeds)} seeds)."
    )
    if args.profile == "standard" and cardinality != 28800:
        raise AssertionError(
            f"Standard grid must contain 28,800 tasks, found {cardinality:,}."
        )
    if args.dry_run:
        return 0
    paths = run_sweep(
        profile=args.profile,
        output_dir=args.output_dir.resolve(),
        workers=args.workers,
        resume=args.resume,
    )
    print(f"Report: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
