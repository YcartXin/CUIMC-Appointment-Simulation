"""Shared infrastructure for the two new-policy hypothesis experiments.

This module is deliberately independent of the older nine-hypothesis
robustness framework under ``experiments/robustness/``, but it now follows
the same Sobol-sampled background-scenario-bank pattern for the same
reason: the two hypotheses here test *new* booking policies (windowed
reservation, standby/requeue) against a background space wide enough
(rho, horizon, class mix, capacity, cancellation, balking, no-show, all
per-class where physically meaningful) that a small hand-picked grid could
not distinguish "the hypothesis's stated condition matters" from "the
hand-picked backgrounds happened to favor the policy." See
experiments/hypothesis_scenario_bank.py for the generator; the two
experiment scripts consume its output rather than sweeping their own
background dimensions. The statistical conventions (0.005
practical-equivalence tolerance, paired-seed bootstrap confidence
intervals, supported / inconclusive / contradicted verdicts, 1%-style
near-tie ranges) are carried over from ``docs/reports/reservation/`` so
the eventual summary reads consistently with the rest of the repo.

Used by:
    experiments/hypothesis_scenario_bank.py
    experiments/h1_short_horizon_reservation.py
    experiments/h2_reject_and_requeue.py
"""

from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from analysis.metrics import result_metrics_from_result  # noqa: E402
from analysis.reservation_policy_selection import bootstrap_mean_ci  # noqa: E402
from simulation.engine import ClinicAppointmentSimulation  # noqa: E402
from simulation.model import PatientClassParams, SimulationConfig, ThresholdRule  # noqa: E402

# Practical-equivalence tolerance on rate/utilization deltas, matching
# docs/reports/reservation/assumption_diagnostics/assumption_report.md.
PRACTICAL_TOLERANCE = 0.005

# Seed conventions reused from the existing robustness study so results
# stay comparable and reproducible across reports.
STAGE1_SEEDS: tuple[int, ...] = tuple(range(1000, 1020))
STAGE2_SEEDS: tuple[int, ...] = tuple(range(2000, 2100))

# Baseline nuisance parameters, matching configs/baseline.yaml. Any stage
# that does not sweep a given dimension holds it at these values.
BASELINE_CANCEL = 0.10
BASELINE_BALK_THRESHOLD = 9
BASELINE_BALK_LOW = 0.0
BASELINE_BALK_HIGH = 0.5
BASELINE_NOSHOW_THRESHOLD = 6
BASELINE_NOSHOW_LOW = 0.0
BASELINE_NOSHOW_HIGH = 0.3
BASELINE_SLOTS_PER_DAY = 32
BASELINE_HORIZON_DAYS = 14
BASELINE_BURN_IN_DAYS = 30
BASELINE_MEASURE_DAYS = 365
BASELINE_COOLDOWN_DAYS = 14


# ---------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------

def build_config(
    *,
    seed: int,
    lambda_1: float,
    lambda_2: float,
    slots_per_day: int = BASELINE_SLOTS_PER_DAY,
    horizon_days: int = BASELINE_HORIZON_DAYS,
    burn_in_days: int = BASELINE_BURN_IN_DAYS,
    measure_days: int = BASELINE_MEASURE_DAYS,
    cooldown_days: int = BASELINE_COOLDOWN_DAYS,
    cancel_1: float = BASELINE_CANCEL,
    cancel_2: float = BASELINE_CANCEL,
    balk_threshold_1: int = BASELINE_BALK_THRESHOLD,
    balk_low_1: float = BASELINE_BALK_LOW,
    balk_high_1: float = BASELINE_BALK_HIGH,
    balk_threshold_2: int = BASELINE_BALK_THRESHOLD,
    balk_low_2: float = BASELINE_BALK_LOW,
    balk_high_2: float = BASELINE_BALK_HIGH,
    noshow_threshold_1: int = BASELINE_NOSHOW_THRESHOLD,
    noshow_low_1: float = BASELINE_NOSHOW_LOW,
    noshow_high_1: float = BASELINE_NOSHOW_HIGH,
    noshow_threshold_2: int = BASELINE_NOSHOW_THRESHOLD,
    noshow_low_2: float = BASELINE_NOSHOW_LOW,
    noshow_high_2: float = BASELINE_NOSHOW_HIGH,
    reserved_class_id: int | None = None,
    reserved_slots_per_day: int = 0,
    reserved_window_days: int | None = None,
    standby_prob_1: float = 0.0,
    standby_prob_2: float = 0.0,
    max_standby_days_1: int | None = None,
    max_standby_days_2: int | None = None,
    standby_eligible_after_days_1: int | None = None,
    standby_eligible_after_days_2: int | None = None,
) -> SimulationConfig:
    """Build a two-class SimulationConfig directly from explicit parameters.

    Unlike experiments/robustness/simulation_adapter.py, this does not
    start from configs/baseline.yaml and override columns from a scenario
    CSV row. These two hypotheses use small, explicit, hand-designed
    grids rather than a sampled scenario bank, so building the config
    directly keeps every experiment script's grid readable in one place.
    """
    classes = {
        1: PatientClassParams(
            class_id=1,
            lambda_per_day=float(lambda_1),
            cancel_prob=float(cancel_1),
            balk_prob=ThresholdRule(
                threshold=int(balk_threshold_1), low=float(balk_low_1), high=float(balk_high_1)
            ),
            no_show_prob=ThresholdRule(
                threshold=int(noshow_threshold_1), low=float(noshow_low_1), high=float(noshow_high_1)
            ),
            standby_prob=float(standby_prob_1),
            max_standby_days=(
                None if max_standby_days_1 is None else int(max_standby_days_1)
            ),
            standby_eligible_after_days=(
                None if standby_eligible_after_days_1 is None else int(standby_eligible_after_days_1)
            ),
        ),
        2: PatientClassParams(
            class_id=2,
            lambda_per_day=float(lambda_2),
            cancel_prob=float(cancel_2),
            balk_prob=ThresholdRule(
                threshold=int(balk_threshold_2), low=float(balk_low_2), high=float(balk_high_2)
            ),
            no_show_prob=ThresholdRule(
                threshold=int(noshow_threshold_2), low=float(noshow_low_2), high=float(noshow_high_2)
            ),
            standby_prob=float(standby_prob_2),
            max_standby_days=(
                None if max_standby_days_2 is None else int(max_standby_days_2)
            ),
            standby_eligible_after_days=(
                None if standby_eligible_after_days_2 is None else int(standby_eligible_after_days_2)
            ),
        ),
    }
    return SimulationConfig(
        slots_per_day=int(slots_per_day),
        horizon_days=int(horizon_days),
        burn_in_days=int(burn_in_days),
        measure_days=int(measure_days),
        cooldown_days=int(cooldown_days),
        classes=classes,
        seed=int(seed),
        reserved_class_id=reserved_class_id,
        reserved_slots_per_day=int(reserved_slots_per_day),
        reserved_window_days=(
            None if reserved_window_days is None else int(reserved_window_days)
        ),
    )


def flatten_result(result: Any, *, seed: int) -> dict[str, Any]:
    """Flatten a SimulationResults into the shared row schema.

    Reuses analysis.metrics.result_metrics_from_result for the metrics
    that already exist repo-wide (utilization, served rate, offered/
    accepted delay, class gaps), and adds the Hypothesis 1 / Hypothesis 2
    diagnostics that are new to this engine extension.
    """
    metrics = result_metrics_from_result(result)
    c1 = result.class_metrics[1]
    c2 = result.class_metrics[2]

    def _rate(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        **metrics,
        "seed": int(seed),
        "reserved_slot_fill_rate": result.reserved_slot_fill_rate,
        "class_1_no_show_rate": _rate(c1.no_show, c1.arrivals),
        "class_2_no_show_rate": _rate(c2.no_show, c2.arrivals),
        "class_1_no_offer_rate": _rate(c1.no_offer, c1.arrivals),
        "class_2_no_offer_rate": _rate(c2.no_offer, c2.arrivals),
        "class_1_standby_joined": c1.standby_joined,
        "class_2_standby_joined": c2.standby_joined,
        "class_1_standby_recalled": c1.standby_recalled,
        "class_2_standby_recalled": c2.standby_recalled,
        "class_1_standby_expired": c1.standby_expired,
        "class_2_standby_expired": c2.standby_expired,
        "class_1_standby_recall_rate": c1.standby_recall_rate,
        "class_2_standby_recall_rate": c2.standby_recall_rate,
        "class_1_mean_standby_wait_days": c1.mean_standby_wait_days,
        "class_2_mean_standby_wait_days": c2.mean_standby_wait_days,
        "class_1_mean_original_offered_delay_recalled": c1.mean_original_offered_delay_recalled,
        "class_2_mean_original_offered_delay_recalled": c2.mean_original_offered_delay_recalled,
    }


def run_one(task: Mapping[str, Any]) -> dict[str, Any]:
    """Run one (config, seed) task and return a flat row with its labels.

    ``task`` must have ``config_kwargs`` (passed to build_config),
    ``seed``, and ``extra_cols`` (label columns copied verbatim into the
    output row, e.g. stage/cell_id/arm).
    """
    config = build_config(seed=int(task["seed"]), **task["config_kwargs"])
    result = ClinicAppointmentSimulation(config).run()
    row = flatten_result(result, seed=int(task["seed"]))
    row.update(task["extra_cols"])
    return row


# ---------------------------------------------------------------------
# Resumable execution
# ---------------------------------------------------------------------

def load_completed_keys(path: Path, key_columns: Sequence[str]) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    existing = pd.read_csv(path, usecols=list(key_columns))
    return {tuple(row) for row in existing[list(key_columns)].itertuples(index=False)}


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_tasks(
    tasks: list[dict[str, Any]],
    *,
    raw_path: Path,
    run_fn: Callable[[Mapping[str, Any]], dict[str, Any]] = run_one,
    workers: int = 1,
    flush_every: int = 100,
) -> None:
    """Execute tasks (already filtered for resume) and append results.

    Mirrors the ProcessPoolExecutor pattern used throughout
    experiments/robustness/*_stage1.py and experiments/strict_reservation_*.
    """
    if not tasks:
        print("Nothing to run: all requested tasks are already completed.")
        return

    buffer: list[dict[str, Any]] = []
    executor = None
    try:
        if workers <= 1:
            iterator = map(run_fn, tasks)
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            iterator = executor.map(run_fn, tasks, chunksize=4)

        for index, row in enumerate(iterator, start=1):
            buffer.append(row)
            if len(buffer) >= flush_every:
                append_rows(raw_path, buffer)
                buffer.clear()
            if index % 200 == 0 or index == len(tasks):
                print(f"Completed {index:,}/{len(tasks):,} new runs")
        append_rows(raw_path, buffer)
    finally:
        if executor is not None:
            executor.shutdown()


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


# ---------------------------------------------------------------------
# Statistics and classification
# ---------------------------------------------------------------------

def paired_delta_ci(
    on_values: Sequence[float],
    off_values: Sequence[float],
    *,
    seed: int,
    draws: int = 4000,
) -> tuple[float, float, float, int]:
    """Bootstrap 95% CI on the seed-paired (on - off) delta.

    on_values and off_values must already be aligned by seed (same order,
    same length). Reuses analysis.reservation_policy_selection's bootstrap
    routine so the interval construction matches the reservation reports.
    """
    if len(on_values) != len(off_values):
        raise ValueError("on_values and off_values must be the same length (paired by seed).")
    deltas = [on - off for on, off in zip(on_values, off_values)]
    return bootstrap_mean_ci(deltas, seed=seed)


def classify_effect(
    mean: float,
    low: float,
    high: float,
    *,
    expected_sign: str,
    tolerance: float = PRACTICAL_TOLERANCE,
) -> str:
    """Classify a paired delta as supported / inconclusive / contradicted.

    expected_sign is "positive" (hypothesis predicts on > off) or
    "negative" (hypothesis predicts on < off). Vocabulary matches
    docs/reports/reservation/assumption_diagnostics/assumption_report.md.
    """
    if any(math.isnan(x) for x in (mean, low, high)):
        return "inconclusive"
    if expected_sign == "positive":
        if mean >= tolerance and low > 0:
            return "supported"
        if mean <= -tolerance and high < 0:
            return "contradicted"
    elif expected_sign == "negative":
        if mean <= -tolerance and high < 0:
            return "supported"
        if mean >= tolerance and low > 0:
            return "contradicted"
    else:
        raise ValueError(f"Unsupported expected_sign: {expected_sign}")
    return "inconclusive"


def best_and_near_tie(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    param_cols: Sequence[str],
    value_col: str,
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """Per group, find the best mean value and the 1%-relative near-tie set.

    df must already be aggregated to one row per (group, *param_cols)
    with a mean value in value_col (e.g. mean utilization across seeds
    for each (Q, w) cell). Returns one row per group with the best
    parameter combination and a near-tie range summary, following the
    1% near-tie convention used throughout docs/reports/reservation/.
    """
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(list(group_cols), sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        group = group.dropna(subset=[value_col])
        if group.empty:
            continue
        best_value = float(group[value_col].max())
        threshold = best_value - tolerance * abs(best_value)
        near = group[group[value_col] >= threshold]
        best_row = group.loc[group[value_col].idxmax()]

        def _label(row: pd.Series) -> str:
            return ",".join(f"{c}={row[c]}" for c in param_cols)

        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "best_value": best_value,
                "best_params": _label(best_row),
                "near_tie_count": int(len(near)),
                "near_tie_params": "; ".join(_label(r) for _, r in near.iterrows()),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(lines: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
