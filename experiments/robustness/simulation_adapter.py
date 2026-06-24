"""Adapter between robustness scenario rows and the clinic simulation.

This module converts one row from ``all_stage1_scenarios.csv`` into the
repository's immutable ``SimulationConfig`` objects and returns a flat metric
row after running the simulation.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

from analysis.metrics import result_metrics_from_result, safe_divide
from simulation.config_loader import load_config
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import SimulationConfig, ThresholdRule

REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = REPO_DIR / "configs" / "baseline.yaml"


class OfferedDelayInstrumentedSimulation(ClinicAppointmentSimulation):
    """Clinic simulation that records offered-delay counts without changing logic."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        self.offered_delay_counts_by_class: dict[int, dict[int, int]] = {
            class_id: {} for class_id in config.classes
        }
        self._record_offer_delays = False

    def process_daily_arrivals(
        self,
        ordered_arrivals: list[int],
        track_patients: bool,
    ) -> None:
        self._record_offer_delays = bool(track_patients)
        try:
            super().process_daily_arrivals(
                ordered_arrivals=ordered_arrivals,
                track_patients=track_patients,
            )
        finally:
            self._record_offer_delays = False

    def find_earliest_open_day(self, class_id: int):
        offer = super().find_earliest_open_day(class_id)
        if self._record_offer_delays and offer is not None:
            delay = int(offer[0])
            counts = self.offered_delay_counts_by_class[class_id]
            counts[delay] = counts.get(delay, 0) + 1
        return offer


def _as_int(row: Mapping[str, Any], key: str) -> int:
    return int(float(row[key]))


def _as_float(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def scenario_to_config(
    row: Mapping[str, Any],
    *,
    seed: int,
    base_config_path: str | Path = DEFAULT_BASE_CONFIG,
    overrides: Mapping[str, Any] | None = None,
) -> SimulationConfig:
    """Build a simulation config from a robustness-scenario row.

    ``overrides`` accepts scenario-column names such as ``cancel_class1`` or
    ``balk_high_class2``. This keeps focal interventions separate from the
    background scenario definition.
    """
    values = dict(row)
    if overrides:
        values.update(overrides)

    base = load_config(base_config_path)
    if 1 not in base.classes or 2 not in base.classes:
        raise ValueError("The robustness framework requires class IDs 1 and 2.")

    h1 = _as_int(values, "horizon_class1")
    h2 = _as_int(values, "horizon_class2")
    global_horizon = max(h1, h2)

    c1 = replace(
        base.classes[1],
        lambda_per_day=_as_float(values, "lambda_class1"),
        cancel_prob=_as_float(values, "cancel_class1"),
        horizon_days=h1,
        balk_prob=ThresholdRule(
            threshold=_as_int(values, "balk_threshold_class1"),
            low=_as_float(values, "balk_low_class1"),
            high=_as_float(values, "balk_high_class1"),
        ),
        no_show_prob=ThresholdRule(
            threshold=_as_int(values, "noshow_threshold_class1"),
            low=_as_float(values, "noshow_low_class1"),
            high=_as_float(values, "noshow_high_class1"),
        ),
    )
    c2 = replace(
        base.classes[2],
        lambda_per_day=_as_float(values, "lambda_class2"),
        cancel_prob=_as_float(values, "cancel_class2"),
        horizon_days=h2,
        balk_prob=ThresholdRule(
            threshold=_as_int(values, "balk_threshold_class2"),
            low=_as_float(values, "balk_low_class2"),
            high=_as_float(values, "balk_high_class2"),
        ),
        no_show_prob=ThresholdRule(
            threshold=_as_int(values, "noshow_threshold_class2"),
            low=_as_float(values, "noshow_low_class2"),
            high=_as_float(values, "noshow_high_class2"),
        ),
    )

    return replace(
        base,
        slots_per_day=_as_int(values, "slots_per_day"),
        horizon_days=global_horizon,
        classes={1: c1, 2: c2},
        seed=int(seed),
        # Robustness tests concern pooled FCFS, not reserved capacity.
        reserved_class_id=None,
        reserved_slots_per_day=0,
    )


def _flatten_result(result: Any, *, seed: int) -> dict[str, Any]:
    """Convert a simulation result to the shared flat robustness schema."""
    metrics = result_metrics_from_result(result)

    c1 = result.class_metrics[1]
    c2 = result.class_metrics[2]
    totals_arrivals = c1.arrivals + c2.arrivals
    totals_offered = c1.offered + c2.offered
    totals_no_offer = c1.no_offer + c2.no_offer

    return {
        **metrics,
        "seed": int(seed),
        "class_1_arrivals": c1.arrivals,
        "class_2_arrivals": c2.arrivals,
        "class_1_offered": c1.offered,
        "class_2_offered": c2.offered,
        "class_1_booked": c1.booked,
        "class_2_booked": c2.booked,
        "class_1_balked": c1.balked,
        "class_2_balked": c2.balked,
        "class_1_canceled": c1.canceled,
        "class_2_canceled": c2.canceled,
        "class_1_no_show": c1.no_show,
        "class_2_no_show": c2.no_show,
        "class_1_served": c1.served,
        "class_2_served": c2.served,
        "class_1_no_offer_rate": safe_divide(c1.no_offer, c1.arrivals),
        "class_2_no_offer_rate": safe_divide(c2.no_offer, c2.arrivals),
        "overall_no_offer_rate": safe_divide(totals_no_offer, totals_arrivals),
        "class_1_mean_offered_booking_delay": getattr(
            c1,
            "mean_offered_booking_delay",
            metrics.get("mean_offered_booking_delay"),
        ),
        "class_2_mean_offered_booking_delay": getattr(
            c2,
            "mean_offered_booking_delay",
            metrics.get("mean_offered_booking_delay"),
        ),
        "class_1_mean_accepted_booking_delay": c1.mean_accepted_booking_delay,
        "class_2_mean_accepted_booking_delay": c2.mean_accepted_booking_delay,
        "class_1_balk_rate_per_arrival": safe_divide(c1.balked, c1.arrivals),
        "class_2_balk_rate_per_arrival": safe_divide(c2.balked, c2.arrivals),
        "class_1_no_show_rate_per_arrival": safe_divide(c1.no_show, c1.arrivals),
        "class_2_no_show_rate_per_arrival": safe_divide(c2.no_show, c2.arrivals),
        "class_1_cancel_rate_per_arrival": safe_divide(c1.canceled, c1.arrivals),
        "class_2_cancel_rate_per_arrival": safe_divide(c2.canceled, c2.arrivals),
        "total_arrivals": totals_arrivals,
        "total_offered": totals_offered,
    }


def run_scenario(
    row: Mapping[str, Any],
    *,
    seed: int,
    base_config_path: str | Path = DEFAULT_BASE_CONFIG,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one scenario/seed combination and return flattened metrics."""
    config = scenario_to_config(
        row,
        seed=seed,
        base_config_path=base_config_path,
        overrides=overrides,
    )
    result = ClinicAppointmentSimulation(config).run()
    return _flatten_result(result, seed=seed)


def run_scenario_with_offered_delay_counts(
    row: Mapping[str, Any],
    *,
    seed: int,
    base_config_path: str | Path = DEFAULT_BASE_CONFIG,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one scenario and also return class-specific offered-delay histograms.

    The histograms count every measured-window offer, including offers that
    were subsequently balked. They are serialized as JSON for safe CSV storage.
    """
    config = scenario_to_config(
        row,
        seed=seed,
        base_config_path=base_config_path,
        overrides=overrides,
    )
    simulation = OfferedDelayInstrumentedSimulation(config)
    result = simulation.run()
    flat = _flatten_result(result, seed=seed)
    flat["class_1_offered_delay_counts_json"] = json.dumps(
        simulation.offered_delay_counts_by_class.get(1, {}),
        sort_keys=True,
    )
    flat["class_2_offered_delay_counts_json"] = json.dumps(
        simulation.offered_delay_counts_by_class.get(2, {}),
        sort_keys=True,
    )
    return flat
