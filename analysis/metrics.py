from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from simulation.model import SimulationResults


TWO_CLASS_IDS = (1, 2)


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    label: str
    unit: str
    direction: str
    description: str


METRIC_DEFINITIONS = {
    "average_utilization": MetricDefinition(
        name="average_utilization",
        label="Average utilization",
        unit="rate",
        direction="higher means more completed slot use",
        description="Completed visits divided by available measured slots.",
    ),
    "overall_percent_serviced": MetricDefinition(
        name="overall_percent_serviced",
        label="Overall percent serviced",
        unit="rate",
        direction="higher means more measured arrivals reached service",
        description="Served measured arrivals divided by all measured arrivals.",
    ),
    "mean_accepted_booking_delay": MetricDefinition(
        name="mean_accepted_booking_delay",
        label="Mean accepted booking delay",
        unit="days",
        direction="lower means accepted patients booked sooner",
        description="Average offered delay among patients who accepted an appointment.",
    ),
    "mean_offered_booking_delay": MetricDefinition(
        name="mean_offered_booking_delay",
        label="Mean offered booking delay",
        unit="days",
        direction="lower means offered appointments were sooner",
        description="Average offered delay among patients who received an offer, including balked patients.",
    ),
    "overall_balking_rate": MetricDefinition(
        name="overall_balking_rate",
        label="Overall balking rate",
        unit="rate",
        direction="higher means more offered patients rejected appointments",
        description="Balked patients divided by patients who received an appointment offer.",
    ),
    "access_advantage_class_1": MetricDefinition(
        name="access_advantage_class_1",
        label="Class 1 access advantage",
        unit="rate difference",
        direction="positive means Class 1 has a higher served rate than Class 2",
        description="Class 1 percent serviced minus Class 2 percent serviced.",
    ),
    "delay_advantage_class_1": MetricDefinition(
        name="delay_advantage_class_1",
        label="Class 1 delay advantage",
        unit="day difference",
        direction="positive means Class 1 has lower offered delay than Class 2",
        description="Class 2 mean offered booking delay minus Class 1 mean offered booking delay.",
    ),
}


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def aggregate_delay_metrics(result: SimulationResults) -> dict[str, float]:
    class_metrics = result.class_metrics.values()
    booked = sum(metrics.booked for metrics in class_metrics)
    offered = sum(metrics.offered for metrics in result.class_metrics.values())
    accepted_delay = sum(metrics.total_booking_delay for metrics in result.class_metrics.values())
    offered_delay = sum(metrics.total_offered_booking_delay for metrics in result.class_metrics.values())

    return {
        "mean_accepted_booking_delay": safe_divide(accepted_delay, booked),
        "mean_offered_booking_delay": safe_divide(offered_delay, offered),
    }


def outcome_totals(result: SimulationResults) -> dict[str, int]:
    totals = {
        "arrivals": 0,
        "booked": 0,
        "balked": 0,
        "no_offer": 0,
        "canceled": 0,
        "no_show": 0,
        "served": 0,
    }
    for metrics in result.class_metrics.values():
        for key in totals:
            totals[key] += getattr(metrics, key)

    resolved_booked = totals["canceled"] + totals["no_show"] + totals["served"]
    totals["offered"] = totals["booked"] + totals["balked"]
    totals["unresolved_booked"] = max(totals["booked"] - resolved_booked, 0)
    return totals


def outcome_rates_from_result(result: SimulationResults) -> dict[str, float]:
    totals = outcome_totals(result)
    arrivals = totals["arrivals"]

    rates: dict[str, float] = {
        "total_arrivals": arrivals,
        "total_booked": totals["booked"],
    }
    for outcome in ["served", "balked", "no_offer", "canceled", "no_show", "unresolved_booked"]:
        rates[f"{outcome}_rate"] = safe_divide(totals[outcome], arrivals)
    rates["lost_after_booking_rate"] = safe_divide(
        totals["canceled"] + totals["no_show"] + totals["unresolved_booked"],
        arrivals,
    )
    return rates


def result_metrics_from_result(
    result: SimulationResults,
    class_ids: tuple[int, int] = TWO_CLASS_IDS,
) -> dict[str, float]:
    class_1_id, class_2_id = class_ids
    c1 = result.class_metrics[class_1_id]
    c2 = result.class_metrics[class_2_id]
    totals = outcome_totals(result)

    class_1_delay = c1.mean_offered_booking_delay
    class_2_delay = c2.mean_offered_booking_delay
    class_1_balking_rate = safe_divide(c1.balked, c1.offered)
    class_2_balking_rate = safe_divide(c2.balked, c2.offered)

    return {
        "average_utilization": result.average_utilization,
        "overall_percent_serviced": result.overall_percent_serviced,
        **aggregate_delay_metrics(result),
        "class_1_percent_serviced": c1.percent_serviced,
        "class_2_percent_serviced": c2.percent_serviced,
        "overall_balking_rate": safe_divide(totals["balked"], totals["offered"]),
        "class_1_balking_rate": class_1_balking_rate,
        "class_2_balking_rate": class_2_balking_rate,
        "class_1_slot_utilization": safe_divide(c1.served, result.total_slots),
        "class_2_slot_utilization": safe_divide(c2.served, result.total_slots),
        "class_1_mean_offered_booking_delay": class_1_delay,
        "class_2_mean_offered_booking_delay": class_2_delay,
        "access_advantage_class_1": c1.percent_serviced - c2.percent_serviced,
        "balking_rate_gap_class_1": class_1_balking_rate - class_2_balking_rate,
        "delay_advantage_class_1": class_2_delay - class_1_delay,
    }


def aggregate_result_row(
    result: SimulationResults,
    fixed_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    totals = outcome_totals(result)
    row: dict[str, Any] = {
        "average_utilization": result.average_utilization,
        "overall_percent_serviced": result.overall_percent_serviced,
        "total_served": result.total_served,
        "total_value": result.total_value,
        **aggregate_delay_metrics(result),
        "overall_balking_rate": safe_divide(totals["balked"], totals["offered"]),
        "total_arrivals": totals["arrivals"],
        "total_booked": totals["booked"],
        "total_offered": totals["offered"],
        "total_balked": totals["balked"],
        "total_no_offer": totals["no_offer"],
        "total_canceled": totals["canceled"],
        "total_no_show": totals["no_show"],
        "total_unresolved_booked": totals["unresolved_booked"],
    }
    return {**(fixed_values or {}), **row}


def class_result_rows(
    result: SimulationResults,
    fixed_values: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    prefix = fixed_values or {}
    for class_id, metrics in result.class_metrics.items():
        rows.append(
            {
                **prefix,
                "class_id": class_id,
                "arrivals": metrics.arrivals,
                "booked": metrics.booked,
                "balked": metrics.balked,
                "offered": metrics.offered,
                "no_offer": metrics.no_offer,
                "canceled": metrics.canceled,
                "no_show": metrics.no_show,
                "served": metrics.served,
                "mean_accepted_booking_delay": metrics.mean_accepted_booking_delay,
                "mean_offered_booking_delay": metrics.mean_offered_booking_delay,
                "percent_serviced": metrics.percent_serviced,
                "slot_utilization": safe_divide(metrics.served, result.total_slots),
                "balking_rate": safe_divide(metrics.balked, metrics.offered),
                "total_booking_delay": metrics.total_booking_delay,
                "total_offered_booking_delay": metrics.total_offered_booking_delay,
            }
        )
    return rows


def metric_definition_rows(names: Iterable[str] | None = None) -> list[dict[str, str]]:
    selected = names or METRIC_DEFINITIONS.keys()
    return [METRIC_DEFINITIONS[name].__dict__.copy() for name in selected]

