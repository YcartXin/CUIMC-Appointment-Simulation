from __future__ import annotations

import math
from typing import Any

from simulation.model import SimulationResults


METRIC_COLUMNS = (
    "Obj_util_raw",
    "Obj_util_norm",
    "Obj_service_raw",
    "Obj_service_norm",
    "T_wait_offered",
    "class_1_service_rate",
    "class_2_service_rate",
    "class_1_no_offer_rate",
    "class_2_no_offer_rate",
    "class_1_avg_offered_wait",
    "class_2_avg_offered_wait",
)


def ratio_or_nan(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def compute_reservation_policy_metrics(
    *,
    Y1: float,
    Y2: float,
    A1: float,
    A2: float,
    S: float,
    offered_1: float,
    offered_2: float,
    sum_tau_offered_1: float,
    sum_tau_offered_2: float,
    no_offer_1: float,
    no_offer_2: float,
    w1: float,
    w2: float,
) -> dict[str, float]:
    """Compute weighted reservation-policy objectives from two-class counts."""
    if w1 < 0 or w2 < 0:
        raise ValueError("Objective weights must be nonnegative.")
    weight_sum = w1 + w2
    if weight_sum <= 0:
        raise ValueError("At least one objective weight must be positive.")

    class_1_service_rate = ratio_or_nan(Y1, A1)
    class_2_service_rate = ratio_or_nan(Y2, A2)
    class_1_no_offer_rate = ratio_or_nan(no_offer_1, A1)
    class_2_no_offer_rate = ratio_or_nan(no_offer_2, A2)
    class_1_avg_offered_wait = ratio_or_nan(
        sum_tau_offered_1,
        offered_1,
    )
    class_2_avg_offered_wait = ratio_or_nan(
        sum_tau_offered_2,
        offered_2,
    )

    if S:
        obj_util_raw = w1 * Y1 / S + w2 * Y2 / S
        obj_util_norm = obj_util_raw / weight_sum
    else:
        obj_util_raw = math.nan
        obj_util_norm = math.nan

    weighted_service_terms = []
    for weight, rate in (
        (w1, class_1_service_rate),
        (w2, class_2_service_rate),
    ):
        if weight == 0:
            continue
        if math.isnan(rate):
            obj_service_raw = math.nan
            break
        weighted_service_terms.append(weight * rate)
    else:
        obj_service_raw = sum(weighted_service_terms)
    obj_service_norm = (
        obj_service_raw / weight_sum
        if not math.isnan(obj_service_raw)
        else math.nan
    )

    weighted_offers = w1 * offered_1 + w2 * offered_2
    weighted_offered_delay = (
        w1 * sum_tau_offered_1 + w2 * sum_tau_offered_2
    )
    t_wait_offered = ratio_or_nan(weighted_offered_delay, weighted_offers)

    return {
        "Obj_util_raw": obj_util_raw,
        "Obj_util_norm": obj_util_norm,
        "Obj_service_raw": obj_service_raw,
        "Obj_service_norm": obj_service_norm,
        "T_wait_offered": t_wait_offered,
        "class_1_service_rate": class_1_service_rate,
        "class_2_service_rate": class_2_service_rate,
        "class_1_no_offer_rate": class_1_no_offer_rate,
        "class_2_no_offer_rate": class_2_no_offer_rate,
        "class_1_avg_offered_wait": class_1_avg_offered_wait,
        "class_2_avg_offered_wait": class_2_avg_offered_wait,
    }


def reservation_policy_metrics_from_result(
    result: SimulationResults,
    *,
    w1: float,
    w2: float,
) -> dict[str, float]:
    c1 = result.class_metrics[1]
    c2 = result.class_metrics[2]
    return compute_reservation_policy_metrics(
        Y1=c1.served,
        Y2=c2.served,
        A1=c1.arrivals,
        A2=c2.arrivals,
        S=result.total_slots,
        offered_1=c1.offered,
        offered_2=c2.offered,
        sum_tau_offered_1=c1.total_offered_booking_delay,
        sum_tau_offered_2=c2.total_offered_booking_delay,
        no_offer_1=c1.no_offer,
        no_offer_2=c2.no_offer,
        w1=w1,
        w2=w2,
    )


def score_simulation_row(
    row: dict[str, Any],
    *,
    w1: float,
    w2: float,
) -> dict[str, Any]:
    metrics = compute_reservation_policy_metrics(
        Y1=row["Y1"],
        Y2=row["Y2"],
        A1=row["A1"],
        A2=row["A2"],
        S=row["S"],
        offered_1=row["offered_1"],
        offered_2=row["offered_2"],
        sum_tau_offered_1=row["sum_tau_offered_1"],
        sum_tau_offered_2=row["sum_tau_offered_2"],
        no_offer_1=row["no_offer_1"],
        no_offer_2=row["no_offer_2"],
        w1=w1,
        w2=w2,
    )
    return {**row, "w1": w1, "w2": w2, **metrics}
