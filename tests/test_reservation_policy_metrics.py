from __future__ import annotations

import math
import unittest

from analysis.reservation_policy_metrics import (
    compute_reservation_policy_metrics,
    reservation_policy_metrics_from_result,
)
from simulation.model import ClassMetrics, SimulationResults, SlotMetrics


class ReservationPolicyMetricsTest(unittest.TestCase):
    def counts(self) -> dict[str, float]:
        return {
            "Y1": 60,
            "Y2": 30,
            "A1": 100,
            "A2": 50,
            "S": 200,
            "offered_1": 80,
            "offered_2": 40,
            "sum_tau_offered_1": 160,
            "sum_tau_offered_2": 120,
            "no_offer_1": 20,
            "no_offer_2": 10,
        }

    def test_requested_formulas(self) -> None:
        metrics = compute_reservation_policy_metrics(
            **self.counts(),
            w1=2,
            w2=1,
        )
        self.assertAlmostEqual(metrics["Obj_util_raw"], 0.75)
        self.assertAlmostEqual(metrics["Obj_util_norm"], 0.25)
        self.assertAlmostEqual(metrics["Obj_service_raw"], 1.8)
        self.assertAlmostEqual(metrics["Obj_service_norm"], 0.6)
        self.assertAlmostEqual(metrics["T_wait_offered"], 2.2)
        self.assertAlmostEqual(metrics["class_1_service_rate"], 0.6)
        self.assertAlmostEqual(metrics["class_2_service_rate"], 0.6)
        self.assertAlmostEqual(metrics["class_1_no_offer_rate"], 0.2)
        self.assertAlmostEqual(metrics["class_2_no_offer_rate"], 0.2)
        self.assertAlmostEqual(metrics["class_1_avg_offered_wait"], 2.0)
        self.assertAlmostEqual(metrics["class_2_avg_offered_wait"], 3.0)

    def test_zero_denominators_are_nan(self) -> None:
        counts = self.counts()
        counts.update(A1=0, offered_1=0, offered_2=0, S=0)
        metrics = compute_reservation_policy_metrics(**counts, w1=1, w2=1)
        self.assertTrue(math.isnan(metrics["Obj_util_raw"]))
        self.assertTrue(math.isnan(metrics["Obj_service_raw"]))
        self.assertTrue(math.isnan(metrics["T_wait_offered"]))
        self.assertTrue(math.isnan(metrics["class_1_service_rate"]))
        self.assertTrue(math.isnan(metrics["class_1_avg_offered_wait"]))

    def test_zero_weight_ignores_undefined_class_rate(self) -> None:
        counts = self.counts()
        counts["A1"] = 0
        metrics = compute_reservation_policy_metrics(**counts, w1=0, w2=1)
        self.assertAlmostEqual(metrics["Obj_service_norm"], 0.6)

    def test_invalid_weights(self) -> None:
        with self.assertRaises(ValueError):
            compute_reservation_policy_metrics(**self.counts(), w1=-1, w2=1)
        with self.assertRaises(ValueError):
            compute_reservation_policy_metrics(**self.counts(), w1=0, w2=0)

    def test_result_wrapper(self) -> None:
        result = SimulationResults(
            class_metrics={
                1: ClassMetrics(
                    arrivals=100,
                    booked=70,
                    balked=10,
                    no_offer=20,
                    served=60,
                    total_offered_booking_delay=160,
                ),
                2: ClassMetrics(
                    arrivals=50,
                    booked=35,
                    balked=5,
                    no_offer=10,
                    served=30,
                    total_offered_booking_delay=120,
                ),
            },
            slot_metrics=SlotMetrics(),
            total_slots=200,
            total_value=0,
            daily_summary_states=[],
            final_full_state=[],
        )
        direct = compute_reservation_policy_metrics(**self.counts(), w1=2, w2=1)
        wrapped = reservation_policy_metrics_from_result(result, w1=2, w2=1)
        self.assertEqual(direct, wrapped)


if __name__ == "__main__":
    unittest.main()
