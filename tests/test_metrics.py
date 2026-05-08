from __future__ import annotations

import unittest

from analysis.metrics import (
    aggregate_delay_metrics,
    aggregate_result_row,
    class_result_rows,
    outcome_rates_from_result,
    result_metrics_from_result,
)
from simulation.model import ClassMetrics, SimulationResults, SlotMetrics


class MetricsTest(unittest.TestCase):
    def make_result(self) -> SimulationResults:
        class_1 = ClassMetrics(
            arrivals=10,
            booked=4,
            balked=2,
            no_offer=1,
            canceled=1,
            no_show=1,
            served=2,
            total_booking_delay=12.0,
            total_offered_booking_delay=30.0,
        )
        class_2 = ClassMetrics(arrivals=5)
        return SimulationResults(
            class_metrics={1: class_1, 2: class_2},
            slot_metrics=SlotMetrics(daily_utilization_sum=5.0, measured_days=10),
            total_slots=20,
            total_value=3.0,
            daily_summary_states=[],
            final_full_state=[],
        )

    def test_aggregate_delay_metrics_match_current_formulas(self) -> None:
        metrics = aggregate_delay_metrics(self.make_result())

        self.assertEqual(metrics["mean_accepted_booking_delay"], 3.0)
        self.assertEqual(metrics["mean_offered_booking_delay"], 5.0)

    def test_zero_denominators_return_zero(self) -> None:
        result = SimulationResults(
            class_metrics={1: ClassMetrics(), 2: ClassMetrics()},
            slot_metrics=SlotMetrics(),
            total_slots=0,
            total_value=0.0,
            daily_summary_states=[],
            final_full_state=[],
        )

        metrics = result_metrics_from_result(result)
        rates = outcome_rates_from_result(result)

        self.assertEqual(metrics["overall_balking_rate"], 0.0)
        self.assertEqual(metrics["class_1_slot_utilization"], 0.0)
        self.assertEqual(metrics["access_advantage_class_1"], 0.0)
        self.assertEqual(rates["served_rate"], 0.0)
        self.assertEqual(rates["lost_after_booking_rate"], 0.0)

    def test_class_gap_sign_conventions(self) -> None:
        metrics = result_metrics_from_result(self.make_result())

        self.assertAlmostEqual(metrics["access_advantage_class_1"], 0.2)
        self.assertAlmostEqual(metrics["balking_rate_gap_class_1"], 2 / 6)
        self.assertAlmostEqual(metrics["delay_advantage_class_1"], -5.0)

    def test_outcome_rates_and_rows(self) -> None:
        result = self.make_result()
        rates = outcome_rates_from_result(result)
        aggregate_row = aggregate_result_row(result, {"seed": 1})
        class_rows = class_result_rows(result, {"seed": 1})

        self.assertAlmostEqual(rates["served_rate"], 2 / 15)
        self.assertAlmostEqual(rates["balked_rate"], 2 / 15)
        self.assertAlmostEqual(rates["no_offer_rate"], 1 / 15)
        self.assertAlmostEqual(rates["canceled_rate"], 1 / 15)
        self.assertAlmostEqual(rates["no_show_rate"], 1 / 15)
        self.assertEqual(rates["unresolved_booked_rate"], 0.0)
        self.assertEqual(aggregate_row["total_offered"], 6)
        self.assertEqual(aggregate_row["total_value"], 3.0)
        self.assertEqual(len(class_rows), 2)
        self.assertAlmostEqual(class_rows[0]["slot_utilization"], 0.1)
        self.assertAlmostEqual(class_rows[0]["balking_rate"], 2 / 6)


if __name__ == "__main__":
    unittest.main()

