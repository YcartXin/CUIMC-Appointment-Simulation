from __future__ import annotations

import math
import unittest

import pandas as pd

from experiments.hypothesis_common import (
    best_and_near_tie,
    build_config,
    classify_effect,
    flatten_result,
    paired_delta_ci,
    run_one,
)
from simulation.engine import ClinicAppointmentSimulation


class BuildConfigTest(unittest.TestCase):
    def test_defaults_match_baseline_convention(self) -> None:
        config = build_config(seed=1, lambda_1=10.0, lambda_2=10.0)
        self.assertEqual(config.slots_per_day, 32)
        self.assertEqual(config.horizon_days, 14)
        self.assertEqual(config.classes[1].cancel_prob, 0.10)
        self.assertEqual(config.classes[1].balk_prob.threshold, 9)
        self.assertEqual(config.classes[1].no_show_prob.threshold, 6)
        self.assertEqual(config.classes[1].standby_prob, 0.0)
        self.assertIsNone(config.reserved_class_id)
        self.assertFalse(config.same_day_cancellation_enabled)
        self.assertFalse(config.release_unused_reservation_same_day)

    def test_same_day_flags_flow_through(self) -> None:
        config = build_config(
            seed=1,
            lambda_1=10.0,
            lambda_2=10.0,
            reserved_class_id=1,
            reserved_slots_per_day=4,
            same_day_cancellation_enabled=True,
            release_unused_reservation_same_day=True,
        )
        self.assertTrue(config.same_day_cancellation_enabled)
        self.assertTrue(config.release_unused_reservation_same_day)

    def test_reservation_kwargs_flow_through(self) -> None:
        config = build_config(
            seed=1,
            lambda_1=10.0,
            lambda_2=10.0,
            reserved_class_id=1,
            reserved_slots_per_day=4,
            reserved_window_days=3,
        )
        self.assertEqual(config.reserved_class_id, 1)
        self.assertEqual(config.reserved_slots_per_day, 4)
        self.assertEqual(config.reserved_window_days, 3)

    def test_standby_kwargs_flow_through(self) -> None:
        config = build_config(
            seed=1,
            lambda_1=10.0,
            lambda_2=10.0,
            standby_prob_1=0.6,
            max_standby_days_1=5,
        )
        self.assertEqual(config.classes[1].standby_prob, 0.6)
        self.assertEqual(config.classes[1].max_standby_days, 5)
        self.assertEqual(config.classes[2].standby_prob, 0.0)
        self.assertIsNone(config.classes[2].max_standby_days)

    def test_asymmetric_no_show_thresholds(self) -> None:
        config = build_config(
            seed=1, lambda_1=10.0, lambda_2=10.0, noshow_threshold_1=6, noshow_threshold_2=12
        )
        self.assertEqual(config.classes[1].no_show_prob.threshold, 6)
        self.assertEqual(config.classes[2].no_show_prob.threshold, 12)


class RunOneAndFlattenTest(unittest.TestCase):
    def test_run_one_returns_labels_and_metrics(self) -> None:
        task = {
            "config_kwargs": {
                "lambda_1": 5.0,
                "lambda_2": 5.0,
                "measure_days": 10,
                "burn_in_days": 3,
                "cooldown_days": 3,
            },
            "seed": 42,
            "extra_cols": {"stage": "X", "cell_id": "test", "arm": "off", "seed": 42},
        }
        row = run_one(task)
        self.assertEqual(row["stage"], "X")
        self.assertEqual(row["seed"], 42)
        self.assertIn("average_utilization", row)
        self.assertIn("reserved_slot_fill_rate", row)
        self.assertIn("class_1_standby_joined", row)
        self.assertGreaterEqual(row["average_utilization"], 0.0)

    def test_flatten_result_includes_weighted_utilization(self) -> None:
        config = build_config(
            seed=11,
            lambda_1=8.0,
            lambda_2=8.0,
            measure_days=30,
            burn_in_days=5,
            cooldown_days=5,
        )
        result = ClinicAppointmentSimulation(config).run()
        row = flatten_result(result, seed=11)

        self.assertIn("weighted_utilization", row)
        c1 = result.class_metrics[1]
        c2 = result.class_metrics[2]
        w1, w2 = 2.0, 1.0
        expected = (
            w1 * (c1.served / c1.arrivals) + w2 * (c2.served / c2.arrivals)
        ) / (w1 + w2)
        self.assertAlmostEqual(row["weighted_utilization"], expected, places=9)
        # Bounded in [0, 1] since it's a weighted average of two rates
        # each already bounded in [0, 1].
        self.assertGreaterEqual(row["weighted_utilization"], 0.0)
        self.assertLessEqual(row["weighted_utilization"], 1.0)

    def test_flatten_result_includes_standby_diagnostics(self) -> None:
        config = build_config(
            seed=7,
            lambda_1=5.0,
            lambda_2=5.0,
            measure_days=10,
            burn_in_days=3,
            cooldown_days=3,
            standby_prob_1=1.0,
            max_standby_days_1=5,
        )
        result = ClinicAppointmentSimulation(config).run()
        row = flatten_result(result, seed=7)
        for key in (
            "class_1_standby_joined",
            "class_1_standby_recalled",
            "class_1_standby_expired",
            "class_1_standby_recall_rate",
            "class_1_mean_standby_wait_days",
            "class_1_mean_original_offered_delay_recalled",
        ):
            self.assertIn(key, row)


class StatsHelpersTest(unittest.TestCase):
    def test_paired_delta_ci_matches_manual_mean(self) -> None:
        on_values = [0.5, 0.6, 0.55]
        off_values = [0.4, 0.4, 0.4]
        mean, low, high, n = paired_delta_ci(on_values, off_values, seed=1)
        self.assertAlmostEqual(mean, sum(a - b for a, b in zip(on_values, off_values)) / 3, places=6)
        self.assertEqual(n, 3)
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)

    def test_paired_delta_ci_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            paired_delta_ci([0.1, 0.2], [0.1], seed=1)

    def test_classify_effect_supported_positive(self) -> None:
        status = classify_effect(0.02, 0.01, 0.03, expected_sign="positive", tolerance=0.005)
        self.assertEqual(status, "supported")

    def test_classify_effect_contradicted_positive(self) -> None:
        status = classify_effect(-0.02, -0.03, -0.01, expected_sign="positive", tolerance=0.005)
        self.assertEqual(status, "contradicted")

    def test_classify_effect_inconclusive_near_zero(self) -> None:
        status = classify_effect(0.001, -0.002, 0.004, expected_sign="positive", tolerance=0.005)
        self.assertEqual(status, "inconclusive")

    def test_classify_effect_inconclusive_when_ci_crosses_zero(self) -> None:
        # Mean clears the tolerance, but the interval still straddles zero.
        status = classify_effect(0.01, -0.001, 0.021, expected_sign="positive", tolerance=0.005)
        self.assertEqual(status, "inconclusive")

    def test_classify_effect_negative_direction(self) -> None:
        status = classify_effect(-0.02, -0.03, -0.01, expected_sign="negative", tolerance=0.005)
        self.assertEqual(status, "supported")

    def test_classify_effect_nan_is_inconclusive(self) -> None:
        status = classify_effect(math.nan, math.nan, math.nan, expected_sign="positive")
        self.assertEqual(status, "inconclusive")


class BestAndNearTieTest(unittest.TestCase):
    def test_identifies_best_and_near_tie_set(self) -> None:
        df = pd.DataFrame(
            {
                "background_id": ["bg1"] * 4,
                "Q": [1, 2, 4, 8],
                "window": [3, 3, 3, 3],
                "average_utilization": [0.80, 0.90, 0.895, 0.60],
            }
        )
        result = best_and_near_tie(
            df,
            group_cols=["background_id"],
            param_cols=["Q", "window"],
            value_col="average_utilization",
            tolerance=0.02,
        )
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertAlmostEqual(row["best_value"], 0.90)
        self.assertEqual(row["near_tie_count"], 2)  # Q=2 (0.90) and Q=4 (0.895) are within 2%

    def test_separate_groups_scored_independently(self) -> None:
        df = pd.DataFrame(
            {
                "background_id": ["bg1", "bg1", "bg2", "bg2"],
                "Q": [1, 2, 1, 2],
                "window": [3, 3, 3, 3],
                "average_utilization": [0.5, 0.9, 0.7, 0.71],
            }
        )
        result = best_and_near_tie(
            df,
            group_cols=["background_id"],
            param_cols=["Q", "window"],
            value_col="average_utilization",
            tolerance=0.01,
        )
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
