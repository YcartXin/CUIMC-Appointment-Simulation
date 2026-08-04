from __future__ import annotations

import unittest

import pandas as pd

from experiments import h1_short_horizon_reservation as h1
from experiments.hypothesis_common import build_config


class ThresholdPreservationTest(unittest.TestCase):
    def test_build_config_preserves_thresholds_when_requested(self) -> None:
        config = build_config(
            seed=1000,
            lambda_1=10.0,
            lambda_2=10.0,
            slots_per_day=30,
            horizon_days=6,
            balk_threshold_1=22,
            balk_threshold_2=22,
            noshow_threshold_1=20,
            noshow_threshold_2=14,
            cap_thresholds_to_horizon=False,
        )

        self.assertEqual(config.classes[1].balk_prob.threshold, 22)
        self.assertEqual(config.classes[2].balk_prob.threshold, 22)
        self.assertEqual(config.classes[1].no_show_prob.threshold, 20)
        self.assertEqual(config.classes[2].no_show_prob.threshold, 14)

    def test_build_config_retains_historical_capping_by_default(self) -> None:
        config = build_config(
            seed=1000,
            lambda_1=10.0,
            lambda_2=10.0,
            slots_per_day=30,
            horizon_days=6,
            balk_threshold_1=22,
            balk_threshold_2=22,
            noshow_threshold_1=20,
            noshow_threshold_2=14,
        )

        self.assertEqual(config.classes[1].balk_prob.threshold, 5)
        self.assertEqual(config.classes[2].balk_prob.threshold, 5)
        self.assertEqual(config.classes[1].no_show_prob.threshold, 5)
        self.assertEqual(config.classes[2].no_show_prob.threshold, 5)

    def test_h1_reads_optional_bank_flag(self) -> None:
        row = pd.Series(
            {
                "slots_per_day": 30,
                "lambda_1": 10.0,
                "lambda_2": 20.0,
                "cancel_1": 0.2,
                "cancel_2": 0.2,
                "balk_threshold_1": 22,
                "balk_low_1": 0.1,
                "balk_high_1": 0.2,
                "balk_threshold_2": 22,
                "balk_low_2": 0.1,
                "balk_high_2": 0.2,
                "noshow_threshold_1": 14,
                "noshow_low_1": 0.1,
                "noshow_high_1": 0.2,
                "noshow_threshold_2": 20,
                "noshow_low_2": 0.05,
                "noshow_high_2": 0.1,
                "cap_thresholds_to_horizon": False,
            }
        )

        kwargs = h1._row_config_kwargs(row)
        self.assertFalse(kwargs["cap_thresholds_to_horizon"])

    def test_h1_defaults_to_historical_capping_for_old_banks(self) -> None:
        row = pd.Series(
            {
                "slots_per_day": 30,
                "lambda_1": 10.0,
                "lambda_2": 20.0,
                "cancel_1": 0.2,
                "cancel_2": 0.2,
                "balk_threshold_1": 22,
                "balk_low_1": 0.1,
                "balk_high_1": 0.2,
                "balk_threshold_2": 22,
                "balk_low_2": 0.1,
                "balk_high_2": 0.2,
                "noshow_threshold_1": 14,
                "noshow_low_1": 0.1,
                "noshow_high_1": 0.2,
                "noshow_threshold_2": 20,
                "noshow_low_2": 0.05,
                "noshow_high_2": 0.1,
            }
        )

        kwargs = h1._row_config_kwargs(row)
        self.assertTrue(kwargs["cap_thresholds_to_horizon"])


if __name__ == "__main__":
    unittest.main()
