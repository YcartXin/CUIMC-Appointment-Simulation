from __future__ import annotations

import unittest

import pandas as pd

from experiments.robustness.h4_stage1 import (
    FOCAL_LEVELS,
    _classify_curve,
    _classification_for_shape,
    demand_regime,
    prepare_h4_backgrounds,
)


def _scenario_row(**updates):
    row = {
        "scenario_id": "S0001",
        "scenario_type": "symmetric",
        "parent_scenario_id": "",
        "design_note": "",
        "asymmetric_dimensions": "",
        "rho": 3.1,
        "class1_share": 0.5,
        "slots_per_day": 32,
        "lambda_total": 99.2,
        "lambda_class1": 49.6,
        "lambda_class2": 49.6,
        "horizon_class1": 14,
        "horizon_class2": 14,
        "cancel_class1": 0.1,
        "cancel_class2": 0.1,
        "balk_threshold_class1": 9,
        "balk_threshold_class2": 9,
        "balk_low_class1": 0.0,
        "balk_low_class2": 0.0,
        "balk_high_class1": 0.5,
        "balk_high_class2": 0.5,
        "noshow_threshold_class1": 6,
        "noshow_threshold_class2": 6,
        "noshow_low_class1": 0.0,
        "noshow_low_class2": 0.0,
        "noshow_high_class1": 0.3,
        "noshow_high_class2": 0.3,
    }
    row.update(updates)
    return row


def _raw_curve(delays):
    rows = []
    for seed in range(1000, 1020):
        for level, delay in zip(FOCAL_LEVELS, delays):
            rows.append(
                {
                    "background_id": "H4B0001",
                    "seed": seed,
                    "common_balk_high_focal": level,
                    "mean_offered_booking_delay": delay,
                    "overall_balk_rate_per_arrival": level * 0.20,
                }
            )
    return pd.DataFrame(rows)


class TestH4Stage1(unittest.TestCase):
    def test_demand_regimes(self):
        self.assertEqual(demand_regime(1.25), "low")
        self.assertEqual(demand_regime(2.0), "boundary")
        self.assertEqual(demand_regime(3.1), "high")

    def test_background_deduplication_removes_focal_rates(self):
        first = _scenario_row()
        second = _scenario_row(
            scenario_id="S0002",
            balk_low_class1=0.3,
            balk_low_class2=0.1,
            balk_high_class1=0.7,
            balk_high_class2=0.5,
        )
        result = prepare_h4_backgrounds(pd.DataFrame([first, second]))
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0]["source_scenario_count"]), 2)

    def test_hump_curve(self):
        result = _classify_curve(
            _raw_curve([1.0, 1.3, 1.8, 1.4, 1.1])
        )
        self.assertEqual(result["curve_shape"], "hump")

    def test_u_shaped_curve(self):
        result = _classify_curve(
            _raw_curve([1.8, 1.4, 1.0, 1.3, 1.7])
        )
        self.assertEqual(result["curve_shape"], "u_shaped")

    def test_high_demand_hump_supported(self):
        classification, reason = _classification_for_shape(
            regime="high",
            shape="hump",
            exposure_active=True,
        )
        self.assertEqual(classification, "supported")
        self.assertEqual(reason, "")

    def test_lower_demand_is_outside_scope(self):
        classification, reason = _classification_for_shape(
            regime="low",
            shape="hump",
            exposure_active=True,
        )
        self.assertEqual(classification, "inactive")
        self.assertEqual(reason, "outside_heavy_oversubscription_scope")


if __name__ == "__main__":
    unittest.main()
