from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.robustness.h3_stage1 import (
    _scenario_effect_rows,
    _threshold_effect_rows,
    prepare_h3_backgrounds,
    valid_h3_thresholds,
)
from experiments.robustness.scenario_space import PARAMETER_COLUMNS


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


def _raw_for_pattern(background_id, threshold_to_delta):
    rows = []
    for seed in range(1000, 1020):
        for threshold, delta in threshold_to_delta.items():
            rows.append(
                {
                    "background_id": background_id,
                    "noshow_threshold_class1_focal": threshold,
                    "arm": "low",
                    "seed": seed,
                    "average_utilization": 0.90,
                    "class_1_no_show_rate_per_arrival": 0.00,
                }
            )
            rows.append(
                {
                    "background_id": background_id,
                    "noshow_threshold_class1_focal": threshold,
                    "arm": "high",
                    "seed": seed,
                    "average_utilization": 0.90 + delta,
                    "class_1_no_show_rate_per_arrival": 0.10,
                }
            )
    return pd.DataFrame(rows)


class TestH3Stage1(unittest.TestCase):
    def test_valid_threshold_rule(self):
        self.assertEqual(valid_h3_thresholds(7), (4,))
        self.assertEqual(valid_h3_thresholds(14), (4, 6, 9, 12))

    def test_background_deduplication_removes_focal_columns(self):
        first = _scenario_row()
        second = _scenario_row(
            scenario_id="S0002",
            noshow_threshold_class1=12,
            noshow_high_class1=0.7,
        )
        result = prepare_h3_backgrounds(pd.DataFrame([first, second]))
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0]["source_scenario_count"]), 2)

    def test_supported_pattern(self):
        design = prepare_h3_backgrounds(pd.DataFrame([_scenario_row()]))
        background_id = str(design.iloc[0]["background_id"])
        raw = _raw_for_pattern(
            background_id,
            {4: -0.08, 6: -0.06, 9: -0.03, 12: -0.01},
        )
        threshold_effects = _threshold_effect_rows(design, raw)
        scenario_effects = _scenario_effect_rows(
            design, raw, threshold_effects
        )
        self.assertEqual(
            scenario_effects.iloc[0]["classification"], "supported"
        )

    def test_reversed_pattern(self):
        design = prepare_h3_backgrounds(pd.DataFrame([_scenario_row()]))
        background_id = str(design.iloc[0]["background_id"])
        raw = _raw_for_pattern(
            background_id,
            {4: -0.01, 6: -0.03, 9: -0.06, 12: -0.08},
        )
        threshold_effects = _threshold_effect_rows(design, raw)
        scenario_effects = _scenario_effect_rows(
            design, raw, threshold_effects
        )
        self.assertEqual(
            scenario_effects.iloc[0]["classification"], "reversed"
        )

    def test_design_inactive_without_rate_contrast(self):
        design = prepare_h3_backgrounds(
            pd.DataFrame([_scenario_row(noshow_low_class1=0.7)])
        )
        self.assertFalse(bool(design.iloc[0]["h3_design_active"]))
        self.assertIn(
            "no_valid_post_threshold_increase",
            design.iloc[0]["h3_design_inactive_reason"],
        )


if __name__ == "__main__":
    unittest.main()
