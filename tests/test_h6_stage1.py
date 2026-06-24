from __future__ import annotations

import unittest

import pandas as pd

from experiments.robustness.h6_stage1 import (
    _scenario_effect_rows,
    _transition_effect_rows,
    prepare_h6_backgrounds,
    valid_h6_thresholds,
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
        "horizon_class1": 7,
        "horizon_class2": 7,
        "cancel_class1": 0.1,
        "cancel_class2": 0.1,
        "balk_threshold_class1": 4,
        "balk_threshold_class2": 4,
        "balk_low_class1": 0.0,
        "balk_low_class2": 0.0,
        "balk_high_class1": 0.5,
        "balk_high_class2": 0.5,
        "noshow_threshold_class1": 4,
        "noshow_threshold_class2": 4,
        "noshow_low_class1": 0.0,
        "noshow_low_class2": 0.0,
        "noshow_high_class1": 0.3,
        "noshow_high_class2": 0.3,
    }
    row.update(updates)
    return row


def _raw(background_id: str, supported: bool) -> pd.DataFrame:
    rows = []
    thresholds = range(0, 6)
    if supported:
        bucket_masses = {1: 0.01, 2: 0.02, 3: 0.04, 4: 0.08, 5: 0.12}
        served = {0: 0.60, 1: 0.601, 2: 0.603, 3: 0.608, 4: 0.620, 5: 0.640}
    else:
        # Reversal fixture: bucket mass decreases while the absolute
        # served-rate jump increases.
        bucket_masses = {
            1: 0.12,
            2: 0.08,
            3: 0.04,
            4: 0.02,
            5: 0.01,
        }
        served = {
            0: 0.600,
            1: 0.601,
            2: 0.603,
            3: 0.608,
            4: 0.620,
            5: 0.640,
        }
    for seed in range(1000, 1020):
        for threshold in thresholds:
            counts = {}
            for delay, mass in bucket_masses.items():
                counts[str(delay)] = int(round(mass * 10000))
            rows.append(
                {
                    "background_id": background_id,
                    "balk_threshold_class1_focal": threshold,
                    "seed": seed,
                    "class_1_percent_serviced": served[threshold],
                    "class_1_offered": 10000,
                    "class_1_offered_delay_counts_json": __import__("json").dumps(counts),
                }
            )
    return pd.DataFrame(rows)


class TestH6Stage1(unittest.TestCase):
    def test_dense_threshold_range(self):
        self.assertEqual(valid_h6_thresholds(7), (0, 1, 2, 3, 4, 5))
        self.assertEqual(len(valid_h6_thresholds(14)), 13)

    def test_background_deduplication(self):
        first = _scenario_row()
        second = _scenario_row(
            scenario_id="S0002",
            balk_threshold_class1=1,
        )
        result = prepare_h6_backgrounds(pd.DataFrame([first, second]))
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0]["source_scenario_count"]), 2)

    def test_supported_bucket_relationship(self):
        design = prepare_h6_backgrounds(pd.DataFrame([_scenario_row()]))
        background_id = str(design.iloc[0]["background_id"])
        transitions = _transition_effect_rows(
            design,
            _raw(background_id, supported=True),
        )
        scenarios = _scenario_effect_rows(design, transitions)
        self.assertEqual(scenarios.iloc[0]["classification"], "supported")

    def test_reversed_bucket_relationship(self):
        design = prepare_h6_backgrounds(pd.DataFrame([_scenario_row()]))
        background_id = str(design.iloc[0]["background_id"])
        transitions = _transition_effect_rows(
            design,
            _raw(background_id, supported=False),
        )
        scenarios = _scenario_effect_rows(design, transitions)
        self.assertEqual(scenarios.iloc[0]["classification"], "reversed")

    def test_no_balking_step_is_inactive(self):
        design = prepare_h6_backgrounds(
            pd.DataFrame(
                [
                    _scenario_row(
                        balk_low_class1=0.3,
                        balk_high_class1=0.3,
                    )
                ]
            )
        )
        self.assertFalse(bool(design.iloc[0]["h6_design_active"]))


if __name__ == "__main__":
    unittest.main()
