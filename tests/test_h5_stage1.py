from __future__ import annotations

import unittest

import pandas as pd

from experiments.robustness.h5_stage1 import (
    _scenario_effect_rows,
    _target_effect_rows,
    prepare_h5_backgrounds,
    valid_h5_steps,
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


def _raw(background_id, supported=True):
    rows = []
    for seed in range(1000, 1020):
        for step in (0.0, 0.1, 0.3, 0.5):
            if step == 0.0:
                accepted = 4.0
                offered = 4.2
                served = 0.60
                balked = 0
            elif supported:
                accepted = 4.0 - 2.0 * step
                offered = 4.2 - 0.2 * step
                served = 0.60 - 0.05 * step / 0.1
                balked = int(1000 * step)
            else:
                accepted = 4.0 + 2.0 * step
                offered = 4.2 + 0.1 * step
                served = 0.60 + 0.05 * step / 0.1
                balked = int(1000 * step)

            rows.append(
                {
                    "background_id": background_id,
                    "balk_step_class1_focal": step,
                    "seed": seed,
                    "class_1_mean_accepted_booking_delay": accepted,
                    "class_1_mean_offered_booking_delay": offered,
                    "class_1_percent_serviced": served,
                    "class_1_balked": balked,
                    "class_1_offered": 10000,
                }
            )
    return pd.DataFrame(rows)


class TestH5Stage1(unittest.TestCase):
    def test_valid_steps(self):
        self.assertEqual(valid_h5_steps(0.0), (0.0, 0.1, 0.3, 0.5))
        self.assertEqual(valid_h5_steps(0.5), (0.0, 0.1))
        self.assertEqual(valid_h5_steps(0.7), (0.0,))

    def test_background_deduplication(self):
        first = _scenario_row()
        second = _scenario_row(
            scenario_id="S0002",
            balk_high_class1=0.7,
        )
        result = prepare_h5_backgrounds(pd.DataFrame([first, second]))
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0]["source_scenario_count"]), 2)

    def test_supported_pattern(self):
        design = prepare_h5_backgrounds(pd.DataFrame([_scenario_row()]))
        background_id = str(design.iloc[0]["background_id"])
        target = _target_effect_rows(design, _raw(background_id, supported=True))
        scenario = _scenario_effect_rows(design, target)
        self.assertEqual(scenario.iloc[0]["classification"], "supported")

    def test_reversed_pattern(self):
        design = prepare_h5_backgrounds(pd.DataFrame([_scenario_row()]))
        background_id = str(design.iloc[0]["background_id"])
        target = _target_effect_rows(design, _raw(background_id, supported=False))
        scenario = _scenario_effect_rows(design, target)
        self.assertEqual(scenario.iloc[0]["classification"], "reversed")

    def test_no_valid_primary_step_is_design_inactive(self):
        design = prepare_h5_backgrounds(
            pd.DataFrame([_scenario_row(balk_low_class1=0.7)])
        )
        self.assertFalse(bool(design.iloc[0]["h5_design_active"]))


if __name__ == "__main__":
    unittest.main()
