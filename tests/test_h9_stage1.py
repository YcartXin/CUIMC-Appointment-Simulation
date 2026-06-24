from __future__ import annotations

import unittest

import pandas as pd

from experiments.robustness.h9_stage1 import (
    _scenario_effect_rows,
    feasible_baseline_probabilities,
    prepare_h9_backgrounds,
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


def _raw(background_id: str, *, supported: bool) -> pd.DataFrame:
    rows = []
    for seed in range(1000, 1020):
        baseline_u = 0.90
        baseline_c = 0.00

        if supported:
            common_u = 0.84
            gap_1_u = 0.89
            gap_2_u = 0.89
            common_c = 0.005
            gap_1_c = 0.040
            gap_2_c = -0.040
        else:
            common_u = 0.89
            gap_1_u = 0.84
            gap_2_u = 0.84
            common_c = 0.040
            gap_1_c = 0.005
            gap_2_c = -0.005

        arm_values = {
            "baseline": (baseline_u, baseline_c, 100, 100),
            "common_up": (common_u, common_c, 200, 200),
            "gap_c1_higher": (gap_1_u, gap_1_c, 150, 50),
            "gap_c2_higher": (gap_2_u, gap_2_c, 50, 150),
        }

        for arm, (utilization, served_gap, no_show_1, no_show_2) in arm_values.items():
            rows.append(
                {
                    "background_id": background_id,
                    "h9_arm": arm,
                    "seed": seed,
                    "average_utilization": utilization,
                    "served_rate_gap": served_gap,
                    "class_1_arrivals": 10000,
                    "class_2_arrivals": 10000,
                    "class_1_booked": 10000,
                    "class_2_booked": 10000,
                    "class_1_no_show": no_show_1,
                    "class_2_no_show": no_show_2,
                }
            )

    return pd.DataFrame(rows)


class TestH9Stage1(unittest.TestCase):
    def test_feasible_probabilities(self):
        self.assertEqual(
            feasible_baseline_probabilities(0.0, 0.0),
            (0.10, 0.30, 0.50, 0.70, 0.80),
        )
        self.assertEqual(
            feasible_baseline_probabilities(0.5, 0.3),
            (0.70, 0.80),
        )
        self.assertEqual(
            feasible_baseline_probabilities(0.7, 0.7),
            (0.80,),
        )

    def test_background_deduplication_removes_high_rates(self):
        first = _scenario_row()
        second = _scenario_row(
            scenario_id="S0002",
            noshow_high_class1=0.7,
            noshow_high_class2=0.5,
        )
        result = prepare_h9_backgrounds(pd.DataFrame([first, second]))
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0]["source_scenario_count"]), 2)

    def test_supported_pattern(self):
        design = prepare_h9_backgrounds(pd.DataFrame([_scenario_row()]))
        background_id = str(design.iloc[0]["background_id"])
        effects = _scenario_effect_rows(
            design,
            _raw(background_id, supported=True),
        )
        self.assertEqual(effects.iloc[0]["classification"], "supported")

    def test_reversed_pattern(self):
        design = prepare_h9_backgrounds(pd.DataFrame([_scenario_row()]))
        background_id = str(design.iloc[0]["background_id"])
        effects = _scenario_effect_rows(
            design,
            _raw(background_id, supported=False),
        )
        self.assertEqual(effects.iloc[0]["classification"], "reversed")

    def test_low_exposure_is_inactive(self):
        design = prepare_h9_backgrounds(pd.DataFrame([_scenario_row()]))
        background_id = str(design.iloc[0]["background_id"])
        raw = _raw(background_id, supported=True)
        common_mask = raw["h9_arm"].eq("common_up")
        baseline_mask = raw["h9_arm"].eq("baseline")
        raw.loc[common_mask, "class_1_no_show"] = 101
        raw.loc[baseline_mask, "class_1_no_show"] = 100
        effects = _scenario_effect_rows(design, raw)
        self.assertEqual(effects.iloc[0]["classification"], "inactive")


if __name__ == "__main__":
    unittest.main()
