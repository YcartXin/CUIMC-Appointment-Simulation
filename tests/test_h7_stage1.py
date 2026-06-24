from __future__ import annotations

import unittest

import pandas as pd

from experiments.robustness.h7_stage1 import (
    _gap_effect_rows,
    _scenario_effect_rows,
    prepare_h7_backgrounds,
    valid_h7_gaps,
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
        for gap in (0.05, 0.10, 0.20, 0.30, 0.50):
            if supported:
                pre_absolute_gap = 0.020 + gap * 0.10
                post_absolute_gap = 0.005 + gap * 0.02
            else:
                pre_absolute_gap = 0.005 + gap * 0.02
                post_absolute_gap = 0.020 + gap * 0.10

            for arm, absolute_gap in (
                ("pre", pre_absolute_gap),
                ("post", post_absolute_gap),
            ):
                rows.append(
                    {
                        "background_id": background_id,
                        "gap_magnitude_focal": gap,
                        "gap_location_arm": arm,
                        "seed": seed,
                        "absolute_served_rate_gap": absolute_gap,
                        "class2_low_regime_offer_share": 0.50,
                        "class2_high_regime_offer_share": 0.50,
                    }
                )
    return pd.DataFrame(rows)


class TestH7Stage1(unittest.TestCase):
    def test_valid_gap_levels(self):
        self.assertEqual(
            valid_h7_gaps(0.0, 0.5),
            (0.05, 0.10, 0.20, 0.30, 0.50),
        )
        self.assertEqual(
            valid_h7_gaps(0.3, 0.5),
            (0.05, 0.10, 0.20),
        )
        self.assertEqual(valid_h7_gaps(0.5, 0.5), ())

    def test_background_deduplication_removes_class2_rates(self):
        first = _scenario_row()
        second = _scenario_row(
            scenario_id="S0002",
            balk_low_class2=0.3,
            balk_high_class2=0.7,
        )
        result = prepare_h7_backgrounds(pd.DataFrame([first, second]))
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0]["source_scenario_count"]), 2)

    def test_supported_pattern(self):
        design = prepare_h7_backgrounds(pd.DataFrame([_scenario_row()]))
        background_id = str(design.iloc[0]["background_id"])
        raw = _raw(background_id, supported=True)
        gaps = _gap_effect_rows(design, raw)
        scenarios = _scenario_effect_rows(design, raw, gaps)
        self.assertEqual(scenarios.iloc[0]["classification"], "supported")

    def test_reversed_pattern(self):
        design = prepare_h7_backgrounds(pd.DataFrame([_scenario_row()]))
        background_id = str(design.iloc[0]["background_id"])
        raw = _raw(background_id, supported=False)
        gaps = _gap_effect_rows(design, raw)
        scenarios = _scenario_effect_rows(design, raw, gaps)
        self.assertEqual(scenarios.iloc[0]["classification"], "reversed")

    def test_fewer_than_two_gaps_is_design_inactive(self):
        design = prepare_h7_backgrounds(
            pd.DataFrame(
                [
                    _scenario_row(
                        balk_low_class1=0.5,
                        balk_high_class1=0.5,
                    )
                ]
            )
        )
        self.assertFalse(bool(design.iloc[0]["h7_design_active"]))


if __name__ == "__main__":
    unittest.main()
