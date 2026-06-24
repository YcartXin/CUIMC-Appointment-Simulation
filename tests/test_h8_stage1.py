from __future__ import annotations

import unittest

import pandas as pd

from experiments.robustness.h8_stage1 import (
    _scenario_effect_rows,
    prepare_h8_backgrounds,
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
        baseline = 0.60
        if supported:
            step = 0.59
            gap = 0.56
        else:
            step = 0.55
            gap = 0.59

        for arm, served in (
            ("baseline", baseline),
            ("step_up", step),
            ("gap_up", gap),
        ):
            rows.append(
                {
                    "background_id": background_id,
                    "h8_arm": arm,
                    "seed": seed,
                    "class_1_percent_serviced": served,
                    "class1_low_regime_offer_share": 0.40,
                    "class1_high_regime_offer_share": 0.60,
                    "class2_high_regime_offer_share": 0.50,
                }
            )
    return pd.DataFrame(rows)


class TestH8Stage1(unittest.TestCase):
    def test_background_deduplication_removes_all_focal_rates(self):
        first = _scenario_row()
        second = _scenario_row(
            scenario_id="S0002",
            balk_low_class1=0.3,
            balk_high_class1=0.7,
            balk_low_class2=0.1,
            balk_high_class2=0.5,
        )
        result = prepare_h8_backgrounds(pd.DataFrame([first, second]))
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0]["source_scenario_count"]), 2)

    def test_start_cells_are_balanced_and_valid(self):
        rows = [
            _scenario_row(scenario_id=f"S{i:04d}", rho=3.1 + i * 0.001)
            for i in range(25)
        ]
        result = prepare_h8_backgrounds(pd.DataFrame(rows))
        cells = set(
            zip(
                result["start_within_step"],
                result["start_post_gap"],
            )
        )
        self.assertEqual(len(cells), 25)
        self.assertTrue((result["baseline_class1_low"] >= 0.1 - 1e-12).all())
        self.assertTrue((result["baseline_class2_high"] >= 0.1 - 1e-12).all())

    def test_supported_pattern(self):
        design = prepare_h8_backgrounds(pd.DataFrame([_scenario_row()]))
        background_id = str(design.iloc[0]["background_id"])
        effects = _scenario_effect_rows(
            design,
            _raw(background_id, supported=True),
        )
        self.assertEqual(effects.iloc[0]["classification"], "supported")

    def test_reversed_pattern(self):
        design = prepare_h8_backgrounds(pd.DataFrame([_scenario_row()]))
        background_id = str(design.iloc[0]["background_id"])
        effects = _scenario_effect_rows(
            design,
            _raw(background_id, supported=False),
        )
        self.assertEqual(effects.iloc[0]["classification"], "reversed")

    def test_insufficient_exposure_is_inactive(self):
        design = prepare_h8_backgrounds(pd.DataFrame([_scenario_row()]))
        background_id = str(design.iloc[0]["background_id"])
        raw = _raw(background_id, supported=True)
        raw["class2_high_regime_offer_share"] = 0.0
        effects = _scenario_effect_rows(design, raw)
        self.assertEqual(effects.iloc[0]["classification"], "inactive")


if __name__ == "__main__":
    unittest.main()
