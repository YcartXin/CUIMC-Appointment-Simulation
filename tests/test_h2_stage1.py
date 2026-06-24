from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.robustness.h2_stage1 import (
    BASELINE_TARGET_SENTINEL,
    _aggregate_scenario_effects,
    classify_stage1,
    prepare_h2_backgrounds,
)


class H2BackgroundTests(unittest.TestCase):
    def _scenario(self) -> dict:
        return {
            "scenario_id": "A001",
            "scenario_type": "anchor",
            "parent_scenario_id": "",
            "design_note": "test",
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
            "asymmetric_dimensions": "",
        }

    def test_deduplicates_all_class1_focal_loss_parameters(self) -> None:
        first = self._scenario()
        second = dict(
            first,
            scenario_id="A002",
            balk_threshold_class1=4,
            balk_low_class1=0.3,
            balk_high_class1=0.7,
            noshow_threshold_class1=4,
            noshow_low_class1=0.3,
            noshow_high_class1=0.7,
        )
        prepared = prepare_h2_backgrounds(pd.DataFrame([first, second]))
        self.assertEqual(len(prepared), 1)
        self.assertEqual(int(prepared.loc[0, "source_scenario_count"]), 2)
        self.assertEqual(float(prepared.loc[0, "balk_low_class1"]), 0.0)
        self.assertEqual(float(prepared.loc[0, "noshow_high_class1"]), 0.0)


class H2ClassificationTests(unittest.TestCase):
    def _calibration(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "background_id": "H2B0001",
                    "source_scenario_ids": "A001",
                    "scenario_type": "anchor",
                    "target_loss_share": target,
                    "balk_probability": target,
                    "noshow_probability": target,
                    "estimated_balk_loss_share": target,
                    "estimated_noshow_loss_share": target,
                    "estimated_match_gap": 0.0,
                    "calibration_valid": True,
                    "rho": 3.1,
                    "class1_share": 0.5,
                    "slots_per_day": 32,
                    "horizon_class1": 14,
                    "horizon_class2": 14,
                    "cancel_class1": 0.1,
                    "cancel_class2": 0.1,
                }
                for target in (0.05, 0.10, 0.20)
            ]
        )

    def _raw(self, reversed_utilization: bool = False) -> pd.DataFrame:
        rows = []
        for seed in range(20):
            noise = (seed - 9.5) * 0.00001
            rows.append(
                {
                    "background_id": "H2B0001",
                    "target_loss_share": BASELINE_TARGET_SENTINEL,
                    "arm": "baseline",
                    "seed": seed,
                    "realized_focal_loss_share": 0.0,
                    "average_utilization": 0.80 + noise,
                    "class_1_percent_serviced": 0.40 + noise,
                }
            )
            for target in (0.05, 0.10, 0.20):
                rows.append(
                    {
                        "background_id": "H2B0001",
                        "target_loss_share": target,
                        "arm": "balk",
                        "seed": seed,
                        "realized_focal_loss_share": target + noise,
                        "average_utilization": (0.70 if reversed_utilization else 0.79) + noise,
                        "class_1_percent_serviced": 0.30 + noise,
                    }
                )
                rows.append(
                    {
                        "background_id": "H2B0001",
                        "target_loss_share": target,
                        "arm": "noshow",
                        "seed": seed,
                        "realized_focal_loss_share": target - noise,
                        "average_utilization": (0.79 if reversed_utilization else 0.70) + noise,
                        "class_1_percent_serviced": 0.29 + noise,
                    }
                )
        return pd.DataFrame(rows)

    def test_supported_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.csv"
            calibration_path = tmp_path / "calibration.csv"
            self._raw(False).to_csv(raw_path, index=False)
            self._calibration().to_csv(calibration_path, index=False)
            effects_path = classify_stage1(
                raw_path=raw_path,
                calibration_path=calibration_path,
                output_dir=tmp_path,
            )
            effects = pd.read_csv(effects_path)
            self.assertEqual(effects.loc[0, "classification"], "supported")

    def test_reversed_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.csv"
            calibration_path = tmp_path / "calibration.csv"
            self._raw(True).to_csv(raw_path, index=False)
            self._calibration().to_csv(calibration_path, index=False)
            effects_path = classify_stage1(
                raw_path=raw_path,
                calibration_path=calibration_path,
                output_dir=tmp_path,
            )
            effects = pd.read_csv(effects_path)
            self.assertEqual(effects.loc[0, "classification"], "reversed")

    def test_one_supported_target_is_not_enough_for_scenario_support(self) -> None:
        target_effects = pd.DataFrame(
            [
                {
                    "background_id": "H2B0001",
                    "source_scenario_ids": "A001",
                    "scenario_type": "anchor",
                    "rho": 3.1,
                    "class1_share": 0.5,
                    "slots_per_day": 32,
                    "horizon_class1": 14,
                    "horizon_class2": 14,
                    "cancel_class1": 0.1,
                    "cancel_class2": 0.1,
                    "loss_match_valid": True,
                    "failure_component": "",
                    "classification": label,
                }
                for label in ("supported", "inconclusive", "inactive")
            ]
        )
        scenario = _aggregate_scenario_effects(target_effects)
        self.assertEqual(scenario.loc[0, "classification"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
