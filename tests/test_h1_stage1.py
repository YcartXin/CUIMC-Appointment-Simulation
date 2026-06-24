from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.robustness.h1_stage1 import (
    classify_stage1,
    prepare_h1_backgrounds,
)


class H1ScenarioPreparationTests(unittest.TestCase):
    def test_deduplicates_class1_cancel_only(self) -> None:
        base = {
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
        second = dict(base, scenario_id="A002", cancel_class1=0.5)
        prepared = prepare_h1_backgrounds(pd.DataFrame([base, second]))
        self.assertEqual(len(prepared), 1)
        self.assertEqual(int(prepared.loc[0, "source_scenario_count"]), 2)


class H1ClassificationTests(unittest.TestCase):
    def _raw_rows(self, rho: float, delay_delta: float) -> pd.DataFrame:
        rows = []
        for seed in range(20):
            noise = (seed - 9.5) * 0.00001
            for cancel in [0.0, 0.1, 0.3, 0.5]:
                fraction = cancel / 0.5
                rows.append(
                    {
                        "background_id": "H1B0001",
                        "source_scenario_ids": "A001",
                        "scenario_type": "anchor",
                        "rho": rho,
                        "class1_share": 0.5,
                        "slots_per_day": 32,
                        "horizon_class1": 14,
                        "horizon_class2": 14,
                        "cancel_class2_background": 0.1,
                        "cancel_class1_focal": cancel,
                        "seed": seed,
                        "class_1_percent_serviced": 0.35 - 0.10 * fraction + noise,
                        "class_2_percent_serviced": 0.20 + 0.08 * fraction,
                        "mean_offered_booking_delay": 10.0 + delay_delta * fraction,
                        "average_utilization": 0.80 + 0.04 * fraction,
                    }
                )
        return pd.DataFrame(rows)

    def test_high_demand_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.csv"
            self._raw_rows(rho=3.1, delay_delta=-1.0).to_csv(raw_path, index=False)
            effects_path = classify_stage1(raw_path=raw_path, output_dir=tmp_path)
            effects = pd.read_csv(effects_path)
            self.assertEqual(effects.loc[0, "classification"], "supported")

    def test_low_demand_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.csv"
            self._raw_rows(rho=0.8, delay_delta=0.0).to_csv(raw_path, index=False)
            effects_path = classify_stage1(raw_path=raw_path, output_dir=tmp_path)
            effects = pd.read_csv(effects_path)
            self.assertEqual(effects.loc[0, "classification"], "inactive")


if __name__ == "__main__":
    unittest.main()
