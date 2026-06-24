from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.robustness.scenario_design import generate_scenario_bank
from experiments.robustness.scenario_space import (
    N_ANCHORS,
    N_ASYMMETRIC_STRESS,
    N_SOBOL_SYMMETRIC,
)


class RobustnessScenarioDesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(cls.temp_dir.name)
        (
            cls.symmetric_df,
            cls.asymmetric_df,
            cls.all_df,
            cls.validation_df,
        ) = generate_scenario_bank(output_dir)
        cls.output_dir = output_dir

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_expected_counts(self) -> None:
        self.assertEqual(len(self.symmetric_df), N_ANCHORS + N_SOBOL_SYMMETRIC)
        self.assertEqual(len(self.asymmetric_df), N_ASYMMETRIC_STRESS)
        self.assertEqual(
            len(self.all_df),
            N_ANCHORS + N_SOBOL_SYMMETRIC + N_ASYMMETRIC_STRESS,
        )

    def test_all_validation_checks_pass(self) -> None:
        failures = self.validation_df[self.validation_df["status"] != "PASS"]
        self.assertTrue(failures.empty, failures.to_dict(orient="records"))

    def test_required_outputs_exist(self) -> None:
        required = {
            "symmetric_scenarios.csv",
            "asymmetric_scenarios.csv",
            "all_stage1_scenarios.csv",
            "stage1_seeds.csv",
            "stage2_seeds.csv",
            "scenario_validation.csv",
            "scenario_generation_summary.md",
        }
        observed = {path.name for path in self.output_dir.iterdir()}
        self.assertTrue(required.issubset(observed))

    def test_threshold_and_probability_constraints(self) -> None:
        for _, row in self.all_df.iterrows():
            for class_id in (1, 2):
                suffix = f"class{class_id}"
                self.assertGreaterEqual(
                    row[f"balk_high_{suffix}"], row[f"balk_low_{suffix}"]
                )
                self.assertGreaterEqual(
                    row[f"noshow_high_{suffix}"], row[f"noshow_low_{suffix}"]
                )
                self.assertLess(
                    row[f"balk_threshold_{suffix}"],
                    row[f"horizon_{suffix}"] - 1,
                )
                self.assertLess(
                    row[f"noshow_threshold_{suffix}"],
                    row[f"horizon_{suffix}"] - 1,
                )

    def test_stress_scenarios_are_actually_asymmetric(self) -> None:
        behavior_prefixes = (
            "horizon",
            "cancel",
            "balk_threshold",
            "balk_low",
            "balk_high",
            "noshow_threshold",
            "noshow_low",
            "noshow_high",
        )
        for _, row in self.asymmetric_df.iterrows():
            differences = [
                row[f"{prefix}_class1"] != row[f"{prefix}_class2"]
                for prefix in behavior_prefixes
            ]
            self.assertTrue(any(differences), row["scenario_id"])


if __name__ == "__main__":
    unittest.main()
