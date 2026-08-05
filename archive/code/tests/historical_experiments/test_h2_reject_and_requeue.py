from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments import h2_reject_and_requeue as h2
from experiments import hypothesis_scenario_bank as bank_module


def _small_bank() -> pd.DataFrame:
    return bank_module.generate_background_bank(
        n_per_horizon=6, seed=7, horizons=(14, 28)
    )


class TaskGenerationTest(unittest.TestCase):
    def _assert_unique_keys(self, tasks: list[dict]) -> None:
        keys = [tuple(t["extra_cols"][c] for c in h2.KEY_COLUMNS) for t in tasks]
        self.assertEqual(len(keys), len(set(keys)), "task keys must be unique for resume to work")

    def test_screen_covers_whole_bank_with_on_off_pairs(self) -> None:
        bank = _small_bank()
        tasks = h2.screen_tasks(bank, smoke=False)
        self._assert_unique_keys(tasks)
        arms = {t["extra_cols"]["arm"] for t in tasks}
        self.assertEqual(arms, {"off", "on"})
        background_ids = {t["extra_cols"]["background_id"] for t in tasks}
        self.assertEqual(background_ids, set(bank["background_id"]))

    def test_screen_off_arm_has_no_standby(self) -> None:
        bank = _small_bank()
        tasks = h2.screen_tasks(bank, smoke=True)
        off_tasks = [t for t in tasks if t["extra_cols"]["arm"] == "off"]
        for t in off_tasks:
            self.assertEqual(t["config_kwargs"]["standby_prob_1"], 0.0)
            self.assertEqual(t["config_kwargs"]["standby_prob_2"], 0.0)

    def test_screen_and_dose_extra_cols_share_the_same_schema(self) -> None:
        bank = _small_bank()
        deep = h2.select_deep_backgrounds(bank)
        screen = h2.screen_tasks(bank, smoke=True)
        dose = h2.dose_tasks(deep, smoke=True)
        screen_keys = set(screen[0]["extra_cols"].keys())
        dose_keys = set(dose[0]["extra_cols"].keys())
        self.assertEqual(screen_keys, dose_keys)

    def test_dose_includes_zero_dose_control(self) -> None:
        bank = _small_bank()
        deep = h2.select_deep_backgrounds(bank)
        tasks = h2.dose_tasks(deep, smoke=True)
        self._assert_unique_keys(tasks)
        probs = {t["extra_cols"]["standby_prob"] for t in tasks}
        self.assertIn(0.0, probs)
        self.assertTrue(any(p > 0.0 for p in probs))

    def test_standby_kwargs_leave_max_standby_days_uncapped(self) -> None:
        kwargs = h2._standby_kwargs(0.3)
        self.assertIsNone(kwargs["max_standby_days_1"])
        self.assertIsNone(kwargs["max_standby_days_2"])
        self.assertEqual(kwargs["standby_eligible_after_days_1"], h2.STANDBY_ELIGIBLE_AFTER_DAYS)

    def test_smoke_reduces_seed_count(self) -> None:
        full = len(h2._seeds(smoke=False))
        smoke = len(h2._seeds(smoke=True))
        self.assertLess(smoke, full)


class SelectDeepBackgroundsTest(unittest.TestCase):
    def test_selection_includes_condition_satisfied_and_all_violation_modes(self) -> None:
        bank = bank_module.generate_background_bank(n_per_horizon=30, seed=11)
        deep = h2.select_deep_backgrounds(bank)
        self.assertGreater(len(deep), 0)
        self.assertIn("deep_bucket", deep.columns)
        long_horizon = deep["horizon_days"] >= 21
        high_demand = deep["rho"] >= 2.0
        self.assertTrue((long_horizon & high_demand).any())


class EndToEndSmokeTest(unittest.TestCase):
    def test_run_and_classify_screen_produces_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank_path = Path(tmp) / "bank.csv"
            _small_bank().to_csv(bank_path, index=False)
            output_dir = Path(tmp) / "out"
            raw_path = h2.run(
                stages=["screen"],
                bank_path=bank_path,
                output_dir=output_dir,
                workers=1,
                smoke=True,
                resume=False,
            )
            raw = pd.read_csv(raw_path)
            self.assertGreater(len(raw), 0)
            for column in ("average_utilization", "class_1_standby_joined", "class_1_no_show_rate"):
                self.assertIn(column, raw.columns)

            h2.classify(raw_path=raw_path, bank_path=bank_path, output_dir=output_dir)
            screen_path = output_dir / "summary" / "screen_by_background.csv"
            self.assertTrue(screen_path.exists())
            screen = pd.read_csv(screen_path)
            self.assertIn("utilization_status", screen.columns)
            self.assertIn("class_1_no_show_rate_status", screen.columns)
            self.assertIn("condition_satisfied", screen.columns)
            # mean_accepted_delay is a diagnostic in H2 (composition effect
            # confound), not a classified verdict.
            self.assertNotIn("mean_accepted_delay_status", screen.columns)
            self.assertIn("class_1_standby_recalled_on_arm", screen.columns)

    def test_dose_response_pairs_every_cell_with_paired_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank_path = Path(tmp) / "bank.csv"
            _small_bank().to_csv(bank_path, index=False)
            output_dir = Path(tmp) / "out"
            raw_path = h2.run(
                stages=["dose"],
                bank_path=bank_path,
                output_dir=output_dir,
                workers=1,
                smoke=True,
                resume=False,
            )
            h2.classify(raw_path=raw_path, bank_path=bank_path, output_dir=output_dir)
            dose_path = output_dir / "summary" / "dose_response.csv"
            self.assertTrue(dose_path.exists())
            dose = pd.read_csv(dose_path)
            self.assertGreater(len(dose), 0)
            self.assertIn("standby_prob", dose.columns)

    def test_resume_skips_already_completed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank_path = Path(tmp) / "bank.csv"
            _small_bank().to_csv(bank_path, index=False)
            output_dir = Path(tmp) / "out"
            h2.run(
                stages=["screen"],
                bank_path=bank_path,
                output_dir=output_dir,
                workers=1,
                smoke=True,
                resume=False,
            )
            raw_path = output_dir / "raw" / "h2_raw.csv"
            first_len = len(pd.read_csv(raw_path))

            h2.run(
                stages=["screen"],
                bank_path=bank_path,
                output_dir=output_dir,
                workers=1,
                smoke=True,
                resume=True,
            )
            second_len = len(pd.read_csv(raw_path))
            self.assertEqual(first_len, second_len)


if __name__ == "__main__":
    unittest.main()
