from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments import h1_short_horizon_reservation as h1
from experiments import hypothesis_scenario_bank as bank_module


def _small_bank() -> pd.DataFrame:
    return bank_module.generate_background_bank(
        n_per_horizon=6, seed=7, horizons=(7, 21)
    )


class TaskGenerationTest(unittest.TestCase):
    def _assert_unique_keys(self, tasks: list[dict]) -> None:
        keys = [tuple(t["extra_cols"][c] for c in h1.KEY_COLUMNS) for t in tasks]
        self.assertEqual(len(keys), len(set(keys)), "task keys must be unique for resume to work")

    def test_screen_covers_whole_bank_with_on_off_pairs(self) -> None:
        bank = _small_bank()
        tasks = h1.screen_tasks(bank, smoke=False)
        self._assert_unique_keys(tasks)
        arms = {t["extra_cols"]["arm"] for t in tasks}
        self.assertEqual(arms, {"off", "on"})
        background_ids = {t["extra_cols"]["background_id"] for t in tasks}
        self.assertEqual(background_ids, set(bank["background_id"]))

    def test_screen_tasks_have_matching_schema_regardless_of_arm(self) -> None:
        # Regression: screen and grid extra_cols must share the same key
        # set so the resumable CSV writer never appends mismatched headers.
        bank = _small_bank()
        tasks = h1.screen_tasks(bank, smoke=True)
        keys = {frozenset(t["extra_cols"].keys()) for t in tasks}
        self.assertEqual(len(keys), 1)

    def test_grid_and_screen_extra_cols_share_the_same_schema(self) -> None:
        bank = _small_bank()
        deep = h1.select_deep_backgrounds(bank)
        screen = h1.screen_tasks(bank, smoke=True)
        grid = h1.grid_tasks(deep, smoke=True)
        screen_keys = set(screen[0]["extra_cols"].keys())
        grid_keys = set(grid[0]["extra_cols"].keys())
        self.assertEqual(screen_keys, grid_keys)

    def test_grid_includes_q_zero_and_positive_q(self) -> None:
        bank = _small_bank()
        deep = h1.select_deep_backgrounds(bank)
        tasks = h1.grid_tasks(deep, smoke=True)
        self._assert_unique_keys(tasks)
        q_values = {t["extra_cols"]["Q"] for t in tasks}
        self.assertIn(0, q_values)
        self.assertTrue(any(q > 0 for q in q_values))

    def test_window_grid_never_exceeds_noshow_threshold_1(self) -> None:
        for threshold in (4, 6, 10, 24):
            windows = h1.window_grid_for_threshold(threshold)
            self.assertTrue(all(w <= threshold for w in windows))

    def test_q_grid_caps_at_half_capacity(self) -> None:
        for capacity in (20, 30, 40, 50):
            q_values = h1.q_grid_for_capacity(capacity)
            self.assertTrue(all(q <= capacity // 2 for q in q_values))
            self.assertTrue(all(q % 5 == 0 for q in q_values))

    def test_smoke_reduces_seed_count(self) -> None:
        full = len(h1._seeds(smoke=False))
        smoke = len(h1._seeds(smoke=True))
        self.assertLess(smoke, full)


class SelectDeepBackgroundsTest(unittest.TestCase):
    def test_selection_includes_condition_satisfied_and_violated(self) -> None:
        bank = bank_module.generate_background_bank(n_per_horizon=30, seed=11)
        deep = h1.select_deep_backgrounds(bank)
        self.assertGreater(len(deep), 0)
        self.assertIn("deep_bucket", deep.columns)
        gap = deep["noshow_threshold_2"] - deep["noshow_threshold_1"]
        self.assertTrue((gap > 0).any())
        self.assertTrue((gap <= 0).any() or len(deep) == (gap > 0).sum())


class EndToEndSmokeTest(unittest.TestCase):
    def test_run_and_classify_screen_produces_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank_path = Path(tmp) / "bank.csv"
            _small_bank().to_csv(bank_path, index=False)
            output_dir = Path(tmp) / "out"
            raw_path = h1.run(
                stages=["screen"],
                bank_path=bank_path,
                output_dir=output_dir,
                workers=1,
                smoke=True,
                resume=False,
            )
            self.assertTrue(raw_path.exists())
            raw = pd.read_csv(raw_path)
            self.assertGreater(len(raw), 0)
            for column in ("average_utilization", "reserved_slot_fill_rate", "arm", "stage"):
                self.assertIn(column, raw.columns)

            h1.classify(raw_path=raw_path, bank_path=bank_path, output_dir=output_dir)
            screen_path = output_dir / "summary" / "screen_by_background.csv"
            self.assertTrue(screen_path.exists())
            screen = pd.read_csv(screen_path)
            self.assertIn("utilization_status", screen.columns)
            self.assertIn("condition_satisfied", screen.columns)
            self.assertTrue(
                screen["utilization_status"].isin(["supported", "inconclusive", "contradicted"]).all()
            )
            # mean_offered_delay has no directional claim: diagnostic delta
            # only, no status column.
            self.assertNotIn("mean_offered_delay_status", screen.columns)

    def test_run_and_classify_grid_produces_optimal_vs_naive_vs_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank_path = Path(tmp) / "bank.csv"
            _small_bank().to_csv(bank_path, index=False)
            output_dir = Path(tmp) / "out"
            raw_path = h1.run(
                stages=["grid"],
                bank_path=bank_path,
                output_dir=output_dir,
                workers=1,
                smoke=True,
                resume=False,
            )
            h1.classify(raw_path=raw_path, bank_path=bank_path, output_dir=output_dir)
            grid_path = output_dir / "summary" / "grid_policy_summary.csv"
            self.assertTrue(grid_path.exists())
            grid = pd.read_csv(grid_path)
            for column in (
                "optimal_utilization",
                "none_utilization",
                "naive_utilization",
                "optimal_minus_none",
            ):
                self.assertIn(column, grid.columns)

    def test_resume_skips_already_completed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank_path = Path(tmp) / "bank.csv"
            _small_bank().to_csv(bank_path, index=False)
            output_dir = Path(tmp) / "out"
            h1.run(
                stages=["screen"],
                bank_path=bank_path,
                output_dir=output_dir,
                workers=1,
                smoke=True,
                resume=False,
            )
            raw_path = output_dir / "raw" / "h1_raw.csv"
            first_len = len(pd.read_csv(raw_path))

            h1.run(
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
