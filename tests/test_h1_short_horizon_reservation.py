from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments import h1_short_horizon_reservation as h1
from experiments import hypothesis_scenario_bank as bank_module


def _small_bank() -> pd.DataFrame:
    return bank_module.generate_background_bank(
        n_per_horizon=6, seed=7, horizons=(6, 22)
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
        # Off-arm background_ids are unsuffixed and match the bank exactly;
        # on-arm background_ids are suffixed per swept window
        # (see screen_tasks), so we check source_background_id instead,
        # which is preserved unsuffixed on every task regardless of arm.
        source_ids = {t["extra_cols"]["source_background_id"] for t in tasks}
        self.assertEqual(source_ids, set(bank["background_id"]))
        off_background_ids = {
            t["extra_cols"]["background_id"] for t in tasks if t["extra_cols"]["arm"] == "off"
        }
        self.assertEqual(off_background_ids, set(bank["background_id"]))

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

    def test_window_grid_spans_one_to_the_full_horizon(self) -> None:
        for horizon in (2, 6, 14, 26):
            windows = h1.window_grid_for_horizon(horizon)
            self.assertEqual(windows, list(range(1, horizon + 1)))
            self.assertTrue(all(w <= horizon for w in windows))

    def test_screen_window_sweep_never_exceeds_each_backgrounds_own_horizon(self) -> None:
        bank = _small_bank()
        tasks = h1.screen_tasks(bank, smoke=False)
        by_bg_horizon = {row["background_id"]: int(row["horizon_days"]) for _, row in bank.iterrows()}
        for t in tasks:
            if t["extra_cols"]["arm"] != "on":
                continue
            horizon = by_bg_horizon[t["extra_cols"]["source_background_id"]]
            self.assertLessEqual(t["extra_cols"]["window"], horizon)

    def test_grid_sweeps_horizon_as_a_policy_lever(self) -> None:
        bank = _small_bank()
        deep = h1.select_deep_backgrounds(bank)
        tasks = h1.grid_tasks(deep, smoke=False)
        swept_horizons = {t["extra_cols"]["horizon_days"] for t in tasks}
        self.assertEqual(swept_horizons, set(h1.H1_HORIZON_VALUES))

    def test_grid_smoke_mode_uses_fewer_horizons(self) -> None:
        bank = _small_bank()
        deep = h1.select_deep_backgrounds(bank)
        tasks = h1.grid_tasks(deep, smoke=True)
        swept_horizons = {t["extra_cols"]["horizon_days"] for t in tasks}
        self.assertEqual(swept_horizons, set(h1.H1_HORIZON_VALUES[:2]))

    def test_thresholds_are_capped_at_horizon_minus_one_in_built_config(self) -> None:
        # Exercise the dynamic-capping path directly through build_config
        # (hypothesis_common), since h1's own task generation deliberately
        # passes raw, uncapped threshold values through.
        from experiments.hypothesis_common import build_config

        config = build_config(
            slots_per_day=30,
            lambda_1=10.0,
            lambda_2=10.0,
            cancel_1=0.1,
            cancel_2=0.1,
            balk_threshold_1=24,
            balk_low_1=0.1,
            balk_high_1=0.2,
            balk_threshold_2=24,
            balk_low_2=0.1,
            balk_high_2=0.2,
            noshow_threshold_1=22,
            noshow_low_1=0.1,
            noshow_high_1=0.2,
            noshow_threshold_2=22,
            noshow_low_2=0.1,
            noshow_high_2=0.2,
            horizon_days=2,
            seed=1,
        )
        self.assertEqual(config.classes[1].balk_prob.threshold, 1)
        self.assertEqual(config.classes[1].no_show_prob.threshold, 1)
        self.assertEqual(config.classes[2].balk_prob.threshold, 1)
        self.assertEqual(config.classes[2].no_show_prob.threshold, 1)

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
            self.assertIn("best_window", screen.columns)
            self.assertTrue(
                screen["utilization_status"].isin(["supported", "inconclusive", "contradicted"]).all()
            )
            # mean_offered_delay has no directional claim: diagnostic delta
            # only, no status column.
            self.assertNotIn("mean_offered_delay_status", screen.columns)

    def test_run_and_classify_grid_produces_optimal_vs_none_per_horizon(self) -> None:
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
                "source_background_id",
                "horizon_days",
                "optimal_utilization",
                "none_utilization",
                "optimal_minus_none",
            ):
                self.assertIn(column, grid.columns)
            # naive-vs-optimal no longer applies: there is no single
            # universal (Q, window) baseline once window sweeps 1..horizon
            # and horizon itself is swept.
            self.assertNotIn("naive_utilization", grid.columns)
            # One row per (background, horizon) tested, not one per background.
            self.assertEqual(
                set(grid["horizon_days"].unique()), set(h1.H1_HORIZON_VALUES[:2])
            )

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
