from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments import h1_short_horizon_reservation as h1
from experiments import hypothesis_scenario_bank as bank_module


def _small_bank() -> pd.DataFrame:
    return bank_module.generate_background_bank(
        n_per_horizon=3, seed=7, horizons=(6, 18)
    )


class CoarseToFineGridTest(unittest.TestCase):
    def test_q_coarse_grid_steps_and_includes_capacity(self) -> None:
        grid = h1.q_coarse_grid(50)
        self.assertEqual(grid, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        self.assertNotIn(0, grid)

    def test_q_coarse_grid_always_includes_capacity_even_off_step(self) -> None:
        grid = h1.q_coarse_grid(22)
        self.assertEqual(grid[-1], 22)

    def test_window_coarse_grid_steps_and_includes_horizon(self) -> None:
        grid = h1.window_coarse_grid(14)
        self.assertEqual(grid, [1, 3, 5, 7, 9, 11, 13, 14])
        self.assertEqual(grid[-1], 14)

    def test_q_fine_grid_covers_neighborhood_excluding_center(self) -> None:
        grid = h1.q_fine_grid(15, 50)
        self.assertEqual(grid, [10, 11, 12, 13, 14, 16, 17, 18, 19, 20])
        self.assertNotIn(15, grid)

    def test_q_fine_grid_clips_to_valid_range(self) -> None:
        grid = h1.q_fine_grid(2, 50)
        self.assertEqual(min(grid), 1)
        grid_hi = h1.q_fine_grid(48, 50)
        self.assertEqual(max(grid_hi), 50)

    def test_q_fine_grid_empty_when_coarse_winner_is_zero(self) -> None:
        self.assertEqual(h1.q_fine_grid(0, 50), [])

    def test_window_fine_grid_covers_neighborhood_excluding_center(self) -> None:
        grid = h1.window_fine_grid(7, 14)
        self.assertEqual(grid, [5, 6, 8, 9])
        self.assertNotIn(7, grid)

    def test_fine_radius_matches_coarse_step_for_full_coverage(self) -> None:
        # The fine radius on each dimension exactly matches that
        # dimension's coarse step, so coarse+fine together give full
        # step-1 coverage out to the adjacent coarse points on both sides.
        self.assertEqual(h1.Q_REFINE_RADIUS, h1.Q_COARSE_STEP)
        self.assertEqual(h1.WINDOW_REFINE_RADIUS, h1.WINDOW_COARSE_STEP)


class TaskGenerationTest(unittest.TestCase):
    def _assert_unique_keys(self, tasks: list[dict]) -> None:
        keys = [tuple(t["extra_cols"][c] for c in h1.KEY_COLUMNS) for t in tasks]
        self.assertEqual(len(keys), len(set(keys)), "task keys must be unique for resume to work")

    def test_baseline_is_one_cell_per_seed_at_native_horizon(self) -> None:
        bank = _small_bank()
        row = bank.iloc[0]
        tasks = h1.baseline_tasks(row, h1.VARIANT_STRICT, smoke=False)
        self.assertEqual(len(tasks), len(h1._seeds(False)))
        for t in tasks:
            self.assertEqual(t["extra_cols"]["Q"], 0)
            self.assertEqual(t["extra_cols"]["horizon_days"], int(row["horizon_days"]))
            self.assertEqual(t["config_kwargs"]["reserved_slots_per_day"], 0)

    def test_horizon_only_sweeps_all_horizons_with_q_zero(self) -> None:
        bank = _small_bank()
        row = bank.iloc[0]
        tasks = h1.horizon_only_tasks(row, h1.VARIANT_STRICT, smoke=False)
        self._assert_unique_keys(tasks)
        horizons = {t["extra_cols"]["horizon_days"] for t in tasks}
        self.assertEqual(horizons, set(h1.H1_HORIZON_VALUES))
        self.assertTrue(all(t["extra_cols"]["Q"] == 0 for t in tasks))

    def test_reservation_only_coarse_never_includes_q_zero(self) -> None:
        bank = _small_bank()
        row = bank.iloc[0]
        tasks = h1.reservation_only_coarse_tasks(row, h1.VARIANT_STRICT, smoke=False)
        self._assert_unique_keys(tasks)
        q_values = {t["extra_cols"]["Q"] for t in tasks}
        self.assertNotIn(0, q_values)
        horizons = {t["extra_cols"]["horizon_days"] for t in tasks}
        self.assertEqual(horizons, {int(row["horizon_days"])})

    def test_reservation_only_fine_refines_around_winner(self) -> None:
        bank = _small_bank()
        row = bank.iloc[0]
        tasks = h1.reservation_only_fine_tasks(
            row, h1.VARIANT_STRICT, smoke=False, best_q=10, best_window=5
        )
        self._assert_unique_keys(tasks)
        q_values = {t["extra_cols"]["Q"] for t in tasks if t["extra_cols"]["window"] == 5}
        self.assertNotIn(10, q_values)  # center excluded, already evaluated in coarse

    def test_both_flexible_coarse_sweeps_every_horizon(self) -> None:
        bank = _small_bank()
        row = bank.iloc[0]
        tasks = h1.both_flexible_coarse_tasks(row, h1.VARIANT_STRICT, smoke=False)
        self._assert_unique_keys(tasks)
        horizons = {t["extra_cols"]["horizon_days"] for t in tasks}
        self.assertEqual(horizons, set(h1.H1_HORIZON_VALUES))
        q_values = {t["extra_cols"]["Q"] for t in tasks}
        self.assertNotIn(0, q_values)

    def test_both_flexible_fine_only_covers_winning_horizons(self) -> None:
        bank = _small_bank()
        row = bank.iloc[0]
        winners = {6: (10, 3), 18: (20, 7)}
        tasks = h1.both_flexible_fine_tasks(row, h1.VARIANT_STRICT, smoke=False, winners=winners)
        horizons = {t["extra_cols"]["horizon_days"] for t in tasks}
        self.assertEqual(horizons, {6, 18})

    def test_release_variant_sets_release_flag_only_for_positive_q(self) -> None:
        bank = _small_bank()
        row = bank.iloc[0]
        tasks = h1.reservation_only_coarse_tasks(row, h1.VARIANT_RELEASE, smoke=False)
        for t in tasks:
            self.assertTrue(t["config_kwargs"]["release_unused_reservation_same_day"])
        baseline = h1.baseline_tasks(row, h1.VARIANT_RELEASE, smoke=False)
        for t in baseline:
            self.assertFalse(t["config_kwargs"]["release_unused_reservation_same_day"])

    def test_strict_variant_never_sets_release_flag(self) -> None:
        bank = _small_bank()
        row = bank.iloc[0]
        tasks = h1.reservation_only_coarse_tasks(row, h1.VARIANT_STRICT, smoke=False)
        for t in tasks:
            self.assertFalse(t["config_kwargs"]["release_unused_reservation_same_day"])

    def test_same_day_cancellation_always_enabled_regardless_of_variant(self) -> None:
        bank = _small_bank()
        row = bank.iloc[0]
        for variant in h1.VARIANTS:
            for tasks in (
                h1.baseline_tasks(row, variant, smoke=False),
                h1.horizon_only_tasks(row, variant, smoke=False),
                h1.reservation_only_coarse_tasks(row, variant, smoke=False),
            ):
                for t in tasks:
                    self.assertTrue(t["config_kwargs"]["same_day_cancellation_enabled"])

    def test_invalid_variant_rejected_by_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank_path = Path(tmp) / "bank.csv"
            _small_bank().to_csv(bank_path, index=False)
            with self.assertRaises(ValueError):
                h1.run(
                    variant="bogus",
                    bank_path=bank_path,
                    output_dir=Path(tmp) / "out",
                    workers=1,
                    smoke=True,
                    resume=False,
                )


class ShardedExecutionTest(unittest.TestCase):
    def test_run_sharded_tasks_routes_rows_by_source_background_id(self) -> None:
        bank = _small_bank()
        row = bank.iloc[0]
        tasks = h1.baseline_tasks(row, h1.VARIANT_STRICT, smoke=True)
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            h1.run_sharded_tasks(tasks, raw_dir=raw_dir, workers=1)
            shard = h1.shard_path(raw_dir, row["background_id"])
            self.assertTrue(shard.exists())
            written = pd.read_csv(shard)
            self.assertEqual(len(written), len(tasks))

    def test_filter_pending_skips_already_completed(self) -> None:
        bank = _small_bank()
        row = bank.iloc[0]
        tasks = h1.baseline_tasks(row, h1.VARIANT_STRICT, smoke=True)
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            h1.run_sharded_tasks(tasks, raw_dir=raw_dir, workers=1)
            pending = h1._filter_pending(tasks, raw_dir)
            self.assertEqual(pending, [])


class EndToEndSmokeTest(unittest.TestCase):
    def test_run_and_classify_produce_expected_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank_path = Path(tmp) / "bank.csv"
            _small_bank().to_csv(bank_path, index=False)
            output_dir = Path(tmp) / "out"

            h1.run(
                variant=h1.VARIANT_STRICT,
                bank_path=bank_path,
                output_dir=output_dir,
                workers=1,
                smoke=True,
                resume=False,
            )
            h1.classify(output_dir=output_dir, bank_path=bank_path, variant=h1.VARIANT_STRICT)

            optima_path = output_dir / h1.VARIANT_STRICT / "summary" / "condition_optima.csv"
            deltas_path = output_dir / h1.VARIANT_STRICT / "summary" / "condition_deltas.csv"
            self.assertTrue(optima_path.exists())
            self.assertTrue(deltas_path.exists())

            optima = pd.read_csv(optima_path)
            self.assertGreater(len(optima), 0)
            for stage in h1.STAGES:
                for suffix in ("horizon_days", "Q", "window", "average_utilization", "weighted_utilization"):
                    self.assertIn(f"{stage}_{suffix}", optima.columns)
            self.assertTrue((optima["baseline_Q"] == 0).all())

            deltas = pd.read_csv(deltas_path)
            self.assertGreater(len(deltas), 0)
            self.assertEqual(
                set(deltas["comparison"].unique()),
                {
                    "both_flexible_vs_baseline",
                    "both_flexible_vs_reservation_only",
                    "both_flexible_vs_horizon_only",
                },
            )
            for metric in ("average_utilization", "weighted_utilization"):
                self.assertIn(f"delta_{metric}", deltas.columns)
                self.assertIn(f"{metric}_status", deltas.columns)

            summary_md = output_dir / h1.VARIANT_STRICT / "summary" / "h1_summary.md"
            self.assertTrue(summary_md.exists())

    def test_resume_skips_already_completed_backgrounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank_path = Path(tmp) / "bank.csv"
            _small_bank().to_csv(bank_path, index=False)
            output_dir = Path(tmp) / "out"

            h1.run(
                variant=h1.VARIANT_RELEASE,
                bank_path=bank_path,
                output_dir=output_dir,
                workers=1,
                smoke=True,
                resume=False,
            )
            raw_dir = output_dir / h1.VARIANT_RELEASE / "raw"
            row_counts_before = {
                shard.name: len(pd.read_csv(shard)) for shard in raw_dir.glob("*.csv")
            }

            h1.run(
                variant=h1.VARIANT_RELEASE,
                bank_path=bank_path,
                output_dir=output_dir,
                workers=1,
                smoke=True,
                resume=True,
            )
            row_counts_after = {
                shard.name: len(pd.read_csv(shard)) for shard in raw_dir.glob("*.csv")
            }
            self.assertEqual(row_counts_before, row_counts_after)

    def test_strict_and_release_variants_write_separate_output_trees(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank_path = Path(tmp) / "bank.csv"
            _small_bank().to_csv(bank_path, index=False)
            output_dir = Path(tmp) / "out"

            h1.run(
                variant=h1.VARIANT_STRICT,
                bank_path=bank_path,
                output_dir=output_dir,
                workers=1,
                smoke=True,
                resume=False,
            )
            h1.run(
                variant=h1.VARIANT_RELEASE,
                bank_path=bank_path,
                output_dir=output_dir,
                workers=1,
                smoke=True,
                resume=False,
            )
            self.assertTrue((output_dir / h1.VARIANT_STRICT / "raw").exists())
            self.assertTrue((output_dir / h1.VARIANT_RELEASE / "raw").exists())


if __name__ == "__main__":
    unittest.main()
