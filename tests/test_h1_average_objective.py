from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments import h1_short_horizon_reservation as h1


class ObjectiveSelectionTest(unittest.TestCase):
    def test_best_qw_uses_requested_objective(self) -> None:
        cells = pd.DataFrame(
            [
                {
                    "Q": 5,
                    "window": 1,
                    "average_utilization": 0.70,
                    "weighted_utilization": 0.90,
                },
                {
                    "Q": 5,
                    "window": 1,
                    "average_utilization": 0.72,
                    "weighted_utilization": 0.88,
                },
                {
                    "Q": 10,
                    "window": 3,
                    "average_utilization": 0.92,
                    "weighted_utilization": 0.75,
                },
                {
                    "Q": 10,
                    "window": 3,
                    "average_utilization": 0.90,
                    "weighted_utilization": 0.77,
                },
            ]
        )

        self.assertEqual(h1._best_qw(cells, "weighted_utilization"), (5, 1))
        self.assertEqual(h1._best_qw(cells, "average_utilization"), (10, 3))

    def test_condition_optimum_uses_requested_objective(self) -> None:
        cells = pd.DataFrame(
            [
                {
                    "stage": h1.STAGE_HORIZON_ONLY,
                    "horizon_days": 7,
                    "Q": 0,
                    "window": -1,
                    "seed": 1000,
                    "average_utilization": 0.75,
                    "weighted_utilization": 0.90,
                },
                {
                    "stage": h1.STAGE_HORIZON_ONLY,
                    "horizon_days": 14,
                    "Q": 0,
                    "window": -1,
                    "seed": 1000,
                    "average_utilization": 0.93,
                    "weighted_utilization": 0.80,
                },
            ]
        )

        weighted = h1._condition_optimum(
            cells,
            h1.STAGE_HORIZON_ONLY,
            objective="weighted_utilization",
        )
        average = h1._condition_optimum(
            cells,
            h1.STAGE_HORIZON_ONLY,
            objective="average_utilization",
        )

        self.assertIsNotNone(weighted)
        self.assertIsNotNone(average)
        self.assertEqual(int(weighted["horizon_days"].iloc[0]), 7)
        self.assertEqual(int(average["horizon_days"].iloc[0]), 14)

    def test_parser_accepts_average_utilization_objective(self) -> None:
        args = h1.build_parser().parse_args(
            [
                "run",
                "--variant",
                "strict",
                "--objective",
                "average_utilization",
            ]
        )
        self.assertEqual(args.objective, "average_utilization")

    def test_invalid_objective_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            h1._validate_objective("not_a_metric")


class AverageObjectiveClassificationTest(unittest.TestCase):
    @staticmethod
    def _seed_rows(
        *,
        stage: str,
        horizon: int,
        q: int,
        window: int,
        average: float,
        weighted: float,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for offset, seed in enumerate((1000, 1001)):
            rows.append(
                {
                    "source_background_id": "BG_TEST",
                    "stage": stage,
                    "horizon_days": horizon,
                    "Q": q,
                    "window": window,
                    "seed": seed,
                    "average_utilization": average + 0.001 * offset,
                    "weighted_utilization": weighted + 0.001 * offset,
                }
            )
        return rows

    def test_classify_writes_average_summary_and_all_six_comparisons(self) -> None:
        rows: list[dict[str, object]] = []
        rows += self._seed_rows(
            stage=h1.STAGE_BASELINE,
            horizon=7,
            q=0,
            window=-1,
            average=0.70,
            weighted=0.85,
        )
        rows += self._seed_rows(
            stage=h1.STAGE_HORIZON_ONLY,
            horizon=7,
            q=0,
            window=-1,
            average=0.72,
            weighted=0.88,
        )
        rows += self._seed_rows(
            stage=h1.STAGE_HORIZON_ONLY,
            horizon=14,
            q=0,
            window=-1,
            average=0.82,
            weighted=0.80,
        )
        rows += self._seed_rows(
            stage=h1.STAGE_RESERVATION_ONLY,
            horizon=7,
            q=5,
            window=3,
            average=0.84,
            weighted=0.79,
        )
        rows += self._seed_rows(
            stage=h1.STAGE_BOTH_FLEXIBLE,
            horizon=14,
            q=5,
            window=3,
            average=0.90,
            weighted=0.78,
        )

        bank = pd.DataFrame(
            [
                {
                    "background_id": "BG_TEST",
                    "horizon_days": 7,
                    "rho": 1.2,
                    "class1_share": 0.4,
                    "slots_per_day": 20,
                    "noshow_threshold_1": 3,
                    "noshow_threshold_2": 6,
                }
            ]
        )

        expected_comparisons = {
            "horizon_only_vs_baseline",
            "reservation_only_vs_baseline",
            "both_flexible_vs_baseline",
            "both_flexible_vs_horizon_only",
            "both_flexible_vs_reservation_only",
            "reservation_only_vs_horizon_only",
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank_path = root / "bank.csv"
            bank.to_csv(bank_path, index=False)

            raw_dir = root / h1.VARIANT_STRICT / "raw"
            raw_dir.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(raw_dir / "BG_TEST.csv", index=False)

            h1.classify(
                output_dir=root,
                bank_path=bank_path,
                variant=h1.VARIANT_STRICT,
                objective="average_utilization",
            )

            summary_dir = root / h1.VARIANT_STRICT / "summary_average_utilization"
            optima = pd.read_csv(summary_dir / "condition_optima.csv")
            deltas = pd.read_csv(summary_dir / "condition_deltas.csv")

            self.assertEqual(
                optima.loc[0, "optimization_objective"],
                "average_utilization",
            )
            self.assertEqual(int(optima.loc[0, "horizon_only_horizon_days"]), 14)
            self.assertEqual(set(deltas["comparison"]), expected_comparisons)
            self.assertTrue(
                (deltas["optimization_objective"] == "average_utilization").all()
            )
            self.assertTrue((summary_dir / "h1_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
