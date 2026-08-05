from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "h1_postprocess_policy_outcomes.py"
SPEC = importlib.util.spec_from_file_location("h1_post", SCRIPT)
assert SPEC and SPEC.loader
h1_post = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h1_post)


def make_row(stage, seed, h, q, w, avg, weighted, c1, c2, arm="exact"):
    return {
        "stage": stage,
        "arm": arm,
        "background_id": f"BG1_{stage}_H={h}_Q={q}_w={w}",
        "source_background_id": "BG1",
        "variant": "strict",
        "seed": seed,
        "horizon_days": h,
        "Q": q,
        "window": w,
        "average_utilization": avg,
        "weighted_utilization": weighted,
        "class_1_percent_serviced": c1,
        "class_2_percent_serviced": c2,
        "class_1_slot_utilization": avg * 0.4,
        "class_2_slot_utilization": avg * 0.6,
        "access_advantage_class_1": c1 - c2,
        "class_1_balking_rate": 0.1,
        "class_2_balking_rate": 0.1,
        "class_1_no_show_rate": 0.1,
        "class_2_no_show_rate": 0.1,
        "class_1_no_offer_rate": 0.0,
        "class_2_no_offer_rate": 0.0,
        "class_1_mean_offered_booking_delay": 2.0,
        "class_2_mean_offered_booking_delay": 2.0,
    }


class H1PostprocessTest(unittest.TestCase):
    def setUp(self):
        rows = []
        for seed in (1001, 1002):
            rows.extend(
                [
                    make_row("baseline", seed, 14, 0, -1, 0.70, 0.70, 0.70, 0.70),
                    make_row("horizon_only", seed, 7, 0, -1, 0.80, 0.72, 0.68, 0.76),
                    make_row("horizon_only", seed, 21, 0, -1, 0.75, 0.74, 0.72, 0.76),
                    make_row("reservation_only", seed, 14, 5, 3, 0.82, 0.78, 0.82, 0.66, "coarse"),
                    # Duplicate policy cell under fine phase must not duplicate seeds.
                    make_row("reservation_only", seed, 14, 5, 3, 0.82, 0.78, 0.82, 0.66, "fine"),
                    make_row("both_flexible", seed, 7, 5, 3, 0.84, 0.80, 0.84, 0.64, "coarse"),
                    make_row("both_flexible", seed, 21, 5, 3, 0.79, 0.83, 0.88, 0.61, "coarse"),
                ]
            )
        self.shard = pd.DataFrame(rows)

    def test_objectives_can_choose_different_cells(self):
        avg = h1_post.select_policies(self.shard, "average_utilization")
        weighted = h1_post.select_policies(self.shard, "weighted_utilization")
        self.assertEqual(int(avg["both_flexible"]["horizon_days"].iloc[0]), 7)
        self.assertEqual(int(weighted["both_flexible"]["horizon_days"].iloc[0]), 21)
        self.assertEqual(avg["reservation_only"]["seed"].nunique(), 2)
        self.assertEqual(len(avg["reservation_only"]), 2)

    def test_tradeoff_category(self):
        selections = h1_post.select_policies(self.shard, "average_utilization")
        row = h1_post._comparison_row(
            background_id="BG1",
            variant="strict",
            objective="average_utilization",
            comparison="reservation_only_vs_baseline",
            first_policy="reservation_only",
            second_policy="baseline",
            first=selections["reservation_only"],
            second=selections["baseline"],
            draws=100,
            tolerance=0.005,
        )
        self.assertEqual(row["patient_group_tradeoff"], "priority_gains_general_loses")

    def test_end_to_end_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "rawroot" / "strict" / "raw"
            raw_dir.mkdir(parents=True)
            self.shard.to_csv(raw_dir / "BG1.csv", index=False)
            bank = pd.DataFrame(
                [{"background_id": "BG1", "rho": 1.2, "class1_share": 0.3}]
            )
            outputs = h1_post.process_variant(
                variant="strict",
                raw_root=root / "rawroot",
                bank=bank,
                output_root=root / "out",
                summary_root=None,
                draws=50,
                tolerance=0.005,
            )
            self.assertEqual(len(outputs["means"]), 8)
            self.assertEqual(len(outputs["deltas"]), 12)
            self.assertEqual(len(outputs["switch"]), 4)
            self.assertTrue((root / "out" / "strict" / "selected_policy_outcomes.csv").exists())
            self.assertTrue((root / "out" / "strict" / "pairwise_group_deltas.csv").exists())


if __name__ == "__main__":
    unittest.main()
