from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analysis.reservation_policy_selection import (
    build_selection_summaries,
    contiguous_q_ranges,
    near_tie_values,
    pair_with_fcfs,
    summarize_by_q,
)
from experiments.strict_reservation_policy_selection import (
    build_tasks,
    expected_cardinality,
)


class ReservationPolicySelectionTest(unittest.TestCase):
    def base_row(self, *, q: int, seed: int, value: float) -> dict[str, float]:
        row = {
            "scenario_id": "t09_09_b05_l25",
            "arrival_rate_class_1": 25,
            "arrival_rate_class_2": 25,
            "tau_1": 9,
            "tau_2": 9,
            "post_threshold_balking_rate": 0.5,
            "w1": 1,
            "w2": 1,
            "seed": seed,
            "Q": q,
            "A1": 100,
            "A2": 100,
            "Y1": 50,
            "Y2": 50,
            "offered_1": 60,
            "offered_2": 60,
            "sum_tau_offered_1": 120,
            "sum_tau_offered_2": 120,
            "Obj_util_raw": value * 2,
            "Obj_util_norm": value,
            "Obj_service_raw": value * 2,
            "Obj_service_norm": value,
            "T_wait_offered": 2 - value,
            "class_1_service_rate": value,
            "class_2_service_rate": value,
            "class_1_no_offer_rate": 0.2,
            "class_2_no_offer_rate": 0.2,
            "class_1_avg_offered_wait": 2 - value,
            "class_2_avg_offered_wait": 2 - value,
        }
        return row

    def test_pairing_includes_q0_zero_deltas(self) -> None:
        frame = pd.DataFrame(
            [
                self.base_row(q=q, seed=seed, value=value)
                for seed in (1, 2)
                for q, value in ((0, 0.4), (4, 0.5))
            ]
        )
        paired = pair_with_fcfs(frame)
        q0 = paired[paired["Q"] == 0]
        self.assertTrue(np.allclose(q0["delta_Obj_util_norm"], 0))
        self.assertTrue(
            np.allclose(
                paired.loc[paired["Q"] == 4, "delta_Obj_util_norm"],
                0.1,
            )
        )

    def test_pairing_rejects_missing_fcfs(self) -> None:
        frame = pd.DataFrame([self.base_row(q=4, seed=1, value=0.5)])
        with self.assertRaises(ValueError):
            pair_with_fcfs(frame)

    def test_near_ties_and_ranges(self) -> None:
        values = pd.Series({0: 1.0, 2: 0.995, 4: 0.98, 8: 0.999})
        exact, near, best = near_tie_values(values, direction="maximize")
        self.assertEqual(exact, [0])
        self.assertEqual(near, [0, 2, 8])
        self.assertEqual(best, 1.0)
        ranges = contiguous_q_ranges(
            near,
            tested_q_values=[0, 2, 4, 8],
        )
        self.assertEqual([row["q_values"] for row in ranges], ["0,2", "8"])

    def test_summary_shapes_and_includes_fcfs(self) -> None:
        frame = pd.DataFrame(
            [
                self.base_row(q=q, seed=seed, value=value)
                for seed in (1, 2, 3)
                for q, value in ((0, 0.5), (2, 0.504), (4, 0.503))
            ]
        )
        q_summary = summarize_by_q(pair_with_fcfs(frame))
        scenarios, best, ranges = build_selection_summaries(
            q_summary,
            tested_q_values=[0, 2, 4],
        )
        self.assertEqual(len(q_summary), 3)
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(len(best), 3)
        util = best[best["objective_name"] == "Obj_util_norm"].iloc[0]
        self.assertEqual(util["near_tie_q_values"], "0,2,4")
        self.assertFalse(ranges.empty)

    def test_experiment_grid_cardinality(self) -> None:
        standard = build_tasks("standard")
        smoke = build_tasks("smoke")
        self.assertEqual(expected_cardinality("standard"), 7200)
        self.assertEqual(len(standard), 7200)
        self.assertEqual(len({task.task_id for task in standard}), 7200)
        self.assertEqual(len(smoke), 24)


if __name__ == "__main__":
    unittest.main()
