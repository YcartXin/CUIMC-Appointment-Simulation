from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.strict_reservation_weight_local_sensitivity import (
    BALK_HIGH_GRID,
    CANCEL_PROB_GRID,
    NO_SHOW_HIGH_GRID,
    NO_SHOW_THRESHOLD_GRID,
    Q_VALUES,
    STANDARD_SEEDS,
    THRESHOLD_GRID,
    WEIGHTS,
    build_config,
    build_tasks,
    expected_cardinality,
    q_level_summary,
    score_rows,
    summarize_best_q,
)


class StrictReservationWeightLocalSensitivityTest(unittest.TestCase):
    def test_grid_cardinality_and_unique_task_ids(self) -> None:
        standard = build_tasks("standard")
        smoke = build_tasks("smoke")

        self.assertEqual(expected_cardinality("standard"), 148_500)
        self.assertEqual(len(standard), 148_500)
        self.assertEqual(len({task.task_id for task in standard}), 148_500)
        self.assertEqual(len(smoke), 2_376)

        behavior_cells = (
            len(BALK_HIGH_GRID) ** 2
            + len(THRESHOLD_GRID) ** 2
            + 2 * len(BALK_HIGH_GRID) * len(THRESHOLD_GRID)
            + len(CANCEL_PROB_GRID) ** 2
            + len(NO_SHOW_HIGH_GRID) ** 2
            + len(NO_SHOW_THRESHOLD_GRID) ** 2
            + 2 * len(NO_SHOW_HIGH_GRID) * len(NO_SHOW_THRESHOLD_GRID)
        )
        self.assertEqual(
            len(standard),
            behavior_cells * len(Q_VALUES) * len(STANDARD_SEEDS),
        )

    def test_configuration_uses_class_specific_balking_inputs(self) -> None:
        task = next(
            task
            for task in build_tasks("standard")
            if task.analysis_family == "class1_balk_surface"
            and task.tau_1 == 7
            and np.isclose(task.balk_high_class_1, 0.7)
            and task.Q == 12
        )
        config = build_config(task)

        self.assertEqual(config.slots_per_day, 32)
        self.assertEqual(config.horizon_days, 14)
        self.assertEqual(config.reserved_class_id, 1)
        self.assertEqual(config.reserved_slots_per_day, 12)
        self.assertEqual(config.classes[1].balk_prob.threshold, 7)
        self.assertAlmostEqual(config.classes[1].balk_prob.high, 0.7)
        self.assertEqual(config.classes[2].balk_prob.threshold, 9)
        self.assertAlmostEqual(config.classes[2].balk_prob.high, 0.5)
        self.assertAlmostEqual(config.classes[1].cancel_prob, 0.10)
        self.assertAlmostEqual(config.classes[2].cancel_prob, 0.10)
        self.assertEqual(config.classes[1].no_show_prob.threshold, 6)
        self.assertAlmostEqual(config.classes[1].no_show_prob.high, 0.30)

    def test_configuration_uses_class_specific_cancellation_inputs(self) -> None:
        task = next(
            task
            for task in build_tasks("standard")
            if task.analysis_family == "cancellation_probability_grid"
            and np.isclose(task.cancel_prob_class_1, 0.15)
            and np.isclose(task.cancel_prob_class_2, 0.20)
            and task.Q == 12
        )
        config = build_config(task)

        self.assertAlmostEqual(config.classes[1].cancel_prob, 0.15)
        self.assertAlmostEqual(config.classes[2].cancel_prob, 0.20)
        self.assertEqual(config.classes[1].balk_prob.threshold, 9)
        self.assertAlmostEqual(config.classes[1].balk_prob.high, 0.50)
        self.assertEqual(config.classes[1].no_show_prob.threshold, 6)
        self.assertAlmostEqual(config.classes[1].no_show_prob.high, 0.30)

    def test_configuration_uses_class_specific_no_show_inputs(self) -> None:
        task = next(
            task
            for task in build_tasks("standard")
            if task.analysis_family == "class2_no_show_surface"
            and task.no_show_threshold_2 == 8
            and np.isclose(task.no_show_high_class_2, 0.4)
            and task.Q == 12
        )
        config = build_config(task)

        self.assertEqual(config.classes[1].no_show_prob.threshold, 6)
        self.assertAlmostEqual(config.classes[1].no_show_prob.high, 0.30)
        self.assertEqual(config.classes[2].no_show_prob.threshold, 8)
        self.assertAlmostEqual(config.classes[2].no_show_prob.high, 0.40)
        self.assertAlmostEqual(config.classes[1].cancel_prob, 0.10)
        self.assertEqual(config.classes[1].balk_prob.threshold, 9)

    def test_q0_configuration_is_pooled_fcfs(self) -> None:
        task = next(task for task in build_tasks("smoke") if task.Q == 0)
        config = build_config(task)

        self.assertIsNone(config.reserved_class_id)
        self.assertEqual(config.reserved_slots_per_day, 0)

    def test_score_rows_expands_weights_and_pairs_to_fcfs(self) -> None:
        rows = []
        base = {
            "profile": "unit",
            "analysis_family": "balk_probability_grid",
            "scenario_id": "unit_scenario",
            "arrival_rate_class_1": 25,
            "arrival_rate_class_2": 25,
            "tau_1": 9,
            "tau_2": 9,
            "balk_high_class_1": 0.5,
            "balk_high_class_2": 0.5,
            "cancel_prob_class_1": 0.10,
            "cancel_prob_class_2": 0.10,
            "no_show_threshold_1": 6,
            "no_show_threshold_2": 6,
            "no_show_high_class_1": 0.30,
            "no_show_high_class_2": 0.30,
            "seed": 1,
            "S": 320,
            "A1": 100,
            "A2": 100,
            "offered_1": 90,
            "offered_2": 90,
            "sum_tau_offered_1": 180,
            "sum_tau_offered_2": 180,
            "no_offer_1": 10,
            "no_offer_2": 10,
        }
        for q, y1, y2 in ((0, 50, 50), (4, 55, 48)):
            rows.append({**base, "Q": q, "Y1": y1, "Y2": y2})

        scored = score_rows(pd.DataFrame(rows))

        self.assertEqual(len(scored), len(rows) * len(WEIGHTS))
        q0 = scored[scored["Q"] == 0]
        self.assertTrue(np.allclose(q0["delta_Obj_util_norm"], 0))
        self.assertFalse(scored["fcfs_Obj_util_norm"].isna().any())
        self.assertIn("delta_Obj_service_norm", scored.columns)
        self.assertIn("delta_T_wait_offered", scored.columns)

    def test_best_q_summary_keeps_exact_and_near_tie_values(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "analysis_family": "threshold_grid",
                    "scenario_id": "unit",
                    "arrival_rate_class_1": 25,
                    "arrival_rate_class_2": 25,
                    "tau_1": 9,
                    "tau_2": 9,
                    "balk_high_class_1": 0.5,
                    "balk_high_class_2": 0.5,
                    "cancel_prob_class_1": 0.10,
                    "cancel_prob_class_2": 0.10,
                    "no_show_threshold_1": 6,
                    "no_show_threshold_2": 6,
                    "no_show_high_class_1": 0.30,
                    "no_show_high_class_2": 0.30,
                    "w1": 1.0,
                    "w2": 1.0,
                    "Q": q,
                    "Obj_util_norm": value,
                    "Obj_service_norm": value,
                    "T_wait_offered": 2.0,
                    "class_1_service_rate": value,
                    "class_2_service_rate": value,
                    "class_1_no_offer_rate": 0.1,
                    "class_2_no_offer_rate": 0.1,
                    "delta_Obj_util_norm": value - 0.5,
                    "delta_Obj_service_norm": value - 0.5,
                    "delta_T_wait_offered": 0.0,
                    "delta_class_1_service_rate": value - 0.5,
                    "delta_class_2_service_rate": value - 0.5,
                    "delta_class_1_no_offer_rate": 0.0,
                    "delta_class_2_no_offer_rate": 0.0,
                }
                for q, value in ((0, 0.5), (2, 0.504), (4, 0.503))
                for _seed in (1, 2)
            ]
        )
        summary = q_level_summary(frame)
        best = summarize_best_q(summary, [0, 2, 4])
        util = best[best["objective_name"] == "Obj_util_norm"].iloc[0]

        self.assertEqual(util["exact_best_q_values"], "2")
        self.assertEqual(util["near_tie_q_values"], "0,2,4")
        self.assertEqual(util["near_tie_q_ranges"], "0-4 [0,2,4]")


if __name__ == "__main__":
    unittest.main()
