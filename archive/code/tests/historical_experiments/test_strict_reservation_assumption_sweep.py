from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

from experiments.strict_reservation_assumption_sweep import (
    HARD_CHECK_NAMES,
    SHARD_SCHEMA_VERSION,
    SLOTS_PER_DAY,
    atomic_write_csv_gz,
    build_config,
    build_tasks,
    capacity_violations,
    check_accounting,
    expected_cardinality,
    result_fingerprint,
    resume_pending_tasks,
    run_task,
    shard_path,
    strict_reservation_oracle,
)
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import Booking


class StrictReservationAssumptionSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.smoke_tasks = build_tasks("smoke")

    def task_with_q(self, q: int):
        return next(
            task
            for task in self.smoke_tasks
            if task.q == q
            and task.regime_id == self.smoke_tasks[0].regime_id
            and task.total_demand == 24
            and task.class_1_share == 0.25
            and task.seed == 61001
        )

    def test_standard_grid_cardinality_and_ids(self) -> None:
        tasks = build_tasks("standard")
        self.assertEqual(expected_cardinality("standard"), 28_800)
        self.assertEqual(len(tasks), 28_800)
        self.assertEqual(len({task.task_id for task in tasks}), 28_800)

    def test_configuration_matches_task(self) -> None:
        task = self.task_with_q(4)
        config = build_config(task)
        self.assertEqual(config.slots_per_day, 32)
        self.assertEqual(config.horizon_days, 14)
        self.assertEqual(config.cooldown_days, 14)
        self.assertEqual(config.reserved_class_id, 1)
        self.assertEqual(config.reserved_slots_per_day, 4)
        self.assertAlmostEqual(config.classes[1].lambda_per_day, 6.0)
        self.assertAlmostEqual(config.classes[2].lambda_per_day, 18.0)
        self.assertEqual(config.classes[1].balk_prob.threshold, 9)
        self.assertEqual(config.classes[1].no_show_prob.threshold, 6)

    def test_oracle_and_ownership_checks(self) -> None:
        config = build_config(self.task_with_q(4))
        calendar = [[] for _ in range(config.horizon_days)]
        calendar[0] = [
            Booking(2, 0, False, False)
            for _ in range(SLOTS_PER_DAY - config.reserved_slots_per_day)
        ]
        self.assertEqual(strict_reservation_oracle(calendar, config, 1), (0, True))
        self.assertEqual(strict_reservation_oracle(calendar, config, 2), (1, False))

        calendar[0].append(Booking(2, 0, False, True))
        violations = capacity_violations(calendar, config)
        self.assertEqual(violations["reserved_ownership"], 1)

    def test_accounting_and_cooldown_resolution(self) -> None:
        task = self.task_with_q(4)
        results = ClinicAppointmentSimulation(build_config(task)).run()
        checks = check_accounting(results)
        self.assertTrue(checks["class_accounting"])
        self.assertTrue(checks["total_accounting"])
        self.assertTrue(checks["delay_accounting"])
        self.assertTrue(checks["slot_utilization_accounting"])
        self.assertTrue(checks["zero_unresolved_with_cooldown"])

    def test_q0_and_qs_hard_checks(self) -> None:
        q0_row = run_task(self.task_with_q(0))
        self.assertTrue(q0_row["check_q0_exact_fcfs_equivalence"])

        qs_row = run_task(self.task_with_q(SLOTS_PER_DAY))
        self.assertTrue(qs_row["check_q_s_class2_exclusion"])
        self.assertEqual(qs_row["class_2_offered"], 0)
        self.assertEqual(qs_row["class_2_no_offer"], qs_row["class_2_arrivals"])
        for name in HARD_CHECK_NAMES:
            self.assertTrue(qs_row[f"check_{name}"], name)

    def test_deterministic_replay(self) -> None:
        task = self.task_with_q(4)
        first = ClinicAppointmentSimulation(build_config(task)).run()
        second = ClinicAppointmentSimulation(build_config(task)).run()
        self.assertEqual(result_fingerprint(first), result_fingerprint(second))

    def test_resume_accepts_valid_shard_and_rejects_corruption(self) -> None:
        tasks = [self.task_with_q(0), self.task_with_q(4)]
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            first_path = shard_path(output_dir, tasks[0])
            atomic_write_csv_gz(
                first_path,
                {
                    "shard_schema_version": SHARD_SCHEMA_VERSION,
                    "task_id": tasks[0].task_id,
                },
            )
            corrupt_path = shard_path(output_dir, tasks[1])
            corrupt_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(corrupt_path, "wt", encoding="utf-8") as handle:
                handle.write("not,a,valid,task,shard\n")

            pending, completed = resume_pending_tasks(output_dir, tasks)
            self.assertEqual(completed, {tasks[0].task_id})
            self.assertEqual([task.task_id for task in pending], [tasks[1].task_id])


if __name__ == "__main__":
    unittest.main()
