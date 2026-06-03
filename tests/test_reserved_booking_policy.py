from __future__ import annotations

import unittest

from simulation.engine import ClinicAppointmentSimulation
from simulation.model import PatientClassParams, SimulationConfig, ThresholdRule


ZERO_RULE = ThresholdRule(threshold=0, low=0.0, high=0.0)


class ReservedBookingPolicyTest(unittest.TestCase):
    def make_config(self, *, release_reserved_slots: bool = False) -> SimulationConfig:
        return SimulationConfig(
            slots_per_day=4,
            horizon_days=1,
            burn_in_days=0,
            measure_days=1,
            cooldown_days=0,
            classes={
                1: PatientClassParams(
                    class_id=1,
                    lambda_per_day=0,
                    balk_prob=ZERO_RULE,
                    cancel_prob=0.0,
                    no_show_prob=ZERO_RULE,
                ),
                2: PatientClassParams(
                    class_id=2,
                    lambda_per_day=0,
                    balk_prob=ZERO_RULE,
                    cancel_prob=0.0,
                    no_show_prob=ZERO_RULE,
                ),
            },
            reserved_class_id=1,
            reserved_slots_per_day=2,
            release_reserved_slots=release_reserved_slots,
        )

    def test_strict_reservation_leaves_unused_class_1_slots_open(self) -> None:
        sim = ClinicAppointmentSimulation(self.make_config())

        sim.process_daily_arrivals([2, 2, 2, 1], track_patients=True)

        self.assertEqual(sim.class_metrics[1].booked, 1)
        self.assertEqual(sim.class_metrics[2].booked, 2)
        self.assertEqual(sim.class_metrics[2].no_offer, 1)
        self.assertEqual(len(sim.calendar[0]), 3)
        self.assertEqual(sum(b.reserved_slot for b in sim.calendar[0]), 1)

    def test_class_1_overflow_uses_general_slots_when_reserved_pool_is_full(self) -> None:
        sim = ClinicAppointmentSimulation(self.make_config())

        sim.process_daily_arrivals([1, 1, 1], track_patients=True)

        self.assertEqual(sim.class_metrics[1].booked, 3)
        self.assertEqual(sum(b.reserved_slot for b in sim.calendar[0]), 2)

    def test_released_reservation_lets_class_2_fill_unused_reserved_slots(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(release_reserved_slots=True)
        )

        sim.process_daily_arrivals([2, 2, 2, 1], track_patients=True)

        self.assertEqual(sim.class_metrics[1].booked, 1)
        self.assertEqual(sim.class_metrics[2].booked, 3)
        self.assertEqual(sim.class_metrics[2].no_offer, 0)
        self.assertEqual(len(sim.calendar[0]), 4)
        self.assertEqual(sum(b.reserved_slot for b in sim.calendar[0]), 2)


if __name__ == "__main__":
    unittest.main()
