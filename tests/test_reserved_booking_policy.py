from __future__ import annotations

import unittest

from analysis.metrics import outcome_rates_from_result, outcome_totals
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import Booking, PatientClassParams, SimulationConfig, ThresholdRule


ZERO_RULE = ThresholdRule(threshold=0, low=0.0, high=0.0)


class ReservedBookingPolicyTest(unittest.TestCase):
    def make_config(
        self,
        *,
        slots_per_day: int = 4,
        horizon_days: int = 1,
        measure_days: int = 1,
        cooldown_days: int = 0,
        reserved_slots_per_day: int = 2,
        release_reserved_slots: bool = False,
        lambda_per_day: float = 0.0,
        balk_prob=ZERO_RULE,
        cancel_prob: float = 0.0,
        no_show_prob=ZERO_RULE,
        seed: int | None = None,
    ) -> SimulationConfig:
        return SimulationConfig(
            slots_per_day=slots_per_day,
            horizon_days=horizon_days,
            burn_in_days=0,
            measure_days=measure_days,
            cooldown_days=cooldown_days,
            classes={
                1: PatientClassParams(
                    class_id=1,
                    lambda_per_day=lambda_per_day,
                    balk_prob=balk_prob,
                    cancel_prob=cancel_prob,
                    no_show_prob=no_show_prob,
                ),
                2: PatientClassParams(
                    class_id=2,
                    lambda_per_day=lambda_per_day,
                    balk_prob=balk_prob,
                    cancel_prob=cancel_prob,
                    no_show_prob=no_show_prob,
                ),
            },
            seed=seed,
            reserved_class_id=1 if reserved_slots_per_day > 0 else None,
            reserved_slots_per_day=reserved_slots_per_day,
            release_reserved_slots=release_reserved_slots,
        )

    def assert_calendar_capacity(self, sim: ClinicAppointmentSimulation) -> None:
        general_slots = (
            sim.config.slots_per_day - sim.config.reserved_slots_per_day
        )

        for day_bookings in sim.calendar:
            reserved_bookings = sum(booking.reserved_slot for booking in day_bookings)
            general_bookings = len(day_bookings) - reserved_bookings

            self.assertLessEqual(len(day_bookings), sim.config.slots_per_day)
            self.assertLessEqual(
                reserved_bookings, sim.config.reserved_slots_per_day
            )
            self.assertLessEqual(general_bookings, general_slots)

    def assert_accounting_invariants(self, result) -> None:
        totals = outcome_totals(result)
        resolved_booked = (
            totals["served"] + totals["canceled"] + totals["no_show"]
        )
        unresolved_booked = totals["booked"] - resolved_booked

        self.assertGreaterEqual(unresolved_booked, 0)
        self.assertEqual(totals["unresolved_booked"], unresolved_booked)
        self.assertEqual(
            totals["arrivals"],
            totals["served"]
            + totals["balked"]
            + totals["no_offer"]
            + totals["canceled"]
            + totals["no_show"]
            + unresolved_booked,
        )
        self.assertEqual(
            totals["booked"],
            totals["served"]
            + totals["canceled"]
            + totals["no_show"]
            + unresolved_booked,
        )
        self.assertEqual(totals["offered"], totals["booked"] + totals["balked"])
        self.assertEqual(totals["arrivals"], totals["offered"] + totals["no_offer"])

        rates = outcome_rates_from_result(result)
        if totals["arrivals"] > 0:
            lost_share = (
                rates["balked_rate"]
                + rates["no_offer_rate"]
                + rates["canceled_rate"]
                + rates["no_show_rate"]
                + rates["unresolved_booked_rate"]
            )
            self.assertAlmostEqual(lost_share, 1.0 - rates["served_rate"])

    def test_strict_reservation_leaves_unused_class_1_slots_open(self) -> None:
        sim = ClinicAppointmentSimulation(self.make_config())

        sim.process_daily_arrivals([2, 2, 2, 1], track_patients=True)

        self.assertEqual(sim.class_metrics[1].booked, 1)
        self.assertEqual(sim.class_metrics[2].booked, 2)
        self.assertEqual(sim.class_metrics[2].no_offer, 1)
        self.assertEqual(len(sim.calendar[0]), 3)
        self.assertEqual(sum(b.reserved_slot for b in sim.calendar[0]), 1)
        self.assert_calendar_capacity(sim)

    def test_strict_reservation_keeps_class_2_out_of_reserved_slots(self) -> None:
        sim = ClinicAppointmentSimulation(self.make_config())

        sim.process_daily_arrivals([2, 2, 2, 2], track_patients=True)

        self.assertEqual(sim.class_metrics[2].booked, 2)
        self.assertEqual(sim.class_metrics[2].no_offer, 2)
        self.assertEqual(sum(b.reserved_slot for b in sim.calendar[0]), 0)
        self.assert_calendar_capacity(sim)

    def test_class_1_uses_reserved_before_general_on_same_day(self) -> None:
        sim = ClinicAppointmentSimulation(self.make_config())

        sim.process_daily_arrivals([1], track_patients=True)

        self.assertEqual(sim.class_metrics[1].booked, 1)
        self.assertTrue(sim.calendar[0][0].reserved_slot)
        self.assert_calendar_capacity(sim)

    def test_class_1_overflow_uses_general_slots_when_reserved_pool_is_full(self) -> None:
        sim = ClinicAppointmentSimulation(self.make_config())

        sim.process_daily_arrivals([1, 1, 1], track_patients=True)

        self.assertEqual(sim.class_metrics[1].booked, 3)
        self.assertEqual(sum(b.reserved_slot for b in sim.calendar[0]), 2)
        self.assertEqual(sum(not b.reserved_slot for b in sim.calendar[0]), 1)
        self.assert_calendar_capacity(sim)

    def test_class_1_first_backfill_reservation_lets_class_2_fill_unused_reserved_slots(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(release_reserved_slots=True)
        )

        sim.process_daily_arrivals([2, 2, 2, 1], track_patients=True)

        self.assertEqual(sim.class_metrics[1].booked, 1)
        self.assertEqual(sim.class_metrics[2].booked, 3)
        self.assertEqual(sim.class_metrics[2].no_offer, 0)
        self.assertEqual(len(sim.calendar[0]), 4)
        self.assertEqual(sum(b.reserved_slot for b in sim.calendar[0]), 2)
        self.assert_calendar_capacity(sim)

    def test_class_1_first_backfill_reservation_processes_class_1_before_class_2(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=1,
                reserved_slots_per_day=1,
                release_reserved_slots=True,
            )
        )

        sim.process_daily_arrivals([2, 1], track_patients=True)

        self.assertEqual(sim.class_metrics[1].booked, 1)
        self.assertEqual(sim.class_metrics[2].booked, 0)
        self.assertEqual(sim.class_metrics[2].no_offer, 1)
        self.assertEqual(sim.calendar[0][0].patient_class, 1)
        self.assertTrue(sim.calendar[0][0].reserved_slot)
        self.assert_calendar_capacity(sim)

    def test_class_1_first_backfill_class_2_uses_leftover_reserved_after_class_1_batch(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=2,
                reserved_slots_per_day=2,
                release_reserved_slots=True,
            )
        )

        sim.process_daily_arrivals([1, 2], track_patients=True)

        self.assertEqual(sim.class_metrics[1].booked, 1)
        self.assertEqual(sim.class_metrics[2].booked, 1)
        self.assertEqual(sum(b.reserved_slot for b in sim.calendar[0]), 2)
        self.assert_calendar_capacity(sim)

    def test_class_1_first_backfill_class_1_takes_day_0_general_before_day_1_reserved(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=2,
                horizon_days=2,
                reserved_slots_per_day=1,
                release_reserved_slots=True,
            )
        )
        sim.calendar[0].append(
            Booking(
                patient_class=1,
                booking_delay=0,
                tracked=False,
                reserved_slot=True,
            )
        )

        sim.process_daily_arrivals([1], track_patients=True)

        self.assertEqual(len(sim.calendar[0]), 2)
        self.assertEqual(len(sim.calendar[1]), 0)
        self.assertFalse(sim.calendar[0][1].reserved_slot)
        self.assertEqual(sim.calendar[0][1].booking_delay, 0)
        self.assert_calendar_capacity(sim)

    def test_class_1_first_backfill_class_1_takes_day_0_reserved_when_available(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=2,
                horizon_days=2,
                reserved_slots_per_day=1,
                release_reserved_slots=True,
            )
        )

        sim.process_daily_arrivals([1], track_patients=True)

        self.assertEqual(sim.class_metrics[1].booked, 1)
        self.assertTrue(sim.calendar[0][0].reserved_slot)
        self.assertEqual(sim.calendar[0][0].booking_delay, 0)
        self.assert_calendar_capacity(sim)

    def test_class_1_first_backfill_class_2_prefers_same_day_leftover_reserved_to_later_general(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=2,
                horizon_days=2,
                reserved_slots_per_day=1,
                release_reserved_slots=True,
            )
        )
        sim.calendar[0].append(
            Booking(
                patient_class=2,
                booking_delay=0,
                tracked=False,
                reserved_slot=False,
            )
        )

        sim.process_daily_arrivals([2], track_patients=True)

        self.assertEqual(len(sim.calendar[0]), 2)
        self.assertEqual(len(sim.calendar[1]), 0)
        self.assertTrue(sim.calendar[0][1].reserved_slot)
        self.assertEqual(sim.calendar[0][1].booking_delay, 0)
        self.assert_calendar_capacity(sim)

    def test_class_1_first_backfill_class_2_uses_general_before_leftover_reserved_on_same_day(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=2,
                reserved_slots_per_day=1,
                release_reserved_slots=True,
            )
        )

        sim.process_daily_arrivals([2], track_patients=True)

        self.assertEqual(sim.class_metrics[2].booked, 1)
        self.assertFalse(sim.calendar[0][0].reserved_slot)
        self.assert_calendar_capacity(sim)

    def test_run_accounting_and_capacity_invariants(self) -> None:
        balk_rule = ThresholdRule(threshold=0, low=0.0, high=0.25)
        no_show_rule = ThresholdRule(threshold=0, low=0.10, high=0.20)
        policies = [
            self.make_config(
                slots_per_day=4,
                horizon_days=3,
                measure_days=8,
                cooldown_days=3,
                reserved_slots_per_day=0,
                lambda_per_day=3.0,
                balk_prob=balk_rule,
                cancel_prob=0.10,
                no_show_prob=no_show_rule,
                seed=101,
            ),
            self.make_config(
                slots_per_day=4,
                horizon_days=3,
                measure_days=8,
                cooldown_days=3,
                reserved_slots_per_day=2,
                lambda_per_day=3.0,
                balk_prob=balk_rule,
                cancel_prob=0.10,
                no_show_prob=no_show_rule,
                seed=102,
            ),
            self.make_config(
                slots_per_day=4,
                horizon_days=3,
                measure_days=8,
                cooldown_days=3,
                reserved_slots_per_day=2,
                release_reserved_slots=True,
                lambda_per_day=3.0,
                balk_prob=balk_rule,
                cancel_prob=0.10,
                no_show_prob=no_show_rule,
                seed=103,
            ),
        ]

        for config in policies:
            sim = ClinicAppointmentSimulation(config)
            result = sim.run()

            self.assert_accounting_invariants(result)
            self.assert_calendar_capacity(sim)


if __name__ == "__main__":
    unittest.main()
