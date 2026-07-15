from __future__ import annotations

import unittest

from analysis.metrics import outcome_rates_from_result, outcome_totals
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import Booking, PatientClassParams, SimulationConfig, ThresholdRule


ZERO_RULE = ThresholdRule(threshold=0, low=0.0, high=0.0)


class WindowedReservationTest(unittest.TestCase):
    def make_config(
        self,
        *,
        slots_per_day: int = 1,
        horizon_days: int = 3,
        measure_days: int = 1,
        cooldown_days: int = 0,
        reserved_slots_per_day: int = 1,
        reserved_window_days=None,
        lambda_per_day: float = 0.0,
        balk_prob=ZERO_RULE,
        cancel_prob: float = 0.0,
        no_show_prob=ZERO_RULE,
        seed=None,
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
            reserved_window_days=reserved_window_days,
        )

    def assert_calendar_capacity(self, sim: ClinicAppointmentSimulation) -> None:
        for day_bookings in sim.calendar:
            self.assertLessEqual(len(day_bookings), sim.config.slots_per_day)

    def assert_accounting_invariants(self, result) -> None:
        totals = outcome_totals(result)
        resolved_booked = totals["served"] + totals["canceled"] + totals["no_show"]
        unresolved_booked = totals["booked"] - resolved_booked

        self.assertGreaterEqual(unresolved_booked, 0)
        self.assertEqual(
            totals["arrivals"],
            totals["served"]
            + totals["balked"]
            + totals["no_offer"]
            + totals["canceled"]
            + totals["no_show"]
            + unresolved_booked,
        )
        self.assertEqual(totals["offered"], totals["booked"] + totals["balked"])

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

    # ---------------------------------------------------------------
    # Window semantics
    # ---------------------------------------------------------------

    def test_class_2_is_locked_out_within_window_but_recovers_beyond_it(self) -> None:
        """
        slots_per_day=1 with the single slot fully reserved for class 1
        inside the window (r=0,1) means class 2 cannot be offered
        anything until r=2, which sits outside the 2-day window and is
        therefore plain pooled capacity.
        """
        sim = ClinicAppointmentSimulation(
            self.make_config(reserved_window_days=2)
        )

        sim.process_daily_arrivals([2], track_patients=True)

        self.assertEqual(sim.class_metrics[2].booked, 1)
        self.assertEqual(len(sim.calendar[0]), 0)
        self.assertEqual(len(sim.calendar[1]), 0)
        self.assertEqual(len(sim.calendar[2]), 1)
        self.assertFalse(sim.calendar[2][0].reserved_slot)
        self.assertEqual(sim.calendar[2][0].booking_delay, 2)
        self.assert_calendar_capacity(sim)

    def test_class_1_still_uses_reserved_capacity_inside_window(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(reserved_window_days=2)
        )

        sim.process_daily_arrivals([1], track_patients=True)

        self.assertEqual(sim.class_metrics[1].booked, 1)
        self.assertEqual(len(sim.calendar[0]), 1)
        self.assertTrue(sim.calendar[0][0].reserved_slot)
        self.assert_calendar_capacity(sim)

    def test_no_reservation_effect_at_or_beyond_window_boundary(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=2,
                horizon_days=4,
                reserved_slots_per_day=1,
                reserved_window_days=2,
            )
        )

        # r=2 is at the boundary (r >= window), so it must behave as plain
        # pooled capacity for every class, with no reserved/general split.
        self.assertEqual(sim._slot_offer_at(2, class_id=1), False)
        self.assertEqual(sim._slot_offer_at(2, class_id=2), False)

        # r=1 is still inside the window, so the split applies.
        self.assertEqual(sim._slot_offer_at(1, class_id=1), True)
        self.assertEqual(sim._slot_offer_at(1, class_id=2), False)

    def test_reservation_window_transition_never_exceeds_slots_per_day(self) -> None:
        """
        Regression test for a capacity-overflow bug: a calendar day that
        filled to slots_per_day under the plain pooled rule (while its
        residual offset r was still >= window) must NOT accept an
        additional reserved-class booking once the day rolls forward and
        its r drops inside the window.

        Before the fix, the r < window branch only checked
        reserved_used < reserved_slots without first re-checking total
        occupancy, so a day already at capacity could still take one more
        reserved-class booking, pushing len(calendar[r]) above
        slots_per_day (and average_utilization above 100%).
        """
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=1,
                horizon_days=3,
                reserved_slots_per_day=1,
                reserved_window_days=2,
            )
        )

        # Simulate a day that filled up via the plain pooled rule while it
        # was still outside the window (r=2 >= window=2), booked by the
        # non-reserved class.
        sim.calendar[2].append(
            Booking(patient_class=2, booking_delay=2, tracked=True, reserved_slot=False)
        )

        # Roll the calendar forward one day: the booking above now sits at
        # r=1, which is inside the window (r=1 < window=2).
        sim.roll_calendar_forward_one_day()
        self.assertEqual(len(sim.calendar[1]), 1)

        # The reserved class must NOT be offered this day: it is already
        # at slots_per_day capacity, even though reserved_used == 0.
        self.assertIsNone(sim._slot_offer_at(1, class_id=1))
        self.assert_calendar_capacity(sim)

        # End-to-end: also fill r=0 (today) so an arriving reserved-class
        # patient is forced to search past the already-full r=1 day. With
        # the fix, that patient must skip r=1 entirely and land on r=2
        # (outside the window, plain pooled, still empty) rather than
        # overflowing r=1.
        sim.calendar[0].append(
            Booking(patient_class=1, booking_delay=0, tracked=True, reserved_slot=True)
        )
        sim.process_daily_arrivals([1], track_patients=True)
        self.assertEqual(sim.class_metrics[1].booked, 1)
        self.assertEqual(len(sim.calendar[1]), 1)
        self.assertEqual(len(sim.calendar[2]), 1)
        self.assert_calendar_capacity(sim)

    def test_window_none_reproduces_full_horizon_reservation(self) -> None:
        """
        An unset window must behave identically to a window covering the
        whole horizon, including the RNG stream, since this is the
        backward-compatible default.
        """
        shared_kwargs = dict(
            slots_per_day=4,
            horizon_days=5,
            measure_days=20,
            cooldown_days=3,
            reserved_slots_per_day=2,
            lambda_per_day=3.0,
            balk_prob=ThresholdRule(threshold=1, low=0.05, high=0.4),
            cancel_prob=0.1,
            no_show_prob=ThresholdRule(threshold=1, low=0.1, high=0.3),
            seed=777,
        )

        result_default = ClinicAppointmentSimulation(
            self.make_config(reserved_window_days=None, **shared_kwargs)
        ).run()
        result_full_window = ClinicAppointmentSimulation(
            self.make_config(reserved_window_days=5, **shared_kwargs)
        ).run()

        self.assertEqual(
            result_default.average_utilization, result_full_window.average_utilization
        )
        self.assertEqual(
            result_default.class_metrics[1].booked,
            result_full_window.class_metrics[1].booked,
        )
        self.assertEqual(
            result_default.class_metrics[2].booked,
            result_full_window.class_metrics[2].booked,
        )
        self.assertEqual(
            result_default.slot_metrics.reserved_slots_booked,
            result_full_window.slot_metrics.reserved_slots_booked,
        )

    # ---------------------------------------------------------------
    # Fill-rate diagnostic and full-run invariants
    # ---------------------------------------------------------------

    def test_reserved_slot_fill_rate_matches_manual_computation(self) -> None:
        config = self.make_config(
            slots_per_day=2,
            horizon_days=2,
            measure_days=6,
            cooldown_days=1,
            reserved_slots_per_day=1,
            reserved_window_days=1,
            lambda_per_day=1.5,
            balk_prob=ZERO_RULE,
            cancel_prob=0.0,
            no_show_prob=ZERO_RULE,
            seed=13,
        )
        result = ClinicAppointmentSimulation(config).run()

        expected_capacity = config.reserved_slots_per_day * config.measure_days
        self.assertEqual(result.reserved_slot_capacity, expected_capacity)
        expected_fill_rate = (
            result.slot_metrics.reserved_slots_booked / expected_capacity
            if expected_capacity > 0
            else 0.0
        )
        self.assertAlmostEqual(result.reserved_slot_fill_rate, expected_fill_rate)
        self.assertGreaterEqual(result.reserved_slot_fill_rate, 0.0)
        self.assertLessEqual(result.reserved_slot_fill_rate, 1.0)

    def test_full_run_accounting_invariants_hold_with_window(self) -> None:
        config = self.make_config(
            slots_per_day=4,
            horizon_days=6,
            measure_days=15,
            cooldown_days=3,
            reserved_slots_per_day=2,
            reserved_window_days=3,
            lambda_per_day=3.0,
            balk_prob=ThresholdRule(threshold=1, low=0.0, high=0.3),
            cancel_prob=0.1,
            no_show_prob=ThresholdRule(threshold=1, low=0.05, high=0.25),
            seed=2024,
        )
        result = ClinicAppointmentSimulation(config).run()

        self.assert_accounting_invariants(result)


if __name__ == "__main__":
    unittest.main()
