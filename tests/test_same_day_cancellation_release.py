from __future__ import annotations

import unittest

from simulation.engine import ClinicAppointmentSimulation
from simulation.model import Booking, PatientClassParams, SimulationConfig, ThresholdRule


ZERO_RULE = ThresholdRule(threshold=0, low=0.0, high=0.0)
ALWAYS_RULE = ThresholdRule(threshold=0, low=1.0, high=1.0)


class SameDayCancellationReleaseTest(unittest.TestCase):
    """
    Tests for the two opt-in, default-off SimulationConfig flags added on
    top of the reservation feature:

    - same_day_cancellation_enabled: extends apply_start_of_day_cancellations
      to residual day r = 0, using the same per-class cancel_prob already
      used for r >= 1. Default False reproduces the original "no same-day
      cancellations" behavior exactly.
    - release_unused_reservation_same_day: at r = 0 only, pools reserved
      capacity with general capacity so any class can take an idle
      reserved slot on a first-come basis. Default False reproduces the
      original strict reservation behavior exactly. A no-op unless a
      reservation is actually configured.

    Both flags are independent of each other and of standby/requeue.
    """

    def make_config(
        self,
        *,
        slots_per_day: int = 4,
        horizon_days: int = 2,
        measure_days: int = 1,
        cooldown_days: int = 0,
        lambda_per_day: float = 0.0,
        balk_prob=ZERO_RULE,
        cancel_prob: float = 0.0,
        no_show_prob=ZERO_RULE,
        reserved_slots_per_day: int = 0,
        reserved_window_days=None,
        standby_prob: float = 0.0,
        max_standby_days=None,
        standby_eligible_after_days=None,
        same_day_cancellation_enabled: bool = False,
        release_unused_reservation_same_day: bool = False,
        seed=None,
    ) -> SimulationConfig:
        classes = {}
        for class_id in (1, 2):
            classes[class_id] = PatientClassParams(
                class_id=class_id,
                lambda_per_day=lambda_per_day,
                balk_prob=balk_prob,
                cancel_prob=cancel_prob,
                no_show_prob=no_show_prob,
                standby_prob=standby_prob,
                max_standby_days=max_standby_days,
                standby_eligible_after_days=standby_eligible_after_days,
            )
        return SimulationConfig(
            slots_per_day=slots_per_day,
            horizon_days=horizon_days,
            burn_in_days=0,
            measure_days=measure_days,
            cooldown_days=cooldown_days,
            classes=classes,
            seed=seed,
            reserved_class_id=1 if reserved_slots_per_day > 0 else None,
            reserved_slots_per_day=reserved_slots_per_day,
            reserved_window_days=reserved_window_days,
            same_day_cancellation_enabled=same_day_cancellation_enabled,
            release_unused_reservation_same_day=release_unused_reservation_same_day,
        )

    # ---------------------------------------------------------------
    # Backward compatibility: both flags default off
    # ---------------------------------------------------------------

    def test_config_defaults_preserve_old_behavior(self) -> None:
        config = self.make_config()
        self.assertFalse(config.same_day_cancellation_enabled)
        self.assertFalse(config.release_unused_reservation_same_day)

    def test_same_day_cancellation_disabled_by_default_leaves_r0_untouched(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(cancel_prob=1.0, same_day_cancellation_enabled=False)
        )
        sim.calendar[0].append(
            Booking(patient_class=1, booking_delay=0, tracked=True)
        )
        sim.apply_start_of_day_cancellations()

        self.assertEqual(len(sim.calendar[0]), 1)
        self.assertEqual(sim.class_metrics[1].canceled, 0)

    def test_release_flag_off_by_default_keeps_strict_reservation(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=4,
                horizon_days=1,
                reserved_slots_per_day=2,
                release_unused_reservation_same_day=False,
            )
        )
        sim.process_daily_arrivals([2, 2, 2, 2], track_patients=True)

        self.assertEqual(sim.class_metrics[2].booked, 2)
        self.assertEqual(sim.class_metrics[2].no_offer, 2)

    # ---------------------------------------------------------------
    # same_day_cancellation_enabled = True
    # ---------------------------------------------------------------

    def test_enabled_same_day_cancellation_removes_r0_booking(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(cancel_prob=1.0, same_day_cancellation_enabled=True)
        )
        sim.calendar[0].append(
            Booking(patient_class=1, booking_delay=0, tracked=True)
        )
        sim.apply_start_of_day_cancellations()

        self.assertEqual(len(sim.calendar[0]), 0)
        self.assertEqual(sim.class_metrics[1].canceled, 1)

    def test_same_day_cancellation_still_respects_cancel_prob_zero(self) -> None:
        # Enabling the flag must not force cancellation -- cancel_prob = 0
        # should still leave the booking untouched at r = 0.
        sim = ClinicAppointmentSimulation(
            self.make_config(cancel_prob=0.0, same_day_cancellation_enabled=True)
        )
        sim.calendar[0].append(
            Booking(patient_class=1, booking_delay=0, tracked=True)
        )
        sim.apply_start_of_day_cancellations()

        self.assertEqual(len(sim.calendar[0]), 1)
        self.assertEqual(sim.class_metrics[1].canceled, 0)

    def test_same_day_cancellation_not_double_counted_as_no_show(self) -> None:
        # A same-day-canceled booking must be removed before serve_today
        # runs, so it is never also counted as a no-show or a served visit.
        sim = ClinicAppointmentSimulation(
            self.make_config(
                cancel_prob=1.0,
                no_show_prob=ALWAYS_RULE,
                same_day_cancellation_enabled=True,
            )
        )
        sim.calendar[0].append(
            Booking(patient_class=1, booking_delay=0, tracked=True)
        )

        sim.apply_start_of_day_cancellations()
        sim.serve_today(count_slot_metrics=True)

        self.assertEqual(sim.class_metrics[1].canceled, 1)
        self.assertEqual(sim.class_metrics[1].no_show, 0)
        self.assertEqual(sim.class_metrics[1].served, 0)
        self.assertEqual(sim.slot_metrics.no_show_slots, 0)
        self.assertEqual(sim.slot_metrics.served_slots, 0)
        self.assertEqual(sim.slot_metrics.booked_slots, 0)

    def test_untracked_same_day_cancellation_does_not_affect_class_metrics(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(cancel_prob=1.0, same_day_cancellation_enabled=True)
        )
        sim.calendar[0].append(
            Booking(patient_class=1, booking_delay=0, tracked=False)
        )
        sim.apply_start_of_day_cancellations()

        self.assertEqual(len(sim.calendar[0]), 0)
        self.assertEqual(sim.class_metrics[1].canceled, 0)

    # ---------------------------------------------------------------
    # release_unused_reservation_same_day = True
    # ---------------------------------------------------------------

    def test_release_pools_reserved_capacity_for_any_class_at_r0(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=4,
                reserved_slots_per_day=2,
                release_unused_reservation_same_day=True,
            )
        )
        sim.process_daily_arrivals([2, 2, 2, 2], track_patients=True)

        self.assertEqual(sim.class_metrics[2].booked, 4)
        self.assertEqual(sim.class_metrics[2].no_offer, 0)
        # Under release, r = 0 capacity is pooled: nothing is tagged as a
        # reserved-slot booking anymore, since the reserved/general split
        # was never applied for this day.
        self.assertEqual(sum(b.reserved_slot for b in sim.calendar[0]), 0)

    def test_release_still_lets_class_1_fill_all_of_its_own_slots(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=4,
                reserved_slots_per_day=2,
                release_unused_reservation_same_day=True,
            )
        )
        sim.process_daily_arrivals([1, 1, 1, 1], track_patients=True)

        self.assertEqual(sim.class_metrics[1].booked, 4)
        self.assertEqual(sim.class_metrics[1].no_offer, 0)

    def test_release_applies_only_at_r0_not_further_out(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=4,
                horizon_days=2,
                reserved_slots_per_day=2,
                release_unused_reservation_same_day=True,
            )
        )
        # Fill r = 0 to capacity so the search must move on to r = 1.
        for _ in range(4):
            sim.calendar[0].append(
                Booking(patient_class=2, booking_delay=0, tracked=False)
            )

        offer = sim.find_earliest_open_day(2)

        # At r = 1, release does not apply: Class 2 can only reach the
        # 2 general slots, not the 2 reserved ones -- same as the
        # original strict behavior at any non-zero residual day.
        self.assertEqual(offer, (1, False))

    def test_release_never_exceeds_slots_per_day_combined_with_cancellation(self) -> None:
        # Regression guard mirroring
        # test_reservation_window_transition_never_exceeds_slots_per_day:
        # combining release with same-day cancellation and real stochastic
        # arrivals must never push a day's utilization above 100%.
        balk_rule = ThresholdRule(threshold=1, low=0.05, high=0.2)
        no_show_rule = ThresholdRule(threshold=1, low=0.05, high=0.15)
        config = self.make_config(
            slots_per_day=6,
            horizon_days=5,
            measure_days=60,
            cooldown_days=5,
            lambda_per_day=4.0,
            balk_prob=balk_rule,
            cancel_prob=0.1,
            no_show_prob=no_show_rule,
            reserved_slots_per_day=3,
            same_day_cancellation_enabled=True,
            release_unused_reservation_same_day=True,
            seed=2026,
        )
        sim = ClinicAppointmentSimulation(config)
        result = sim.run()

        self.assertLessEqual(result.average_utilization, 1.0)
        self.assertGreaterEqual(result.average_utilization, 0.0)
        for day_bookings in sim.calendar:
            self.assertLessEqual(len(day_bookings), config.slots_per_day)

    # ---------------------------------------------------------------
    # Interaction with standby/requeue
    # ---------------------------------------------------------------

    def test_standby_recall_reaches_r0_opening_freed_by_same_day_cancellation(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=1,
                horizon_days=3,
                cancel_prob=1.0,
                same_day_cancellation_enabled=True,
                standby_prob=1.0,
            )
        )
        # A class-1 patient joined standby earlier having declined an
        # offer 2 days out; only a strictly earlier day (r < 2) qualifies
        # for recall.
        from simulation.model import StandbyEntry

        sim.standby_queue[1].append(
            StandbyEntry(
                patient_class=1,
                original_offered_delay=2,
                days_waited=0,
                tracked=True,
            )
        )
        # Today's only slot is occupied, but will cancel this morning.
        sim.calendar[0].append(
            Booking(patient_class=2, booking_delay=0, tracked=False)
        )

        sim.apply_start_of_day_cancellations()
        sim.process_standby_recalls()

        self.assertEqual(len(sim.standby_queue[1]), 0)
        self.assertEqual(len(sim.calendar[0]), 1)
        self.assertTrue(sim.calendar[0][0].standby_recalled)
        self.assertEqual(sim.class_metrics[1].standby_recalled, 1)

    def test_standby_recall_does_not_reach_r0_when_cancellation_disabled(self) -> None:
        # Without same_day_cancellation_enabled, r = 0 bookings can never
        # cancel, so a standby entry can never be recalled into today --
        # the same-day opening this test relies on simply never occurs.
        from simulation.model import StandbyEntry

        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=1,
                horizon_days=3,
                cancel_prob=1.0,
                same_day_cancellation_enabled=False,
                standby_prob=1.0,
            )
        )
        sim.standby_queue[1].append(
            StandbyEntry(
                patient_class=1,
                original_offered_delay=2,
                days_waited=0,
                tracked=True,
            )
        )
        sim.calendar[0].append(
            Booking(patient_class=2, booking_delay=0, tracked=False)
        )
        # Also fill r = 1 so the only day that could possibly open up is
        # r = 0 -- isolating whether a same-day opening (which requires
        # same_day_cancellation_enabled) is what the recall depends on,
        # rather than r = 1 simply having been open the whole time.
        sim.calendar[1].append(
            Booking(patient_class=2, booking_delay=1, tracked=False)
        )

        sim.apply_start_of_day_cancellations()
        sim.process_standby_recalls()

        self.assertEqual(len(sim.standby_queue[1]), 1)
        self.assertEqual(len(sim.calendar[0]), 1)
        self.assertFalse(sim.calendar[0][0].standby_recalled)


if __name__ == "__main__":
    unittest.main()
