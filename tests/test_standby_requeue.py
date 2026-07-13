from __future__ import annotations

import unittest

from analysis.metrics import outcome_rates_from_result, outcome_totals
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import (
    Booking,
    PatientClassParams,
    SimulationConfig,
    StandbyEntry,
    ThresholdRule,
)


ZERO_RULE = ThresholdRule(threshold=0, low=0.0, high=0.0)
ALWAYS_BALK = ThresholdRule(threshold=0, low=1.0, high=1.0)


class StandbyRequeueTest(unittest.TestCase):
    def make_config(
        self,
        *,
        slots_per_day: int = 1,
        horizon_days: int = 5,
        measure_days: int = 1,
        cooldown_days: int = 0,
        lambda_per_day: float = 0.0,
        balk_prob=ZERO_RULE,
        cancel_prob: float = 0.0,
        no_show_prob=ZERO_RULE,
        standby_prob: float = 0.0,
        max_standby_days=None,
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
            )
        return SimulationConfig(
            slots_per_day=slots_per_day,
            horizon_days=horizon_days,
            burn_in_days=0,
            measure_days=measure_days,
            cooldown_days=cooldown_days,
            classes=classes,
            seed=seed,
        )

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
    # Backward compatibility
    # ---------------------------------------------------------------

    def test_default_standby_prob_reproduces_original_balking(self) -> None:
        shared_kwargs = dict(
            slots_per_day=3,
            horizon_days=4,
            measure_days=20,
            cooldown_days=3,
            lambda_per_day=2.5,
            balk_prob=ThresholdRule(threshold=1, low=0.05, high=0.5),
            cancel_prob=0.1,
            no_show_prob=ThresholdRule(threshold=1, low=0.1, high=0.3),
            seed=555,
        )
        result_omitted = ClinicAppointmentSimulation(
            self.make_config(**shared_kwargs)
        ).run()
        result_explicit_zero = ClinicAppointmentSimulation(
            self.make_config(standby_prob=0.0, **shared_kwargs)
        ).run()

        for class_id in (1, 2):
            m1 = result_omitted.class_metrics[class_id]
            m2 = result_explicit_zero.class_metrics[class_id]
            self.assertEqual(m1.booked, m2.booked)
            self.assertEqual(m1.balked, m2.balked)
            self.assertEqual(m1.served, m2.served)
            self.assertEqual(m1.no_show, m2.no_show)
        self.assertEqual(result_omitted.average_utilization, result_explicit_zero.average_utilization)

    # ---------------------------------------------------------------
    # Joining standby
    # ---------------------------------------------------------------

    def test_would_be_balker_joins_standby_instead_of_exiting(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                horizon_days=3,
                balk_prob=ALWAYS_BALK,
                standby_prob=1.0,
            )
        )
        # Fill today's single slot so the arrival is offered a later day
        # (r > 0), guaranteeing a nonzero delay and a certain balk.
        sim.calendar[0].append(Booking(patient_class=2, booking_delay=0, tracked=False))

        sim.process_daily_arrivals([1], track_patients=True)

        metrics = sim.class_metrics[1]
        # Joining standby must not be counted as balked, booked, or
        # no_offer yet -- the decision is deferred until recall/expiry.
        self.assertEqual(metrics.booked, 0)
        self.assertEqual(metrics.balked, 0)
        self.assertEqual(metrics.no_offer, 0)
        self.assertEqual(metrics.total_offered_booking_delay, 0.0)
        self.assertEqual(metrics.standby_joined, 1)
        self.assertEqual(len(sim.standby_queue[1]), 1)
        entry = sim.standby_queue[1][0]
        self.assertEqual(entry.original_offered_delay, 1)
        self.assertEqual(entry.days_waited, 0)
        self.assertEqual(metrics.offered, metrics.booked + metrics.balked)

    # ---------------------------------------------------------------
    # Recall
    # ---------------------------------------------------------------

    def test_recall_counts_as_a_fresh_offer_not_a_reclassified_balk(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(slots_per_day=1, horizon_days=3, standby_prob=1.0)
        )
        # Fill r=0 so the nearest day this entry can actually claim is r=1.
        sim.calendar[0].append(Booking(patient_class=2, booking_delay=0, tracked=False))
        sim.standby_queue[1].append(
            StandbyEntry(patient_class=1, original_offered_delay=2, days_waited=0, tracked=True)
        )

        sim.process_standby_recalls()

        self.assertEqual(len(sim.standby_queue[1]), 0)
        self.assertEqual(len(sim.calendar[1]), 1)
        recalled_booking = sim.calendar[1][0]
        self.assertTrue(recalled_booking.standby_recalled)
        self.assertEqual(recalled_booking.booking_delay, 1)

        metrics = sim.class_metrics[1]
        self.assertEqual(metrics.balked, 0)
        self.assertEqual(metrics.no_offer, 0)
        self.assertEqual(metrics.booked, 1)
        self.assertEqual(metrics.standby_recalled, 1)
        self.assertEqual(metrics.total_booking_delay, 1)
        # The recalled day is the real offer: offered_delay and accepted
        # delay both use it, exactly like a normal direct-accept booking.
        self.assertEqual(metrics.total_offered_booking_delay, 1)
        self.assertEqual(metrics.offered, metrics.booked + metrics.balked)

    def test_recall_only_offers_strictly_earlier_days(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(slots_per_day=1, horizon_days=3, standby_prob=1.0)
        )
        # Fill r=0 so it cannot satisfy the entry; r=1 is open but is not
        # < original_offered_delay (1), so it must not be offered either.
        sim.calendar[0].append(Booking(patient_class=2, booking_delay=0, tracked=False))
        sim.standby_queue[1].append(
            StandbyEntry(patient_class=1, original_offered_delay=1, days_waited=0, tracked=True)
        )

        sim.process_standby_recalls()

        self.assertEqual(len(sim.standby_queue[1]), 1)
        self.assertEqual(len(sim.calendar[1]), 0)

    def test_fifo_head_blocks_later_entries_on_a_given_day(self) -> None:
        """
        Documents the FIFO simplification: only the head of the queue is
        considered for a given residual day. A later-joined entry with a
        more favorable original_offered_delay is not skipped ahead of an
        earlier-joined entry that isn't eligible yet, even across the
        rest of the day's scan.
        """
        sim = ClinicAppointmentSimulation(
            self.make_config(slots_per_day=1, horizon_days=3, standby_prob=1.0)
        )
        # r=0 is full, so the front entry (which only wants r=0) can never
        # be recalled, and it blocks the second entry (which would happily
        # take r=1 or r=2) from ever being considered.
        sim.calendar[0].append(Booking(patient_class=2, booking_delay=0, tracked=False))
        sim.standby_queue[1].append(
            StandbyEntry(patient_class=1, original_offered_delay=1, days_waited=0, tracked=True)
        )
        sim.standby_queue[1].append(
            StandbyEntry(patient_class=1, original_offered_delay=5, days_waited=0, tracked=True)
        )

        sim.process_standby_recalls()

        self.assertEqual(len(sim.standby_queue[1]), 2)
        self.assertEqual(len(sim.calendar[0]), 1)
        self.assertEqual(len(sim.calendar[1]), 0)
        self.assertEqual(len(sim.calendar[2]), 0)

    def test_recall_eligibility_delay_blocks_early_recall(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=1,
                horizon_days=3,
                standby_prob=1.0,
            )
        )
        # eligible_after_days is not exposed through make_config's
        # standby_prob-only kwargs, so set it directly on the class params.
        params = sim.config.classes[1]
        object.__setattr__(params, "standby_eligible_after_days", 2)
        sim.standby_queue[1].append(
            StandbyEntry(patient_class=1, original_offered_delay=2, days_waited=0, tracked=True)
        )

        # days_waited=0 < eligible_after=2: not yet eligible even though
        # r=0 < original_offered_delay and capacity is open.
        sim.process_standby_recalls()
        self.assertEqual(len(sim.standby_queue[1]), 1)
        self.assertEqual(len(sim.calendar[0]), 0)

        sim.age_standby_queue()  # days_waited -> 1, still < 2
        sim.process_standby_recalls()
        self.assertEqual(len(sim.standby_queue[1]), 1)

        sim.age_standby_queue()  # days_waited -> 2, now eligible
        sim.process_standby_recalls()
        self.assertEqual(len(sim.standby_queue[1]), 0)
        self.assertEqual(len(sim.calendar[0]), 1)

    def test_recall_respects_capacity(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(slots_per_day=1, horizon_days=3, standby_prob=1.0)
        )
        # Fill both r=0 and r=1 so every day this entry would accept
        # (r < original_offered_delay=2) is already at capacity.
        sim.calendar[0].append(Booking(patient_class=2, booking_delay=0, tracked=False))
        sim.calendar[1].append(Booking(patient_class=2, booking_delay=1, tracked=False))
        sim.standby_queue[1].append(
            StandbyEntry(patient_class=1, original_offered_delay=2, days_waited=0, tracked=True)
        )

        sim.process_standby_recalls()

        # No day it wants has open capacity, so the entry stays queued.
        self.assertEqual(len(sim.standby_queue[1]), 1)
        self.assertEqual(len(sim.calendar[0]), 1)
        self.assertEqual(len(sim.calendar[1]), 1)

    # ---------------------------------------------------------------
    # Expiry
    # ---------------------------------------------------------------

    def test_expired_entry_is_counted_as_no_offer_not_balked(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=1, horizon_days=3, standby_prob=1.0, max_standby_days=1
            )
        )
        sim.standby_queue[1].append(
            StandbyEntry(patient_class=1, original_offered_delay=2, days_waited=0, tracked=True)
        )

        sim.age_standby_queue()

        self.assertEqual(len(sim.standby_queue[1]), 0)
        self.assertEqual(sim.class_metrics[1].standby_expired, 1)
        self.assertEqual(sim.class_metrics[1].balked, 0)
        self.assertEqual(sim.class_metrics[1].no_offer, 1)

    def test_entry_survives_until_max_standby_days(self) -> None:
        sim = ClinicAppointmentSimulation(
            self.make_config(
                slots_per_day=1, horizon_days=3, standby_prob=1.0, max_standby_days=2
            )
        )
        sim.standby_queue[1].append(
            StandbyEntry(patient_class=1, original_offered_delay=5, days_waited=0, tracked=True)
        )

        sim.age_standby_queue()
        self.assertEqual(len(sim.standby_queue[1]), 1)
        self.assertEqual(sim.standby_queue[1][0].days_waited, 1)

        sim.age_standby_queue()
        self.assertEqual(len(sim.standby_queue[1]), 0)
        self.assertEqual(sim.class_metrics[1].standby_expired, 1)
        self.assertEqual(sim.class_metrics[1].no_offer, 1)

    # ---------------------------------------------------------------
    # Full-run invariants
    # ---------------------------------------------------------------

    def test_full_run_accounting_invariants_hold_with_standby_enabled(self) -> None:
        config = self.make_config(
            slots_per_day=4,
            horizon_days=10,
            measure_days=20,
            cooldown_days=10,
            lambda_per_day=3.0,
            balk_prob=ThresholdRule(threshold=2, low=0.0, high=0.6),
            cancel_prob=0.1,
            no_show_prob=ThresholdRule(threshold=1, low=0.05, high=0.3),
            standby_prob=0.6,
            max_standby_days=5,
            seed=99,
        )
        result = ClinicAppointmentSimulation(config).run()

        self.assert_accounting_invariants(result)
        for class_id in (1, 2):
            metrics = result.class_metrics[class_id]
            self.assertGreaterEqual(metrics.standby_recalled, 0)
            self.assertLessEqual(metrics.standby_recalled, metrics.standby_joined)


if __name__ == "__main__":
    unittest.main()
