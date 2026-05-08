from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from simulation.model import (
    Booking,
    ClassMetrics,
    SimulationConfig,
    SimulationResults,
    SlotMetrics,
)


# =========================
# Simulation engine
# =========================

class ClinicAppointmentSimulation:
    """
    Day-level clinic appointment simulation with:
    - 2+ patient classes
    - daily arrivals generated once per day
    - one random permutation of the day's arrivals
    - FCFS booking to the earliest day with available capacity, including same-day
    - delay-dependent balking
    - delay-dependent no-show
    - constant cancellation applied once per day to future appointments only
    - no same-day cancellations
    - no rebooking of no-show slots
    - day-level calendar state with booking audit records
    - derived summary state at the start of each measured day

    Internal calendar representation:
        self.calendar[r] is a list of Booking objects scheduled for day D + r

    Capacity rule:
        len(self.calendar[r]) <= slots_per_day
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Day-level calendar: one booking list per residual day
        self.calendar: List[List[Booking]] = [
            [] for _ in range(config.horizon_days)
        ]

        self.class_metrics: Dict[int, ClassMetrics] = {
            class_id: ClassMetrics() for class_id in config.classes
        }
        self.slot_metrics = SlotMetrics()
        self.total_value: float = 0.0

        # One summary state per measured day, recorded after start-of-day cancellations
        self.daily_summary_states: List[Dict[int, List[int]]] = []

    # -------------------------
    # State views
    # -------------------------

    def full_state_view(self) -> List[List[Union[int, Tuple[int, int]]]]:
        """
        Return a padded day-level view for compatibility with existing outputs.

        Each row is a list of length slots_per_day:
        - booked patients are shown as (i, tau)
        - remaining capacity is shown as 0

        Note: within-day ordering in this view is not a true slot position anymore.
        It is only a diagnostic representation.
        """
        view: List[List[Union[int, Tuple[int, int]]]] = []

        for day_bookings in self.calendar:
            row_view: List[Union[int, Tuple[int, int]]] = [
                (b.patient_class, b.booking_delay) for b in day_bookings
            ]
            remaining = self.config.slots_per_day - len(day_bookings)
            row_view.extend([0] * remaining)
            view.append(row_view)

        return view

    def summary_state(self) -> Dict[int, List[int]]:
        """
        Derived summary state at the start of a day:
            X^D_{i,r} = number of class-i patients scheduled for day D+r
        """
        summary: Dict[int, List[int]] = {
            class_id: [0 for _ in range(self.config.horizon_days)]
            for class_id in self.config.classes
        }

        for r in range(self.config.horizon_days):
            for booking in self.calendar[r]:
                summary[booking.patient_class][r] += 1

        return summary

    # -------------------------
    # Booking logic
    # -------------------------

    def find_earliest_open_day(self) -> Optional[int]:
        """
        Find the earliest day with available capacity.

        Same-day booking is allowed, so the search starts at r = 0.
        """
        for r in range(self.config.horizon_days):
            if len(self.calendar[r]) < self.config.slots_per_day:
                return r
        return None

    def generate_daily_arrival_order(self) -> List[int]:
        """
        Generate class-specific daily Poisson arrivals, convert them into
        individual patients, and randomize the day's order once.
        """
        arrivals: List[int] = []

        for class_id, params in self.config.classes.items():
            n = int(self.rng.poisson(params.lambda_per_day))
            arrivals.extend([class_id] * n)

        if arrivals:
            arrivals = self.rng.permutation(arrivals).tolist()

        return arrivals

    def process_daily_arrivals(
        self,
        ordered_arrivals: List[int],
        track_patients: bool,
    ) -> None:
        """
        Process the full day's arrivals in one random order.

        If no slot is available for one patient, then that patient and all
        remaining arrivals for the day are counted as no_offer and the
        booking step stops for the day.
        """
        if track_patients:
            for class_id in ordered_arrivals:
                self.class_metrics[class_id].arrivals += 1

        for idx, class_id in enumerate(ordered_arrivals):
            params = self.config.classes[class_id]
            metrics = self.class_metrics[class_id]

            offered_day = self.find_earliest_open_day()

            if offered_day is None:
                if track_patients:
                    for remaining_class_id in ordered_arrivals[idx:]:
                        self.class_metrics[remaining_class_id].no_offer += 1
                return

            tau = offered_day  # offered booking delay in days; tau = 0 is allowed

            # Record the offered delay including balk.
            if track_patients:
                metrics.total_offered_booking_delay += tau

            # Balking decision
            if self.rng.random() < params.balk_prob(tau):
                if track_patients:
                    metrics.balked += 1
                continue

            # Accept and book
            self.calendar[offered_day].append(
                Booking(
                    patient_class=class_id,
                    booking_delay=tau,
                    tracked=track_patients,
                )
            )

            if track_patients:
                metrics.booked += 1
                metrics.total_booking_delay += tau

    # -------------------------
    # Daily service logic
    # -------------------------

    def serve_today(self, count_slot_metrics: bool) -> None:
        """Resolve all appointments scheduled for today (r = 0)."""

        todays_bookings = self.calendar[0]
        booked_today = len(todays_bookings)
        served_today = 0

        if count_slot_metrics:
            self.slot_metrics.booked_slots += booked_today

        for booking in todays_bookings:
            class_id = booking.patient_class
            tau = booking.booking_delay
            params = self.config.classes[class_id]
            metrics = self.class_metrics[class_id]

            if self.rng.random() < params.no_show_prob(tau):
                if booking.tracked:
                    metrics.no_show += 1
                if count_slot_metrics:
                    self.slot_metrics.no_show_slots += 1
            else:
                served_today += 1

                if booking.tracked:
                    metrics.served += 1
                if count_slot_metrics:
                    self.slot_metrics.served_slots += 1

                self.total_value += params.value

        if count_slot_metrics:
            daily_utilization = served_today / self.config.slots_per_day
            self.slot_metrics.daily_utilization_sum += daily_utilization
            self.slot_metrics.measured_days += 1

        self.calendar[0] = []

    # -------------------------
    # Start-of-day cancellations
    # -------------------------

    def apply_start_of_day_cancellations(self) -> None:
        """
        Apply cancellations only to future appointments with r >= 1.
        Same-day cancellations are not allowed.

        All future bookings may cancel, but only tracked bookings count toward
        class-level cancellation metrics.
        """
        for r in range(1, self.config.horizon_days):
            surviving_bookings: List[Booking] = []

            for booking in self.calendar[r]:
                class_id = booking.patient_class
                params = self.config.classes[class_id]

                if self.rng.random() < params.cancel_prob:
                    if booking.tracked:
                        self.class_metrics[class_id].canceled += 1
                else:
                    surviving_bookings.append(booking)

            self.calendar[r] = surviving_bookings

    def roll_calendar_forward_one_day(self) -> None:
        """
        End of day transition:
        - drop day 0
        - shift future days forward by one
        - append a new empty day at the horizon end
        """
        self.calendar.pop(0)
        self.calendar.append([])

    # -------------------------
    # Main run
    # -------------------------

    def run(self) -> SimulationResults:
        """
        Run the day-level simulation with:
        - burn-in days
        - measurement days
        - cooldown days

        Day order:
        1. start-of-day cancellations on future appointments
        2. record start-of-day summary state
        3. generate all daily arrivals
        4. randomly permute arrivals
        5. process offers/balking in FCFS order
        6. capture final calendar snapshot on the last simulated day
        7. resolve no-shows/service for today's scheduled patients
        8. roll the calendar forward
        """
        total_days = (
            self.config.burn_in_days
            + self.config.measure_days
            + self.config.cooldown_days
        )

        first_measure_day = self.config.burn_in_days
        last_measure_day_exclusive = self.config.burn_in_days + self.config.measure_days

        final_full_state_snapshot = None

        for day in range(total_days):
            in_measurement_window = first_measure_day <= day < last_measure_day_exclusive

            # 1. Start-of-day cancellations for future appointments only
            self.apply_start_of_day_cancellations()

            # 2. Record start-of-day summary state after cancellations
            if in_measurement_window:
                start_of_day_summary = self.summary_state()
                self.daily_summary_states.append({
                    class_id: counts.copy()
                    for class_id, counts in start_of_day_summary.items()
                })

            # 3-5. Generate, permute, and process the day's arrivals
            ordered_arrivals = self.generate_daily_arrival_order()
            self.process_daily_arrivals(
                ordered_arrivals=ordered_arrivals,
                track_patients=in_measurement_window,
            )

            # 6. Capture the final calendar view before service and before rolling forward
            if day == total_days - 1:
                final_full_state_snapshot = self.full_state_view()

            # 7. Resolve today's scheduled appointments
            self.serve_today(count_slot_metrics=in_measurement_window)

            # 8. Move to next day
            self.roll_calendar_forward_one_day()

        return SimulationResults(
            class_metrics=self.class_metrics,
            slot_metrics=self.slot_metrics,
            total_slots=self.config.measure_days * self.config.slots_per_day,
            total_value=self.total_value,
            daily_summary_states=self.daily_summary_states,
            final_full_state=final_full_state_snapshot,
        )