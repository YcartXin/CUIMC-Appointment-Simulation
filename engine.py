from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from model import (
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
    Slot-by-slot clinic appointment simulation with:
    - 2+ patient classes
    - FCFS booking to earliest open slot strictly after active slot
    - delay-dependent balking
    - delay-dependent no-show
    - constant cancellation, applied only at end of day, and never same-day
    - rolling calendar full state
    - derived summary state at the start of each measured day

    Internal calendar representation:
        self.calendar[r][m] is either None (open slot) or Booking(i, tau, tracked)

    Public full-state view:
        0 or (i, tau)
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Rolling slot-level calendar Y_t(r, m), stored internally as None or Booking
        self.calendar: List[List[Optional[Booking]]] = [
            [None for _ in range(config.slots_per_day)]
            for _ in range(config.horizon_days)
        ]

        self.class_metrics: Dict[int, ClassMetrics] = {
            class_id: ClassMetrics() for class_id in config.classes
        }
        self.slot_metrics = SlotMetrics()
        self.total_value: float = 0.0

        # One summary state per measured day, recorded at the start of each measured day
        self.daily_summary_states: List[Dict[int, List[int]]] = []

    # -------------------------
    # State views
    # -------------------------

    def full_state_view(self) -> List[List[Union[int, Tuple[int, int]]]]:
        """
        Return Y_t(r, m) with cells shown exactly as:
            0
            (i, tau)
        """
        view: List[List[Union[int, Tuple[int, int]]]] = []
        for day_row in self.calendar:
            row_view: List[Union[int, Tuple[int, int]]] = []
            for cell in day_row:
                if cell is None:
                    row_view.append(0)
                else:
                    row_view.append((cell.patient_class, cell.booking_delay))
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
            for m in range(self.config.slots_per_day):
                cell = self.calendar[r][m]
                if cell is not None:
                    summary[cell.patient_class][r] += 1

        return summary

    # -------------------------
    # Booking logic
    # -------------------------

    def find_earliest_open_slot(self, active_slot: int) -> Optional[Tuple[int, int]]:
        """
        Find the earliest open slot strictly after the current active slot.

        Eligible slots:
        - same day, only m > active_slot
        - future days, any slot
        """
        # Same day: later slots only
        for m in range(active_slot + 1, self.config.slots_per_day):
            if self.calendar[0][m] is None:
                return (0, m)

        # Future days
        for r in range(1, self.config.horizon_days):
            for m in range(self.config.slots_per_day):
                if self.calendar[r][m] is None:
                    return (r, m)

        return None

    def process_one_arrival(
        self,
        class_id: int,
        active_slot: int,
        track_patient: bool,
    ) -> None:
        """
        Process one arriving patient from class i.
        Only measurement-window arrivals are tracked in class metrics.
        """
        params = self.config.classes[class_id]
        metrics = self.class_metrics[class_id]

        if track_patient:
            metrics.arrivals += 1

        offered = self.find_earliest_open_slot(active_slot)

        if offered is None:
            if track_patient:
                metrics.no_offer += 1
            return

        r, m = offered
        tau = r  # offered booking delay in days

        # Balking decision
        if self.rng.random() < params.balk_prob(tau):
            if track_patient:
                metrics.balked += 1
            return

        # Accept and book
        self.calendar[r][m] = Booking(
            patient_class=class_id,
            booking_delay=tau,
            tracked=track_patient,
        )

        if track_patient:
            metrics.booked += 1
            metrics.total_booking_delay += tau

    # -------------------------
    # Service logic
    # -------------------------

    # -------------------------
    # Service logic
    # -------------------------

    def serve_active_slot(self, active_slot: int, count_slot_metrics: bool) -> None:
        """
        Serve the active slot (r = 0, m = active_slot).

        Slot metrics and value are counted only during measured days.
        Class service/no-show metrics are counted only for tracked patients.
        """
        cell = self.calendar[0][active_slot]

        if cell is None:
            if count_slot_metrics:
                self.slot_metrics.empty_slots += 1
            return

        if count_slot_metrics:
            self.slot_metrics.booked_slots += 1

        class_id = cell.patient_class
        tau = cell.booking_delay
        params = self.config.classes[class_id]
        metrics = self.class_metrics[class_id]

        # No-show decision depends on original tau, not current residual delay
        if self.rng.random() < params.no_show_prob(tau):
            if cell.tracked:
                metrics.no_show += 1
            if count_slot_metrics:
                self.slot_metrics.no_show_slots += 1
        else:
            if cell.tracked:
                metrics.served += 1
            if count_slot_metrics:
                self.slot_metrics.served_slots += 1
                self.total_value += params.value

        # Active slot is consumed after service/no-show
        self.calendar[0][active_slot] = None

    # -------------------------
    # End-of-day logic
    # -------------------------

    def apply_end_of_day_cancellations(self) -> None:
        """
        Apply cancellations only to future appointments with r >= 1.
        Same-day cancellations are not allowed.

        All future bookings may cancel, but only tracked bookings count toward
        class-level cancellation metrics.
        """
        for r in range(1, self.config.horizon_days):
            for m in range(self.config.slots_per_day):
                cell = self.calendar[r][m]
                if cell is None:
                    continue

                class_id = cell.patient_class
                params = self.config.classes[class_id]

                if self.rng.random() < params.cancel_prob:
                    if cell.tracked:
                        self.class_metrics[class_id].canceled += 1
                    self.calendar[r][m] = None

    def roll_calendar_forward_one_day(self) -> None:
        """
        End of day transition:
        - drop day 0
        - shift future days forward by one
        - append a new empty day at the horizon end
        """
        self.calendar.pop(0)
        self.calendar.append([None for _ in range(self.config.slots_per_day)])

    # -------------------------
    # Slot arrivals
    # -------------------------

    def generate_ordered_individual_arrivals(self) -> List[int]:
        """
        Generate class-specific Poisson arrivals for the current slot,
        convert to individual patients, and randomize within-slot order.
        """
        arrivals: List[int] = []

        for class_id, params in self.config.classes.items():
            n = int(self.rng.poisson(params.lambda_per_slot))
            arrivals.extend([class_id] * n)

        if arrivals:
            arrivals = self.rng.permutation(arrivals).tolist()

        return arrivals

    # -------------------------
    # Main run
    # -------------------------

    def run(self) -> SimulationResults:
        """
        Run the slot-by-slot simulation with:
        - burn-in days
        - measurement days
        - cooldown days

        Class metrics track only arrivals from the measurement window.
        Slot metrics and total value count only service occurring on measured days.
        """
        total_days = (
            self.config.burn_in_days
            + self.config.measure_days
            + self.config.cooldown_days
        )

        first_measure_day = self.config.burn_in_days
        last_measure_day_exclusive = self.config.burn_in_days + self.config.measure_days

        for day in range(total_days):
            in_measurement_window = first_measure_day <= day < last_measure_day_exclusive

            # Record summary state only for measured days
            if in_measurement_window:
                start_of_day_summary = self.summary_state()
                self.daily_summary_states.append({
                    class_id: counts.copy()
                    for class_id, counts in start_of_day_summary.items()
                })

            # Process all S active slots in day D
            for s in range(self.config.slots_per_day):
                ordered_arrivals = self.generate_ordered_individual_arrivals()

                for class_id in ordered_arrivals:
                    self.process_one_arrival(
                        class_id=class_id,
                        active_slot=s,
                        track_patient=in_measurement_window,
                    )

                self.serve_active_slot(
                    active_slot=s,
                    count_slot_metrics=in_measurement_window,
                )

            # End-of-day cancellations for r >= 1 only
            self.apply_end_of_day_cancellations()

            # Move to next day
            self.roll_calendar_forward_one_day()

        return SimulationResults(
            class_metrics=self.class_metrics,
            slot_metrics=self.slot_metrics,
            total_slots=self.config.measure_days * self.config.slots_per_day,
            total_value=self.total_value,
            daily_summary_states=self.daily_summary_states,
            final_full_state=self.full_state_view(),
        )