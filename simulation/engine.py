from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .model import (
    Booking,
    ClassMetrics,
    SimulationConfig,
    SimulationResults,
    SlotMetrics,
    StandbyEntry,
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
    - optional windowed reservation: a class-specific slot reservation that
      only applies within a near-term residual-day window
    - optional standby/requeue: patients who would otherwise balk a
      far-out offer can instead wait off-calendar for an earlier opening
      freed by a cancellation

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

        # One standby queue per class, FIFO by join order.
        self.standby_queue: Dict[int, List[StandbyEntry]] = {
            class_id: [] for class_id in config.classes
        }

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

    def _class_horizon(self, class_id: int) -> int:
        """
        Return the effective booking horizon for a patient class.
        Uses the per-class horizon if set, otherwise the global horizon.
        The result is capped at the calendar size (config.horizon_days).
        """
        class_h = self.config.classes[class_id].horizon_days
        if class_h is None:
            return self.config.horizon_days
        return min(class_h, self.config.horizon_days)

    def _reservation_window(self) -> int:
        """
        Return the number of leading residual days (r = 0 .. window-1)
        over which the configured reservation applies. Defaults to the
        full calendar when no window is set, which reproduces the
        original whole-horizon strict-reservation behavior exactly.
        """
        window = self.config.reserved_window_days
        if window is None:
            return self.config.horizon_days
        return min(window, self.config.horizon_days)

    def _slot_offer_at(self, r: int, class_id: int) -> Optional[bool]:
        """
        Return whether class_id can be offered/booked into day r right
        now, and if so, whether that would consume reserved capacity.

        Returns:
            True  -> capacity available, and it is reserved capacity
            False -> capacity available, and it is general capacity
            None  -> no capacity available for class_id at day r

        This is shared by new-offer search (find_earliest_open_day) and
        standby recall (process_standby_recalls) so both respect the same
        capacity/reservation rules.
        """
        reserved_slots = self.config.reserved_slots_per_day
        reserved_class_id = self.config.reserved_class_id

        if reserved_slots == 0 or reserved_class_id is None or r >= self._reservation_window():
            if len(self.calendar[r]) < self.config.slots_per_day:
                return False
            return None

        general_slots = self.config.slots_per_day - reserved_slots
        reserved_used = sum(1 for b in self.calendar[r] if b.reserved_slot)
        general_used = len(self.calendar[r]) - reserved_used

        if class_id == reserved_class_id and reserved_used < reserved_slots:
            return True

        if general_used < general_slots:
            return False

        return None

    def find_earliest_open_day(self, class_id: int) -> Optional[Tuple[int, bool]]:
        """
        Find the earliest day with available capacity.

        Same-day booking is allowed, so the search starts at r = 0.
        The search is bounded by the class-specific horizon.
        """
        horizon = self._class_horizon(class_id)

        for r in range(horizon):
            offer = self._slot_offer_at(r, class_id)
            if offer is not None:
                return r, offer

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
        Process the full day's arrivals under the configured booking rule.
        """
        if track_patients:
            for class_id in ordered_arrivals:
                self.class_metrics[class_id].arrivals += 1

        def process_one(class_id: int) -> bool:
            params = self.config.classes[class_id]
            metrics = self.class_metrics[class_id]

            offer = self.find_earliest_open_day(class_id)

            if offer is None:
                if track_patients:
                    metrics.no_offer += 1
                return False

            offered_day, reserved_slot = offer

            tau = offered_day  # offered booking delay in days; tau = 0 is allowed

            # Balking decision
            if self.rng.random() < params.balk_prob(tau):
                # standby_prob defaults to 0.0, and the short-circuit below
                # skips the extra RNG draw entirely in that case, so
                # existing configs replay with an identical RNG stream.
                if params.standby_prob > 0.0 and self.rng.random() < params.standby_prob:
                    # This far-out offer is deliberately NOT counted as an
                    # offer yet, and the patient is NOT counted as balked.
                    # Joining the standby queue defers the decision: if a
                    # later day frees up and this entry is recalled, THAT
                    # day becomes their real offer (booked, counted in
                    # offered_delay/accepted_delay). If they expire
                    # unresolved, they are counted as no_offer instead,
                    # exactly as if the system never had a slot for them.
                    # See process_standby_recalls and age_standby_queue.
                    self.standby_queue[class_id].append(
                        StandbyEntry(
                            patient_class=class_id,
                            original_offered_delay=tau,
                            days_waited=0,
                            tracked=track_patients,
                        )
                    )
                    if track_patients:
                        metrics.standby_joined += 1
                    return True

                # True immediate balk: this offer is real and rejected for
                # good, so it counts toward offered_delay and balked.
                if track_patients:
                    metrics.total_offered_booking_delay += tau
                    metrics.balked += 1
                return True

            # Accept and book
            self.calendar[offered_day].append(
                Booking(
                    patient_class=class_id,
                    booking_delay=tau,
                    tracked=track_patients,
                    reserved_slot=reserved_slot,
                )
            )

            if track_patients:
                metrics.total_offered_booking_delay += tau
                metrics.booked += 1
                metrics.total_booking_delay += tau
                metrics.accepted_delay_counts[tau] = (
                    metrics.accepted_delay_counts.get(tau, 0) + 1
                )

            return True

        for class_id in ordered_arrivals:
            process_one(class_id)

    # -------------------------
    # Standby / requeue logic
    # -------------------------

    def process_standby_recalls(self) -> None:
        """
        Offer freshly opened capacity to the standby queue before new
        arrivals are processed for the day, nearest residual day first.

        Within a class, only the head of that class's queue is
        considered for a given day: FIFO by join order, not by which
        entry happens to have the smallest original offered delay. A
        recall is only made when the day is strictly better than what
        the entry originally declined (r < original_offered_delay), so a
        recalled patient is always moving to a shorter wait.

        Recall respects the same capacity/reservation split as a fresh
        offer via _slot_offer_at, so this coexists correctly with a
        windowed reservation. If a class sets
        standby_eligible_after_days, an entry that has not yet waited
        that many days is treated as ineligible today, same as if no
        day were open for it.
        """
        if all(params.standby_prob == 0.0 for params in self.config.classes.values()):
            return

        for r in range(self.config.horizon_days):
            for class_id, queue in self.standby_queue.items():
                eligible_after = self.config.classes[class_id].standby_eligible_after_days or 0
                while queue:
                    entry = queue[0]
                    if entry.days_waited < eligible_after:
                        # Not yet eligible for a recall at all; consistent
                        # with the FIFO simplification elsewhere, this
                        # blocks the rest of the class's queue for today
                        # rather than skipping ahead to a later entry.
                        break
                    if r >= entry.original_offered_delay:
                        break

                    offer = self._slot_offer_at(r, class_id)
                    if offer is None:
                        break

                    queue.pop(0)
                    self.calendar[r].append(
                        Booking(
                            patient_class=class_id,
                            booking_delay=r,
                            tracked=entry.tracked,
                            reserved_slot=offer,
                            standby_recalled=True,
                        )
                    )

                    if entry.tracked:
                        metrics = self.class_metrics[class_id]
                        metrics.standby_recalled += 1
                        # The recalled day is this patient's real offer:
                        # they were never counted as offered or balked
                        # when they joined the queue, so both the
                        # offered-delay and accepted-delay sums use the
                        # recalled (shorter) day here, not the original
                        # rejected one.
                        metrics.total_offered_booking_delay += r
                        metrics.booked += 1
                        metrics.total_booking_delay += r
                        metrics.accepted_delay_counts[r] = (
                            metrics.accepted_delay_counts.get(r, 0) + 1
                        )
                        metrics.total_standby_wait_days += entry.days_waited
                        metrics.total_original_offered_delay_recalled += (
                            entry.original_offered_delay
                        )

    def age_standby_queue(self) -> None:
        """
        Advance every remaining standby entry by one day. Entries that
        reach their class's max_standby_days without being recalled
        expire and leave the queue permanently. An expired entry was
        never counted as offered or balked, so it is counted as
        no_offer here: from the class metrics' point of view, this
        patient never received a usable slot, exactly like a patient
        who arrived when the booking horizon was already full.
        """
        for class_id, queue in self.standby_queue.items():
            if not queue:
                continue

            max_days = self.config.classes[class_id].max_standby_days
            remaining: List[StandbyEntry] = []

            for entry in queue:
                entry.days_waited += 1
                if max_days is not None and entry.days_waited >= max_days:
                    if entry.tracked:
                        metrics = self.class_metrics[class_id]
                        metrics.standby_expired += 1
                        metrics.no_offer += 1
                else:
                    remaining.append(entry)

            self.standby_queue[class_id] = remaining

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
            self.slot_metrics.reserved_slots_booked += sum(
                1 for b in todays_bookings if b.reserved_slot
            )

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
        2. offer newly freed capacity to the standby queue, nearest day first
        3. age the standby queue and expire patients past their patience cap
        4. record start-of-day summary state
        5. generate all daily arrivals
        6. randomly permute arrivals
        7. process offers/balking in FCFS order
        8. capture final calendar snapshot on the last simulated day
        9. resolve no-shows/service for today's scheduled patients
        10. roll the calendar forward

        Note on measurement semantics: cooldown must be long enough not
        only for late measurement-window bookings to resolve (as with
        plain FCFS), but also for standby queue entries to resolve, i.e.
        cooldown_days should be at least as large as the largest
        configured max_standby_days when standby is enabled. Otherwise
        some measured patients can be left on standby when the run ends,
        uncounted in booked, balked, or no_offer alike -- the same
        "unresolved" caveat that already applies to late plain bookings,
        just extended to the standby queue.
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

            # 2-3. Standby recall into newly freed capacity, then age the queue
            self.process_standby_recalls()
            self.age_standby_queue()

            # 4. Record start-of-day summary state after cancellations/recalls
            if in_measurement_window:
                start_of_day_summary = self.summary_state()
                self.daily_summary_states.append({
                    class_id: counts.copy()
                    for class_id, counts in start_of_day_summary.items()
                })

            # 5-7. Generate, permute, and process the day's arrivals
            ordered_arrivals = self.generate_daily_arrival_order()
            self.process_daily_arrivals(
                ordered_arrivals=ordered_arrivals,
                track_patients=in_measurement_window,
            )

            # 8. Capture the final calendar view before service and before rolling forward
            if day == total_days - 1:
                final_full_state_snapshot = self.full_state_view()

            # 9. Resolve today's scheduled appointments
            self.serve_today(count_slot_metrics=in_measurement_window)

            # 10. Move to next day
            self.roll_calendar_forward_one_day()

        reserved_slot_capacity = (
            self.config.reserved_slots_per_day * self.config.measure_days
            if self.config.reserved_slots_per_day > 0 and self._reservation_window() > 0
            else 0
        )

        return SimulationResults(
            class_metrics=self.class_metrics,
            slot_metrics=self.slot_metrics,
            total_slots=self.config.measure_days * self.config.slots_per_day,
            total_value=self.total_value,
            daily_summary_states=self.daily_summary_states,
            final_full_state=final_full_state_snapshot,
            reserved_slot_capacity=reserved_slot_capacity,
        )