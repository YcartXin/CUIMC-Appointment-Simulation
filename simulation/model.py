from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union


# =========================
# Probability rule objects
# =========================

@dataclass(frozen=True)
class ThresholdRule:
    """
    Threshold probability rule:
        p(tau) = low  if tau <= threshold
                 high otherwise
    """
    threshold: int
    low: float
    high: float

    def __post_init__(self) -> None:
        for x in (self.low, self.high):
            if not (0.0 <= x <= 1.0):
                raise ValueError("Probabilities must lie in [0, 1].")
        if self.threshold < 0:
            raise ValueError("Threshold must be nonnegative.")

    def __call__(self, tau: int) -> float:
        return self.low if tau <= self.threshold else self.high


# =========================
# Model parameter objects
# =========================

ProbabilityFn = Callable[[int], float]


@dataclass(frozen=True)
class PatientClassParams:
    """
    Parameters for one patient class i.

    horizon_days: per-class maximum booking window.  When None the class
    uses the global SimulationConfig.horizon_days (backward-compatible).

    standby_prob: when a class-i patient would balk (the balk draw fires),
    this is the probability that, instead of exiting immediately, the
    patient joins the class's standby queue and waits for a possible
    earlier opening. Default 0.0 reproduces the original immediate-exit
    balking behavior exactly, including RNG-stream compatibility (the
    engine short-circuits and never draws for standby when this is 0.0).

    max_standby_days: how many additional days a standby patient waits
    before giving up and exiting permanently. None means no patience cap
    (the patient waits until the horizon closes out during cooldown).
    Ignored when standby_prob is 0.0.

    standby_eligible_after_days: how many days a standby patient must
    have already waited before the engine will offer them a recalled
    day at all. None (default) means eligible immediately, the original
    behavior. A nonzero value models a clinic that does not attempt to
    backfill from the standby list right away.
    """
    class_id: int
    lambda_per_day: float
    balk_prob: ProbabilityFn
    cancel_prob: float
    no_show_prob: ProbabilityFn
    value: float = 1.0
    horizon_days: Optional[int] = None
    standby_prob: float = 0.0
    max_standby_days: Optional[int] = None
    standby_eligible_after_days: Optional[int] = None

    def __post_init__(self) -> None:
        if self.class_id <= 0:
            raise ValueError("class_id must be positive.")
        if self.lambda_per_day < 0:
            raise ValueError("Arrival rate must be nonnegative.")
        if not (0.0 <= self.cancel_prob <= 1.0):
            raise ValueError("Cancellation probability must lie in [0, 1].")
        if self.horizon_days is not None and self.horizon_days <= 0:
            raise ValueError("Per-class horizon_days must be positive.")
        if not (0.0 <= self.standby_prob <= 1.0):
            raise ValueError("standby_prob must lie in [0, 1].")
        if self.max_standby_days is not None and self.max_standby_days <= 0:
            raise ValueError("max_standby_days must be positive when set.")
        if (
            self.standby_eligible_after_days is not None
            and self.standby_eligible_after_days < 0
        ):
            raise ValueError("standby_eligible_after_days must be nonnegative when set.")


@dataclass(frozen=True)
class SimulationConfig:
    """
    Global simulation configuration.

    reserved_window_days: when reservation is configured
    (reserved_slots_per_day > 0), restricts the reservation to residual
    days r < reserved_window_days. Days at or beyond the window behave as
    plain pooled FCFS for every class, including the reserved class. None
    means the reservation applies across the full horizon (the original,
    backward-compatible behavior). A window covering the full horizon and
    a window of None are equivalent.
    """
    slots_per_day: int
    horizon_days: int
    burn_in_days: int
    measure_days: int
    cooldown_days: int
    classes: Dict[int, PatientClassParams]
    seed: Optional[int] = None
    reserved_class_id: Optional[int] = None
    reserved_slots_per_day: int = 0
    reserved_window_days: Optional[int] = None

    def __post_init__(self) -> None:
        if self.slots_per_day <= 0:
            raise ValueError("slots_per_day must be positive.")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive.")
        if self.burn_in_days < 0:
            raise ValueError("burn_in_days must be nonnegative.")
        if self.measure_days <= 0:
            raise ValueError("measure_days must be positive.")
        if self.cooldown_days < 0:
            raise ValueError("cooldown_days must be nonnegative.")
        if not self.classes:
            raise ValueError("At least one patient class is required.")
        if self.reserved_slots_per_day < 0:
            raise ValueError("reserved_slots_per_day must be nonnegative.")
        if self.reserved_slots_per_day > self.slots_per_day:
            raise ValueError("reserved_slots_per_day cannot exceed slots_per_day.")
        if self.reserved_slots_per_day > 0:
            if self.reserved_class_id is None:
                raise ValueError("reserved_class_id is required when slots are reserved.")
            if self.reserved_class_id not in self.classes:
                raise ValueError("reserved_class_id must identify a configured class.")
        if self.reserved_window_days is not None and self.reserved_window_days < 0:
            raise ValueError("reserved_window_days must be nonnegative when set.")


# ==========================
# State and metrics objects
# ==========================

@dataclass
class Booking:
    """
    One booked appointment.

    booking_delay = tau = original offered booking delay in days, or (for
    a standby-recalled patient) the delay of the recalled day, which is
    what actually governs their no-show probability.
    patient_class = i
    tracked = whether the patient arrived during the measurement window
    standby_recalled = True if this booking came from the standby queue
    rather than a direct offer.
    """
    patient_class: int
    booking_delay: int
    tracked: bool
    reserved_slot: bool = False
    standby_recalled: bool = False


@dataclass
class StandbyEntry:
    """
    One patient waiting off-calendar for an earlier opening after
    declining an offer whose delay exceeded what they were willing to
    accept outright.

    original_offered_delay: tau of the offer they declined. A later
    opening is only offered to this entry when its residual day r is
    strictly smaller than this value, so a recall is always a strict
    improvement over what the patient already turned down.
    days_waited: number of days this entry has been on standby, aged by
    one once per simulated day. Compared against the class's
    max_standby_days to decide expiry.
    tracked: whether the original arrival happened during the
    measurement window.
    """
    patient_class: int
    original_offered_delay: int
    days_waited: int = 0
    tracked: bool = False


@dataclass
class ClassMetrics:
    """
    Metrics tracked for one patient class.
    """
    arrivals: int = 0
    booked: int = 0
    balked: int = 0
    no_offer: int = 0
    canceled: int = 0
    no_show: int = 0
    served: int = 0

    # Sum of delays only for patients who accepted/booked an offered slot.
    # For standby-recalled patients this uses the recalled (final) delay.
    total_booking_delay: float = 0.0

    # Count of accepted/booked patients by original booking delay tau
    accepted_delay_counts: Dict[int, int] = field(default_factory=dict)

    # Sum of delays for all patients who received an offer including balked
    total_offered_booking_delay: float = 0.0

    # --- Standby/requeue diagnostics (Hypothesis 2) ---
    # Joining the standby queue is NOT counted as balked, and the far-out
    # offer that triggered it is not counted toward offered_delay either.
    # The decision is deferred: if the entry is later recalled into an
    # earlier day, that day is counted as its real offer (booked,
    # contributing to offered = booked + balked as usual). If the entry
    # expires without being recalled, it is counted as no_offer instead,
    # exactly as if the system never had a slot for it. standby_joined,
    # standby_recalled, and standby_expired are diagnostic-only counts
    # layered on top of that accounting; they do not need to sum to
    # anything in particular against booked/balked/no_offer.
    standby_joined: int = 0
    standby_recalled: int = 0
    standby_expired: int = 0
    total_standby_wait_days: float = 0.0
    total_original_offered_delay_recalled: float = 0.0

    @property
    def offered(self) -> int:
        return self.booked + self.balked

    @property
    def mean_accepted_booking_delay(self) -> float:
        return self.total_booking_delay / self.booked if self.booked > 0 else 0.0

    @property
    def mean_offered_booking_delay(self) -> float:
        return (
            self.total_offered_booking_delay / self.offered
            if self.offered > 0
            else 0.0
        )

    @property
    def mean_booking_delay(self) -> float:
        """
        Backward-compatible alias for accepted booking delay.
        """
        return self.mean_accepted_booking_delay

    @property
    def percent_serviced(self) -> float:
        return self.served / self.arrivals if self.arrivals > 0 else 0.0

    @property
    def standby_recall_rate(self) -> float:
        return (
            self.standby_recalled / self.standby_joined
            if self.standby_joined > 0
            else 0.0
        )

    @property
    def mean_standby_wait_days(self) -> float:
        return (
            self.total_standby_wait_days / self.standby_recalled
            if self.standby_recalled > 0
            else 0.0
        )

    @property
    def mean_original_offered_delay_recalled(self) -> float:
        """
        Average delay of the offer a recalled patient originally declined,
        i.e. how bad the rejected offer was, for contrast against the
        much shorter delay they were actually booked and served under.
        """
        return (
            self.total_original_offered_delay_recalled / self.standby_recalled
            if self.standby_recalled > 0
            else 0.0
        )


@dataclass
class SlotMetrics:
    booked_slots: int = 0
    served_slots: int = 0
    no_show_slots: int = 0

    # Reserved-capacity diagnostic (Hypothesis 1): how many measured
    # same-day slots that were booked used reserved capacity.
    reserved_slots_booked: int = 0

    daily_utilization_sum: float = 0.0
    measured_days: int = 0


@dataclass
class SimulationResults:
    """
    Final simulation outputs.
    """
    class_metrics: Dict[int, ClassMetrics]
    slot_metrics: SlotMetrics
    total_slots: int
    total_value: float
    daily_summary_states: List[Dict[int, List[int]]]
    final_full_state: List[List[Union[int, Tuple[int, int]]]]

    # Reserved slot-days available across the measured window, i.e.
    # reserved_slots_per_day * measure_days when a window >= 1 day was in
    # effect during measurement, else 0. Used as the denominator for
    # reserved_slot_fill_rate.
    reserved_slot_capacity: int = 0

    @property
    def total_served(self) -> int:
        return sum(m.served for m in self.class_metrics.values())

    @property
    def overall_percent_serviced(self) -> float:
        total_arrivals = sum(m.arrivals for m in self.class_metrics.values())
        return self.total_served / total_arrivals if total_arrivals > 0 else 0.0

    @property
    def average_utilization(self) -> float:
        return (
            self.slot_metrics.daily_utilization_sum / self.slot_metrics.measured_days
            if self.slot_metrics.measured_days > 0
            else 0.0
        )

    @property
    def booked_slot_utilization(self) -> float:
        return (
            self.slot_metrics.booked_slots / self.total_slots
            if self.total_slots > 0
            else 0.0
        )

    @property
    def reserved_slot_fill_rate(self) -> float:
        """
        Share of reserved capacity that was actually booked. Low values
        indicate the reserved class's demand isn't filling the near-term
        capacity set aside for it, i.e. the reservation is going empty
        rather than preventing no-shows.
        """
        return (
            self.slot_metrics.reserved_slots_booked / self.reserved_slot_capacity
            if self.reserved_slot_capacity > 0
            else 0.0
        )