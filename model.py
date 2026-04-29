from __future__ import annotations

from dataclasses import dataclass
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
    """
    class_id: int
    lambda_per_day: float
    balk_prob: ProbabilityFn
    cancel_prob: float
    no_show_prob: ProbabilityFn
    value: float = 1.0

    def __post_init__(self) -> None:
        if self.class_id <= 0:
            raise ValueError("class_id must be positive.")
        if self.lambda_per_day < 0:
            raise ValueError("Arrival rate must be nonnegative.")
        if not (0.0 <= self.cancel_prob <= 1.0):
            raise ValueError("Cancellation probability must lie in [0, 1].")


@dataclass(frozen=True)
class SimulationConfig:
    """
    Global simulation configuration.
    """
    slots_per_day: int
    horizon_days: int
    burn_in_days: int
    measure_days: int
    cooldown_days: int
    classes: Dict[int, PatientClassParams]
    seed: Optional[int] = None

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


# ==========================
# State and metrics objects
# ==========================

@dataclass
class Booking:
    """
    One booked appointment.

    booking_delay = tau = original offered booking delay in days
    patient_class = i
    tracked = whether the patient arrived during the measurement window
    """
    patient_class: int
    booking_delay: int
    tracked: bool


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
    total_booking_delay: float = 0.0

    @property
    def mean_booking_delay(self) -> float:
        return self.total_booking_delay / self.booked if self.booked > 0 else 0.0

    @property
    def percent_serviced(self) -> float:
        return self.served / self.arrivals if self.arrivals > 0 else 0.0

    def attended_utilization(self, total_slots: int) -> float:
        return self.served / total_slots if total_slots > 0 else 0.0


@dataclass
class SlotMetrics:
    booked_slots: int = 0
    served_slots: int = 0
    no_show_slots: int = 0
    empty_slots: int = 0


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

    @property
    def total_served(self) -> int:
        return sum(m.served for m in self.class_metrics.values())

    @property
    def overall_percent_serviced(self) -> float:
        total_arrivals = sum(m.arrivals for m in self.class_metrics.values())
        return self.total_served / total_arrivals if total_arrivals > 0 else 0.0

    @property
    def overall_attended_utilization(self) -> float:
        return self.total_served / self.total_slots if self.total_slots > 0 else 0.0

    @property
    def overall_scheduled_utilization(self) -> float:
        return self.slot_metrics.booked_slots / self.total_slots if self.total_slots > 0 else 0.0
