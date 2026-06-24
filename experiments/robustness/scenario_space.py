"""Shared Stage 1 robustness-scenario definitions.

This module contains only design constants. It does not run simulations.
"""

from __future__ import annotations

from itertools import product
from typing import Final

RHO_VALUES: Final[tuple[float, ...]] = (0.8, 1.25, 2.0, 3.1, 4.0)
CLASS1_SHARE_VALUES: Final[tuple[float, ...]] = (0.1, 0.3, 0.5, 0.7, 0.9)
CAPACITY_VALUES: Final[tuple[int, ...]] = (16, 32, 64)
HORIZON_VALUES: Final[tuple[int, ...]] = (7, 14, 21, 28)
CANCEL_VALUES: Final[tuple[float, ...]] = (0.0, 0.1, 0.3, 0.5)
BALK_THRESHOLD_VALUES: Final[tuple[int, ...]] = (4, 6, 9, 12)
BALK_LOW_VALUES: Final[tuple[float, ...]] = (0.0, 0.1, 0.3, 0.5, 0.7)
BALK_HIGH_VALUES: Final[tuple[float, ...]] = (0.0, 0.1, 0.3, 0.5, 0.7)
NOSHOW_THRESHOLD_VALUES: Final[tuple[int, ...]] = (4, 6, 9, 12)
NOSHOW_LOW_VALUES: Final[tuple[float, ...]] = (0.0, 0.3, 0.5, 0.7)
NOSHOW_HIGH_VALUES: Final[tuple[float, ...]] = (0.0, 0.1, 0.3, 0.5, 0.7)

VALID_BALK_PAIRS: Final[tuple[tuple[float, float], ...]] = tuple(
    (low, high)
    for low, high in product(BALK_LOW_VALUES, BALK_HIGH_VALUES)
    if high >= low
)

VALID_NOSHOW_PAIRS: Final[tuple[tuple[float, float], ...]] = tuple(
    (low, high)
    for low, high in product(NOSHOW_LOW_VALUES, NOSHOW_HIGH_VALUES)
    if high >= low
)

N_ANCHORS: Final[int] = 32
N_SOBOL_SYMMETRIC: Final[int] = 224
N_ASYMMETRIC_STRESS: Final[int] = 128

SOBOL_SEED: Final[int] = 2026
ASYMMETRY_SEED: Final[int] = 2027
STAGE1_SEEDS: Final[tuple[int, ...]] = tuple(range(1000, 1020))
STAGE2_SEEDS: Final[tuple[int, ...]] = tuple(range(2000, 2100))

ASYMMETRY_DIMENSIONS: Final[tuple[str, ...]] = (
    "horizon",
    "cancel",
    "balk_threshold",
    "balk_low",
    "balk_high",
    "noshow_threshold",
    "noshow_low",
    "noshow_high",
)

PARAMETER_COLUMNS: Final[tuple[str, ...]] = (
    "rho",
    "class1_share",
    "slots_per_day",
    "lambda_total",
    "lambda_class1",
    "lambda_class2",
    "horizon_class1",
    "horizon_class2",
    "cancel_class1",
    "cancel_class2",
    "balk_threshold_class1",
    "balk_threshold_class2",
    "balk_low_class1",
    "balk_low_class2",
    "balk_high_class1",
    "balk_high_class2",
    "noshow_threshold_class1",
    "noshow_threshold_class2",
    "noshow_low_class1",
    "noshow_low_class2",
    "noshow_high_class1",
    "noshow_high_class2",
)

OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "scenario_type",
    "parent_scenario_id",
    "design_note",
    *PARAMETER_COLUMNS,
    "asymmetric_dimensions",
)
