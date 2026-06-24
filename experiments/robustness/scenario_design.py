"""Generate and validate the Stage 1 robustness scenario bank.

The generator creates:

* 32 deterministic anchors;
* 224 unique scrambled-Sobol symmetric scenarios;
* 128 sparse asymmetric stress scenarios;
* fixed Stage 1 and Stage 2 seed lists; and
* validation and design-summary reports.

Run from the repository root:

    python experiments/robustness/scenario_design.py

No clinic simulations are run by this script.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from experiments.robustness.scenario_space import (  # noqa: E402
    ASYMMETRY_DIMENSIONS,
    ASYMMETRY_SEED,
    BALK_HIGH_VALUES,
    BALK_LOW_VALUES,
    BALK_THRESHOLD_VALUES,
    CANCEL_VALUES,
    CAPACITY_VALUES,
    CLASS1_SHARE_VALUES,
    HORIZON_VALUES,
    N_ANCHORS,
    N_ASYMMETRIC_STRESS,
    N_SOBOL_SYMMETRIC,
    NOSHOW_HIGH_VALUES,
    NOSHOW_LOW_VALUES,
    NOSHOW_THRESHOLD_VALUES,
    OUTPUT_COLUMNS,
    PARAMETER_COLUMNS,
    RHO_VALUES,
    SOBOL_SEED,
    STAGE1_SEEDS,
    STAGE2_SEEDS,
    VALID_BALK_PAIRS,
    VALID_NOSHOW_PAIRS,
)

DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "robustness" / "scenarios"
FLOAT_TOL = 1e-10


def _choice_from_unit(u: float, values: Sequence[Any]) -> Any:
    """Map a unit-interval value to one element of a discrete sequence."""
    if not values:
        raise ValueError("Cannot choose from an empty sequence.")
    index = min(int(float(u) * len(values)), len(values) - 1)
    return values[index]


def valid_thresholds(horizon: int, values: Sequence[int]) -> tuple[int, ...]:
    """Return threshold values that activate before the last offered delay."""
    return tuple(int(tau) for tau in values if int(tau) < int(horizon) - 1)


def _scenario_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return a stable parameter-only signature used for deduplication."""
    signature: list[Any] = []
    for column in PARAMETER_COLUMNS:
        value = row[column]
        if isinstance(value, (float, np.floating)):
            signature.append(round(float(value), 12))
        elif isinstance(value, (int, np.integer)):
            signature.append(int(value))
        else:
            signature.append(value)
    return tuple(signature)


def _make_symmetric_row(
    *,
    rho: float,
    class1_share: float,
    slots_per_day: int,
    horizon: int,
    cancel: float,
    balk_threshold: int,
    balk_pair: tuple[float, float],
    noshow_threshold: int,
    noshow_pair: tuple[float, float],
    scenario_type: str,
    design_note: str,
) -> dict[str, Any]:
    """Build one valid symmetric scenario row."""
    if balk_threshold >= horizon - 1:
        raise ValueError(
            f"Invalid balk threshold {balk_threshold} for horizon {horizon}."
        )
    if noshow_threshold >= horizon - 1:
        raise ValueError(
            f"Invalid no-show threshold {noshow_threshold} for horizon {horizon}."
        )
    balk_low, balk_high = balk_pair
    noshow_low, noshow_high = noshow_pair
    if balk_high < balk_low:
        raise ValueError("balk_high must be at least balk_low.")
    if noshow_high < noshow_low:
        raise ValueError("noshow_high must be at least noshow_low.")

    lambda_total = float(rho) * int(slots_per_day)
    lambda_class1 = float(class1_share) * lambda_total
    lambda_class2 = (1.0 - float(class1_share)) * lambda_total

    return {
        "scenario_id": "",
        "scenario_type": scenario_type,
        "parent_scenario_id": "",
        "design_note": design_note,
        "rho": float(rho),
        "class1_share": float(class1_share),
        "slots_per_day": int(slots_per_day),
        "lambda_total": lambda_total,
        "lambda_class1": lambda_class1,
        "lambda_class2": lambda_class2,
        "horizon_class1": int(horizon),
        "horizon_class2": int(horizon),
        "cancel_class1": float(cancel),
        "cancel_class2": float(cancel),
        "balk_threshold_class1": int(balk_threshold),
        "balk_threshold_class2": int(balk_threshold),
        "balk_low_class1": float(balk_low),
        "balk_low_class2": float(balk_low),
        "balk_high_class1": float(balk_high),
        "balk_high_class2": float(balk_high),
        "noshow_threshold_class1": int(noshow_threshold),
        "noshow_threshold_class2": int(noshow_threshold),
        "noshow_low_class1": float(noshow_low),
        "noshow_low_class2": float(noshow_low),
        "noshow_high_class1": float(noshow_high),
        "noshow_high_class2": float(noshow_high),
        "asymmetric_dimensions": "",
    }


def build_anchor_scenarios() -> list[dict[str, Any]]:
    """Create 32 deterministic anchor scenarios covering key regimes."""
    rows: list[dict[str, Any]] = []

    # First 15 anchors guarantee every rho × capacity combination.
    share_cycle = list(CLASS1_SHARE_VALUES)
    for index, (rho, capacity) in enumerate(
        (pair for rho in RHO_VALUES for pair in ((rho, c) for c in CAPACITY_VALUES))
    ):
        share = share_cycle[index % len(share_cycle)]
        rows.append(
            _make_symmetric_row(
                rho=rho,
                class1_share=share,
                slots_per_day=capacity,
                horizon=14,
                cancel=0.1,
                balk_threshold=9,
                balk_pair=(0.0, 0.5),
                noshow_threshold=6,
                noshow_pair=(0.0, 0.3),
                scenario_type="anchor",
                design_note=f"rho_capacity_grid_rho_{rho}_capacity_{capacity}",
            )
        )

    special_specs = [
        # Coverage anchors: jointly cover all requested levels.
        (0.8, 0.5, 32, 7, 0.0, 4, (0.0, 0.0), 4, (0.0, 0.0), "short_horizon_zero_risk"),
        (1.25, 0.3, 64, 21, 0.3, 6, (0.1, 0.1), 9, (0.0, 0.1), "long_horizon_low_step"),
        (2.0, 0.7, 16, 28, 0.5, 12, (0.3, 0.3), 12, (0.3, 0.3), "longest_horizon_mid_risk"),
        (3.1, 0.5, 32, 14, 0.0, 4, (0.5, 0.7), 4, (0.5, 0.5), "high_balk_low_threshold"),
        (4.0, 0.5, 32, 14, 0.3, 6, (0.7, 0.7), 6, (0.7, 0.7), "maximum_common_risk"),
        # Regime and interaction anchors.
        (0.8, 0.1, 64, 28, 0.1, 12, (0.0, 0.1), 12, (0.0, 0.1), "low_pressure_long_horizon"),
        (4.0, 0.9, 16, 7, 0.5, 4, (0.5, 0.7), 4, (0.5, 0.7), "high_pressure_short_horizon"),
        (3.1, 0.1, 32, 21, 0.3, 9, (0.3, 0.7), 6, (0.3, 0.7), "class1_minority_high_risk"),
        (3.1, 0.9, 32, 21, 0.3, 9, (0.3, 0.7), 6, (0.3, 0.7), "class1_majority_high_risk"),
        (4.0, 0.5, 32, 28, 0.1, 4, (0.0, 0.7), 4, (0.0, 0.7), "high_post_threshold_exposure"),
        (2.0, 0.5, 32, 14, 0.1, 12, (0.0, 0.7), 12, (0.0, 0.7), "low_post_threshold_exposure"),
        (1.25, 0.5, 16, 21, 0.0, 6, (0.0, 0.3), 9, (0.0, 0.3), "small_scale_balanced"),
        (1.25, 0.5, 64, 21, 0.0, 6, (0.0, 0.3), 9, (0.0, 0.3), "large_scale_balanced"),
        (3.1, 0.3, 32, 14, 0.5, 9, (0.0, 0.5), 6, (0.0, 0.0), "high_cancel_low_noshow"),
        (3.1, 0.7, 32, 14, 0.0, 9, (0.0, 0.5), 6, (0.5, 0.7), "low_cancel_high_noshow"),
        (3.1, 0.5, 32, 14, 0.1, 6, (0.3, 0.7), 9, (0.0, 0.1), "high_balk_low_noshow"),
        (3.1, 0.5, 32, 14, 0.1, 6, (0.0, 0.1), 9, (0.3, 0.7), "low_balk_high_noshow"),
    ]

    for (
        rho,
        share,
        capacity,
        horizon,
        cancel,
        balk_threshold,
        balk_pair,
        noshow_threshold,
        noshow_pair,
        note,
    ) in special_specs:
        rows.append(
            _make_symmetric_row(
                rho=rho,
                class1_share=share,
                slots_per_day=capacity,
                horizon=horizon,
                cancel=cancel,
                balk_threshold=balk_threshold,
                balk_pair=balk_pair,
                noshow_threshold=noshow_threshold,
                noshow_pair=noshow_pair,
                scenario_type="anchor",
                design_note=note,
            )
        )

    if len(rows) != N_ANCHORS:
        raise RuntimeError(f"Expected {N_ANCHORS} anchors; generated {len(rows)}.")

    seen: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows, start=1):
        signature = _scenario_signature(row)
        if signature in seen:
            raise RuntimeError(f"Duplicate deterministic anchor at position {index}.")
        seen.add(signature)
        row["scenario_id"] = f"A{index:03d}"
    return rows


def _sobol_row(point: Sequence[float]) -> dict[str, Any]:
    """Map one nine-dimensional Sobol point to a valid symmetric scenario."""
    if len(point) != 9:
        raise ValueError("Sobol points must contain nine coordinates.")

    rho = _choice_from_unit(point[0], RHO_VALUES)
    share = _choice_from_unit(point[1], CLASS1_SHARE_VALUES)
    capacity = _choice_from_unit(point[2], CAPACITY_VALUES)
    horizon = _choice_from_unit(point[3], HORIZON_VALUES)
    cancel = _choice_from_unit(point[4], CANCEL_VALUES)
    balk_threshold = _choice_from_unit(
        point[5], valid_thresholds(horizon, BALK_THRESHOLD_VALUES)
    )
    balk_pair = _choice_from_unit(point[6], VALID_BALK_PAIRS)
    noshow_threshold = _choice_from_unit(
        point[7], valid_thresholds(horizon, NOSHOW_THRESHOLD_VALUES)
    )
    noshow_pair = _choice_from_unit(point[8], VALID_NOSHOW_PAIRS)

    return _make_symmetric_row(
        rho=rho,
        class1_share=share,
        slots_per_day=capacity,
        horizon=horizon,
        cancel=cancel,
        balk_threshold=balk_threshold,
        balk_pair=balk_pair,
        noshow_threshold=noshow_threshold,
        noshow_pair=noshow_pair,
        scenario_type="sobol_symmetric",
        design_note="scrambled_sobol",
    )


def build_sobol_symmetric_scenarios(
    anchors: Sequence[Mapping[str, Any]],
    n: int = N_SOBOL_SYMMETRIC,
    seed: int = SOBOL_SEED,
) -> list[dict[str, Any]]:
    """Generate unique symmetric scenarios with a scrambled Sobol design."""
    existing = {_scenario_signature(row) for row in anchors}
    rows: list[dict[str, Any]] = []

    # 4096 candidate points are ample after mapping to this discrete space.
    sampler = qmc.Sobol(d=9, scramble=True, seed=seed)
    points = sampler.random_base2(m=12)

    for point in points:
        row = _sobol_row(point)
        signature = _scenario_signature(row)
        if signature in existing:
            continue
        existing.add(signature)
        row["scenario_id"] = f"S{len(rows) + 1:03d}"
        rows.append(row)
        if len(rows) == n:
            break

    if len(rows) != n:
        raise RuntimeError(
            f"Only generated {len(rows)} unique Sobol scenarios; expected {n}."
        )
    return rows


def _different_choice(
    rng: np.random.Generator,
    values: Sequence[Any],
    current: Any,
) -> Any | None:
    alternatives = [value for value in values if value != current]
    if not alternatives:
        return None
    return alternatives[int(rng.integers(0, len(alternatives)))]


def _mutate_dimension(
    row: dict[str, Any],
    *,
    class_id: int,
    dimension: str,
    rng: np.random.Generator,
) -> bool:
    """Mutate one class-specific dimension while preserving all constraints."""
    suffix = f"class{class_id}"

    if dimension == "horizon":
        current = int(row[f"horizon_{suffix}"])
        balk_tau = int(row[f"balk_threshold_{suffix}"])
        noshow_tau = int(row[f"noshow_threshold_{suffix}"])
        valid = [
            h
            for h in HORIZON_VALUES
            if h != current and balk_tau < h - 1 and noshow_tau < h - 1
        ]
        new_value = _different_choice(rng, valid, current)
        if new_value is None:
            return False
        row[f"horizon_{suffix}"] = int(new_value)
        return True

    if dimension == "cancel":
        key = f"cancel_{suffix}"
        new_value = _different_choice(rng, CANCEL_VALUES, float(row[key]))
        if new_value is None:
            return False
        row[key] = float(new_value)
        return True

    if dimension == "balk_threshold":
        key = f"balk_threshold_{suffix}"
        horizon = int(row[f"horizon_{suffix}"])
        valid = valid_thresholds(horizon, BALK_THRESHOLD_VALUES)
        new_value = _different_choice(rng, valid, int(row[key]))
        if new_value is None:
            return False
        row[key] = int(new_value)
        return True

    if dimension == "balk_low":
        key = f"balk_low_{suffix}"
        high = float(row[f"balk_high_{suffix}"])
        valid = [value for value in BALK_LOW_VALUES if value <= high]
        new_value = _different_choice(rng, valid, float(row[key]))
        if new_value is None:
            return False
        row[key] = float(new_value)
        return True

    if dimension == "balk_high":
        key = f"balk_high_{suffix}"
        low = float(row[f"balk_low_{suffix}"])
        valid = [value for value in BALK_HIGH_VALUES if value >= low]
        new_value = _different_choice(rng, valid, float(row[key]))
        if new_value is None:
            return False
        row[key] = float(new_value)
        return True

    if dimension == "noshow_threshold":
        key = f"noshow_threshold_{suffix}"
        horizon = int(row[f"horizon_{suffix}"])
        valid = valid_thresholds(horizon, NOSHOW_THRESHOLD_VALUES)
        new_value = _different_choice(rng, valid, int(row[key]))
        if new_value is None:
            return False
        row[key] = int(new_value)
        return True

    if dimension == "noshow_low":
        key = f"noshow_low_{suffix}"
        high = float(row[f"noshow_high_{suffix}"])
        valid = [value for value in NOSHOW_LOW_VALUES if value <= high]
        new_value = _different_choice(rng, valid, float(row[key]))
        if new_value is None:
            return False
        row[key] = float(new_value)
        return True

    if dimension == "noshow_high":
        key = f"noshow_high_{suffix}"
        low = float(row[f"noshow_low_{suffix}"])
        valid = [value for value in NOSHOW_HIGH_VALUES if value >= low]
        new_value = _different_choice(rng, valid, float(row[key]))
        if new_value is None:
            return False
        row[key] = float(new_value)
        return True

    raise ValueError(f"Unknown asymmetric dimension: {dimension}")


def _asymmetry_schedule(n: int, rng: np.random.Generator) -> list[int]:
    """Allocate approximately 50%, 35%, and 15% to one, two, or three changes."""
    n_one = int(round(0.50 * n))
    n_two = int(round(0.35 * n))
    n_three = n - n_one - n_two
    schedule = [1] * n_one + [2] * n_two + [3] * n_three
    rng.shuffle(schedule)
    return schedule


def build_asymmetric_stress_scenarios(
    symmetric_rows: Sequence[Mapping[str, Any]],
    n: int = N_ASYMMETRIC_STRESS,
    seed: int = ASYMMETRY_SEED,
) -> list[dict[str, Any]]:
    """Generate sparse, unique class-asymmetric stress scenarios."""
    rng = np.random.default_rng(seed)
    schedule = _asymmetry_schedule(n, rng)
    existing = {_scenario_signature(row) for row in symmetric_rows}
    rows: list[dict[str, Any]] = []
    max_attempts = 100_000
    attempts = 0

    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        target_changes = schedule[len(rows)]
        base = dict(symmetric_rows[int(rng.integers(0, len(symmetric_rows)))])
        row = dict(base)
        dimensions = list(
            rng.choice(ASYMMETRY_DIMENSIONS, size=target_changes, replace=False)
        )
        changed: list[str] = []
        success = True

        for dimension in dimensions:
            class_id = int(rng.choice((1, 2)))
            if not _mutate_dimension(
                row, class_id=class_id, dimension=str(dimension), rng=rng
            ):
                success = False
                break
            changed.append(f"class{class_id}:{dimension}")

        if not success:
            continue

        row["scenario_id"] = f"X{len(rows) + 1:03d}"
        row["scenario_type"] = "asymmetric_stress"
        row["parent_scenario_id"] = str(base["scenario_id"])
        row["design_note"] = f"sparse_asymmetry_{target_changes}_dimension"
        row["asymmetric_dimensions"] = ";".join(changed)

        signature = _scenario_signature(row)
        if signature in existing:
            continue
        existing.add(signature)
        rows.append(row)

    if len(rows) != n:
        raise RuntimeError(
            f"Only generated {len(rows)} asymmetric scenarios after "
            f"{attempts} attempts; expected {n}."
        )
    return rows


def _class_behavior_equal(row: Mapping[str, Any]) -> bool:
    prefixes = (
        "horizon",
        "cancel",
        "balk_threshold",
        "balk_low",
        "balk_high",
        "noshow_threshold",
        "noshow_low",
        "noshow_high",
    )
    return all(row[f"{prefix}_class1"] == row[f"{prefix}_class2"] for prefix in prefixes)


def _check_row_constraints(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for class_id in (1, 2):
        suffix = f"class{class_id}"
        horizon = int(row[f"horizon_{suffix}"])
        balk_tau = int(row[f"balk_threshold_{suffix}"])
        noshow_tau = int(row[f"noshow_threshold_{suffix}"])
        balk_low = float(row[f"balk_low_{suffix}"])
        balk_high = float(row[f"balk_high_{suffix}"])
        noshow_low = float(row[f"noshow_low_{suffix}"])
        noshow_high = float(row[f"noshow_high_{suffix}"])

        if balk_high < balk_low:
            errors.append(f"{suffix}: balk_high < balk_low")
        if noshow_high < noshow_low:
            errors.append(f"{suffix}: noshow_high < noshow_low")
        if balk_tau >= horizon - 1:
            errors.append(f"{suffix}: balk threshold is inactive for horizon")
        if noshow_tau >= horizon - 1:
            errors.append(f"{suffix}: no-show threshold is inactive for horizon")
        if not 0.0 <= float(row[f"cancel_{suffix}"]) <= 1.0:
            errors.append(f"{suffix}: cancellation probability is outside [0, 1]")

    lambda_sum = float(row["lambda_class1"]) + float(row["lambda_class2"])
    expected_total = float(row["rho"]) * int(row["slots_per_day"])
    if not math.isclose(lambda_sum, float(row["lambda_total"]), abs_tol=FLOAT_TOL):
        errors.append("class arrival rates do not sum to lambda_total")
    if not math.isclose(float(row["lambda_total"]), expected_total, abs_tol=FLOAT_TOL):
        errors.append("lambda_total != rho * slots_per_day")
    if not math.isclose(
        float(row["lambda_class1"]) / float(row["lambda_total"]),
        float(row["class1_share"]),
        abs_tol=FLOAT_TOL,
    ):
        errors.append("lambda_class1 share does not match class1_share")
    return errors


def validate_scenario_bank(
    symmetric_df: pd.DataFrame,
    asymmetric_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a row-per-check validation report."""
    all_df = pd.concat([symmetric_df, asymmetric_df], ignore_index=True)
    checks: list[dict[str, str]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    add(
        "symmetric_count",
        len(symmetric_df) == N_ANCHORS + N_SOBOL_SYMMETRIC,
        f"observed={len(symmetric_df)}, expected={N_ANCHORS + N_SOBOL_SYMMETRIC}",
    )
    add(
        "asymmetric_count",
        len(asymmetric_df) == N_ASYMMETRIC_STRESS,
        f"observed={len(asymmetric_df)}, expected={N_ASYMMETRIC_STRESS}",
    )
    add(
        "unique_scenario_ids",
        all_df["scenario_id"].is_unique,
        f"unique={all_df['scenario_id'].nunique()}, total={len(all_df)}",
    )
    duplicate_parameters = all_df.duplicated(subset=list(PARAMETER_COLUMNS)).sum()
    add(
        "unique_parameter_rows",
        duplicate_parameters == 0,
        f"duplicate_parameter_rows={duplicate_parameters}",
    )

    row_errors: list[str] = []
    for _, row in all_df.iterrows():
        for error in _check_row_constraints(row):
            row_errors.append(f"{row['scenario_id']}: {error}")
    add(
        "row_constraints",
        not row_errors,
        "all rows valid" if not row_errors else " | ".join(row_errors[:10]),
    )

    symmetric_behavior = symmetric_df.apply(_class_behavior_equal, axis=1)
    add(
        "symmetric_rows_are_symmetric",
        bool(symmetric_behavior.all()),
        f"non_symmetric_rows={(~symmetric_behavior).sum()}",
    )
    asymmetric_behavior = asymmetric_df.apply(_class_behavior_equal, axis=1)
    add(
        "stress_rows_are_asymmetric",
        bool((~asymmetric_behavior).all()),
        f"still_symmetric_rows={asymmetric_behavior.sum()}",
    )

    parent_ids = set(symmetric_df["scenario_id"])
    invalid_parents = sorted(
        set(asymmetric_df["parent_scenario_id"]) - parent_ids
    )
    add(
        "valid_parent_ids",
        not invalid_parents,
        "all parents valid" if not invalid_parents else str(invalid_parents),
    )

    coverage_specs: list[tuple[str, Iterable[Any], str]] = [
        ("rho", RHO_VALUES, "rho"),
        ("class1_share", CLASS1_SHARE_VALUES, "class1_share"),
        ("capacity", CAPACITY_VALUES, "slots_per_day"),
        ("horizon", HORIZON_VALUES, "horizon_class1"),
        ("cancellation", CANCEL_VALUES, "cancel_class1"),
        ("balk_threshold", BALK_THRESHOLD_VALUES, "balk_threshold_class1"),
        ("balk_low", BALK_LOW_VALUES, "balk_low_class1"),
        ("balk_high", BALK_HIGH_VALUES, "balk_high_class1"),
        ("noshow_threshold", NOSHOW_THRESHOLD_VALUES, "noshow_threshold_class1"),
        ("noshow_low", NOSHOW_LOW_VALUES, "noshow_low_class1"),
        ("noshow_high", NOSHOW_HIGH_VALUES, "noshow_high_class1"),
    ]
    for label, expected_values, column in coverage_specs:
        observed = set(symmetric_df[column].tolist())
        missing = [value for value in expected_values if value not in observed]
        add(
            f"coverage_{label}",
            not missing,
            "complete" if not missing else f"missing={missing}",
        )

    observed_rho_capacity = set(
        zip(symmetric_df["rho"], symmetric_df["slots_per_day"], strict=False)
    )
    expected_rho_capacity = {
        (rho, capacity) for rho in RHO_VALUES for capacity in CAPACITY_VALUES
    }
    missing_pairs = sorted(expected_rho_capacity - observed_rho_capacity)
    add(
        "rho_capacity_cross_coverage",
        not missing_pairs,
        "complete" if not missing_pairs else f"missing={missing_pairs}",
    )

    return pd.DataFrame(checks)


def _markdown_counts(df: pd.DataFrame, column: str) -> str:
    counts = df[column].value_counts(dropna=False).sort_index()
    lines = ["| Value | Count |", "|---:|---:|"]
    for value, count in counts.items():
        lines.append(f"| {value} | {int(count)} |")
    return "\n".join(lines)


def write_generation_summary(
    path: Path,
    symmetric_df: pd.DataFrame,
    asymmetric_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> None:
    """Write a concise Markdown summary of the generated bank."""
    all_df = pd.concat([symmetric_df, asymmetric_df], ignore_index=True)
    failed = validation_df[validation_df["status"] == "FAIL"]
    text = f"""# Stage 1 Scenario Generation Summary

## Counts

- Deterministic anchors: {int((symmetric_df['scenario_type'] == 'anchor').sum())}
- Sobol symmetric scenarios: {int((symmetric_df['scenario_type'] == 'sobol_symmetric').sum())}
- Total symmetric scenarios: {len(symmetric_df)}
- Asymmetric stress scenarios: {len(asymmetric_df)}
- Total Stage 1 background scenarios: {len(all_df)}
- Stage 1 paired seeds: {len(STAGE1_SEEDS)}
- Stage 2 confirmation seeds: {len(STAGE2_SEEDS)}

## Design constraints

- `balk_high >= balk_low`
- `noshow_high >= noshow_low`
- `threshold < horizon - 1` for both balking and no-show rules
- `lambda_total = rho * slots_per_day`
- `lambda_class1 = class1_share * lambda_total`
- `lambda_class2 = (1 - class1_share) * lambda_total`

## Scenario types

{_markdown_counts(all_df, 'scenario_type')}

## Demand-to-capacity ratio

{_markdown_counts(all_df, 'rho')}

## Class 1 arrival share

{_markdown_counts(all_df, 'class1_share')}

## Daily capacity

{_markdown_counts(all_df, 'slots_per_day')}

## Booking horizon: Class 1

{_markdown_counts(all_df, 'horizon_class1')}

## Validation

- Passed checks: {int((validation_df['status'] == 'PASS').sum())}
- Failed checks: {len(failed)}

"""
    if not failed.empty:
        text += "### Failed checks\n\n"
        for _, row in failed.iterrows():
            text += f"- **{row['check']}**: {row['detail']}\n"
    else:
        text += "All validation checks passed.\n"
    path.write_text(text, encoding="utf-8")


def generate_scenario_bank(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate, validate, and write the complete Stage 1 scenario bank."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anchors = build_anchor_scenarios()
    sobol_rows = build_sobol_symmetric_scenarios(anchors)
    symmetric_rows = [*anchors, *sobol_rows]
    asymmetric_rows = build_asymmetric_stress_scenarios(symmetric_rows)

    symmetric_df = pd.DataFrame(symmetric_rows, columns=OUTPUT_COLUMNS)
    asymmetric_df = pd.DataFrame(asymmetric_rows, columns=OUTPUT_COLUMNS)
    all_df = pd.concat([symmetric_df, asymmetric_df], ignore_index=True)
    validation_df = validate_scenario_bank(symmetric_df, asymmetric_df)

    symmetric_df.to_csv(output_dir / "symmetric_scenarios.csv", index=False)
    asymmetric_df.to_csv(output_dir / "asymmetric_scenarios.csv", index=False)
    all_df.to_csv(output_dir / "all_stage1_scenarios.csv", index=False)
    validation_df.to_csv(output_dir / "scenario_validation.csv", index=False)
    pd.DataFrame({"seed": STAGE1_SEEDS}).to_csv(
        output_dir / "stage1_seeds.csv", index=False
    )
    pd.DataFrame({"seed": STAGE2_SEEDS}).to_csv(
        output_dir / "stage2_seeds.csv", index=False
    )
    write_generation_summary(
        output_dir / "scenario_generation_summary.md",
        symmetric_df,
        asymmetric_df,
        validation_df,
    )

    failed = validation_df[validation_df["status"] == "FAIL"]
    if not failed.empty:
        failed_text = "; ".join(
            f"{row['check']}: {row['detail']}" for _, row in failed.iterrows()
        )
        raise RuntimeError(f"Scenario-bank validation failed: {failed_text}")

    return symmetric_df, asymmetric_df, all_df, validation_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Stage 1 robustness scenario bank."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symmetric_df, asymmetric_df, all_df, validation_df = generate_scenario_bank(
        args.output_dir
    )
    print("Stage 1 robustness scenario bank generated successfully.")
    print(f"  Symmetric scenarios: {len(symmetric_df)}")
    print(f"  Asymmetric scenarios: {len(asymmetric_df)}")
    print(f"  Total scenarios: {len(all_df)}")
    print(f"  Validation checks passed: {(validation_df['status'] == 'PASS').sum()}")
    print(f"  Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
