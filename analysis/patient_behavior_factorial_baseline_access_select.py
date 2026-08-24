#!/usr/bin/env python3
"""Select baseline-referenced access-balanced policy cells under an avg-utilization floor.

Research estimand
-----------------
For each background b and policy family p:

1. Freeze the ORIGINAL average-utilization-optimal cell selected by the main
   patient-behavior-factorial search. Let its search-seed mean utilization be U*.
2. Restrict candidate policy cells to:
       U(c) >= U* - 0.005
3. Measure access relative to the matched NO-POLICY baseline:
       dSR1(c) = SR1(c) - SR1_baseline
       dSR2(c) = SR2(c) - SR2_baseline
4. Select the feasible cell maximizing:
       min(dSR1(c), dSR2(c))

Thus the utilization constraint is relative to the optimized policy, while
win-win / win-neutral access is relative to baseline.

This script DOES NOT overwrite the original selected_cells.csv or the previous
access-optimization outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


POLICIES = ("horizon_only", "reservation_only", "both_flexible")
CELL_COLS = ["horizon_days", "Q", "window"]
SEED_COL = "seed"

UTIL_TOL = 0.005
PRACTICAL_BAND = 0.005


def _dedupe_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per cell x seed.

    Identical configurations may appear in several refinement arms. Because the
    simulation is deterministic conditional on seed/config, any duplicate is
    equivalent. Prefer exact/original rows deterministically when possible.
    """
    if frame.empty:
        return frame.copy()

    out = frame.copy()

    arm_rank = {
        "exact": 0,
        "coarse": 1,
        "fine_average_utilization": 2,
        "fine_priority_weighted_utilization": 3,
        "fine_access_balance": 4,
        "fine_baseline_access": 5,
    }
    out["_arm_rank"] = out.get(
        "arm",
        pd.Series("", index=out.index),
    ).map(arm_rank).fillna(99)

    return (
        out.sort_values(
            CELL_COLS + [SEED_COL, "_arm_rank"],
            kind="stable",
        )
        .drop_duplicates(CELL_COLS + [SEED_COL], keep="first")
        .drop(columns="_arm_rank")
    )


def _policy_candidates(search: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Return valid cells for one policy family.

    Mirrors the main factorial policy definitions:
      horizon_only    = horizon-only rows
      reservation_only= reservation rows + baseline (Q=0)
      both_flexible   = both-flexible rows + horizon-only (Q=0)
    """
    baseline = search[search["stage"] == "baseline"].copy()
    horizon = search[search["stage"] == "horizon_only"].copy()

    if policy == "horizon_only":
        return _dedupe_cells(horizon)

    if policy == "reservation_only":
        return _dedupe_cells(
            pd.concat(
                [
                    search[search["stage"] == "reservation_only"],
                    baseline,
                ],
                ignore_index=True,
            )
        )

    if policy == "both_flexible":
        return _dedupe_cells(
            pd.concat(
                [
                    search[search["stage"] == "both_flexible"],
                    horizon,
                ],
                ignore_index=True,
            )
        )

    raise ValueError(f"Unknown policy: {policy}")


def _baseline_means(search: pd.DataFrame, expected_seeds: int) -> dict[str, float]:
    baseline = _dedupe_cells(
        search[search["stage"] == "baseline"].copy()
    )

    n = int(baseline[SEED_COL].nunique())
    if n != expected_seeds:
        raise ValueError(
            f"Baseline has {n} search seeds; expected {expected_seeds}."
        )

    return {
        "baseline_average_utilization": float(
            baseline["average_utilization"].mean()
        ),
        "baseline_class_1_percent_serviced": float(
            baseline["class_1_percent_serviced"].mean()
        ),
        "baseline_class_2_percent_serviced": float(
            baseline["class_2_percent_serviced"].mean()
        ),
    }


def _aggregate_candidates(
    candidates: pd.DataFrame,
    expected_seeds: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    grouped = (
        candidates.groupby(CELL_COLS, as_index=False)
        .agg(
            search_average_utilization=("average_utilization", "mean"),
            search_class_1_percent_serviced=(
                "class_1_percent_serviced",
                "mean",
            ),
            search_class_2_percent_serviced=(
                "class_2_percent_serviced",
                "mean",
            ),
            search_n_seeds=(SEED_COL, "nunique"),
        )
    )

    # Never let a partially completed refinement cell win selection.
    grouped = grouped[
        grouped["search_n_seeds"] == expected_seeds
    ].copy()

    return grouped


def _source_metadata(
    candidates: pd.DataFrame,
    h: int,
    q: int,
    w: int,
) -> tuple[str, str]:
    chosen = candidates[
        (candidates["horizon_days"] == h)
        & (candidates["Q"] == q)
        & (candidates["window"] == w)
    ]

    stage = (
        str(chosen["stage"].mode().iloc[0])
        if "stage" in chosen.columns and not chosen.empty
        else ""
    )
    arm = (
        str(chosen["arm"].mode().iloc[0])
        if "arm" in chosen.columns and not chosen.empty
        else ""
    )
    return stage, arm


def _search_point_classification(d1: float, d2: float) -> str:
    """Diagnostic only; final classification must use independent eval seeds."""
    if d1 >= PRACTICAL_BAND and d2 >= PRACTICAL_BAND:
        return "point_win_win"
    if d1 >= PRACTICAL_BAND and abs(d2) <= PRACTICAL_BAND:
        return "point_c1_win_c2_neutral"
    if d2 >= PRACTICAL_BAND and abs(d1) <= PRACTICAL_BAND:
        return "point_c2_win_c1_neutral"
    if abs(d1) <= PRACTICAL_BAND and abs(d2) <= PRACTICAL_BAND:
        return "point_both_neutral"
    if d1 >= PRACTICAL_BAND and d2 <= -PRACTICAL_BAND:
        return "point_c1_gain_c2_harm"
    if d2 >= PRACTICAL_BAND and d1 <= -PRACTICAL_BAND:
        return "point_c2_gain_c1_harm"
    return "point_other"


def select_one(
    *,
    search: pd.DataFrame,
    background_id: str,
    policy: str,
    ref: pd.Series,
) -> dict[str, object]:
    expected_seeds = int(ref["search_n_seeds"])
    baseline = _baseline_means(search, expected_seeds)

    candidates_raw = _policy_candidates(search, policy)
    candidates = _aggregate_candidates(
        candidates_raw,
        expected_seeds,
    )

    if candidates.empty:
        raise ValueError(
            f"{background_id} / {policy}: no complete candidate cells."
        )

    frozen_u_star = float(ref["search_objective_mean"])

    candidates["utilization_gap_from_avg_opt"] = (
        candidates["search_average_utilization"] - frozen_u_star
    )

    eligible = candidates[
        candidates["search_average_utilization"]
        >= frozen_u_star - UTIL_TOL
    ].copy()

    if eligible.empty:
        # The frozen avg-opt cell should itself always be feasible if it is
        # present and complete, so this signals inconsistent inputs.
        raise RuntimeError(
            f"{background_id} / {policy}: no cell is within "
            f"{UTIL_TOL:.3f} of frozen average-utilization optimum."
        )

    eligible["delta_sr1_vs_baseline"] = (
        eligible["search_class_1_percent_serviced"]
        - baseline["baseline_class_1_percent_serviced"]
    )
    eligible["delta_sr2_vs_baseline"] = (
        eligible["search_class_2_percent_serviced"]
        - baseline["baseline_class_2_percent_serviced"]
    )
    eligible["worst_access_gain_vs_baseline"] = np.minimum(
        eligible["delta_sr1_vs_baseline"],
        eligible["delta_sr2_vs_baseline"],
    )
    eligible["sum_access_gain_vs_baseline"] = (
        eligible["delta_sr1_vs_baseline"]
        + eligible["delta_sr2_vs_baseline"]
    )

    # Core objective: maximize the worse baseline-referenced class access gain.
    # Tie-break toward less utilization sacrifice, then greater total access
    # gain, then simpler/smaller policy parameters for deterministic selection.
    best = (
        eligible.sort_values(
            [
                "worst_access_gain_vs_baseline",
                "search_average_utilization",
                "sum_access_gain_vs_baseline",
                "Q",
                "window",
                "horizon_days",
            ],
            ascending=[False, False, False, True, True, True],
            kind="stable",
        )
        .iloc[0]
    )

    h = int(best["horizon_days"])
    q = int(best["Q"])
    w = int(best["window"])
    source_stage, source_arm = _source_metadata(
        candidates_raw, h, q, w
    )

    ref_h = int(ref["selected_horizon_days"])
    ref_q = int(ref["selected_Q"])
    ref_w = int(ref["selected_window"])

    d1 = float(best["delta_sr1_vs_baseline"])
    d2 = float(best["delta_sr2_vs_baseline"])

    return {
        "background_id": background_id,
        "policy": policy,
        "utilization_reference": "average_utilization_optimum",
        "access_reference": "matched_no_policy_baseline",
        "utilization_tolerance": UTIL_TOL,
        "selected_horizon_days": h,
        "selected_Q": q,
        "selected_window": w,
        "selected_source_stage": source_stage,
        "selected_source_arm": source_arm,
        "search_n_seeds": int(best["search_n_seeds"]),
        "search_average_utilization": float(
            best["search_average_utilization"]
        ),
        "frozen_avg_opt_average_utilization": frozen_u_star,
        "search_utilization_change_vs_avg_opt": float(
            best["utilization_gap_from_avg_opt"]
        ),
        "search_baseline_average_utilization": float(
            baseline["baseline_average_utilization"]
        ),
        "search_class_1_percent_serviced": float(
            best["search_class_1_percent_serviced"]
        ),
        "search_class_2_percent_serviced": float(
            best["search_class_2_percent_serviced"]
        ),
        "search_baseline_class_1_percent_serviced": float(
            baseline["baseline_class_1_percent_serviced"]
        ),
        "search_baseline_class_2_percent_serviced": float(
            baseline["baseline_class_2_percent_serviced"]
        ),
        "search_delta_sr1_vs_baseline": d1,
        "search_delta_sr2_vs_baseline": d2,
        "search_worst_access_gain_vs_baseline": float(
            best["worst_access_gain_vs_baseline"]
        ),
        "search_sum_access_gain_vs_baseline": float(
            best["sum_access_gain_vs_baseline"]
        ),
        "search_point_classification": _search_point_classification(d1, d2),
        "avg_opt_horizon_days": ref_h,
        "avg_opt_Q": ref_q,
        "avg_opt_window": ref_w,
        "same_as_avg_utilization_optimum": bool(
            h == ref_h and q == ref_q and w == ref_w
        ),
        "n_eligible_cells": int(len(eligible)),
        "n_complete_policy_cells": int(len(candidates)),
    }


def build_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for policy, g in selected.groupby("policy", sort=False):
        point_counts = g["search_point_classification"].value_counts()

        rows.append(
            {
                "policy": policy,
                "backgrounds": len(g),
                "share_same_as_avg_opt": float(
                    g["same_as_avg_utilization_optimum"].mean()
                ),
                "median_utilization_change_vs_avg_opt": float(
                    g["search_utilization_change_vs_avg_opt"].median()
                ),
                "min_utilization_change_vs_avg_opt": float(
                    g["search_utilization_change_vs_avg_opt"].min()
                ),
                "median_worst_access_gain_vs_baseline": float(
                    g["search_worst_access_gain_vs_baseline"].median()
                ),
                "point_win_win": int(
                    point_counts.get("point_win_win", 0)
                ),
                "point_c1_win_c2_neutral": int(
                    point_counts.get(
                        "point_c1_win_c2_neutral", 0
                    )
                ),
                "point_c2_win_c1_neutral": int(
                    point_counts.get(
                        "point_c2_win_c1_neutral", 0
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Existing patient_behavior_factorial_3_5 output directory, "
            "e.g. /scratch/$USER/patient_behavior_factorial_3_5"
        ),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help=(
            "Optional destination directory. Defaults to "
            "<output-dir>/baseline_access_optimization."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    destination = (
        args.destination.resolve()
        if args.destination is not None
        else output_dir / "baseline_access_optimization"
    )

    raw_dir = output_dir / "search" / "raw"
    selected_path = output_dir / "selection" / "selected_cells.csv"

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Search raw directory not found: {raw_dir}"
        )
    if not selected_path.exists():
        raise FileNotFoundError(
            f"Original selection file not found: {selected_path}"
        )

    original = pd.read_csv(selected_path)
    original_avg = original[
        original["selection_objective"] == "average_utilization"
    ].copy()

    needed = {
        "background_id",
        "policy",
        "selected_horizon_days",
        "selected_Q",
        "selected_window",
        "search_objective_mean",
        "search_n_seeds",
    }
    missing = sorted(needed - set(original_avg.columns))
    if missing:
        raise ValueError(
            f"selected_cells.csv is missing columns: {missing}"
        )

    ref_index = (
        original_avg[
            original_avg["policy"].isin(POLICIES)
        ]
        .set_index(["background_id", "policy"])
    )

    shards = sorted(raw_dir.glob("*.csv"))
    if not shards:
        raise FileNotFoundError(
            f"No search shards found under {raw_dir}"
        )

    rows: list[dict[str, object]] = []

    for i, path in enumerate(shards, start=1):
        search = pd.read_csv(path)

        if "source_background_id" not in search.columns:
            raise ValueError(
                f"{path} is missing source_background_id"
            )

        background_id = str(
            search["source_background_id"].iloc[0]
        )

        for policy in POLICIES:
            key = (background_id, policy)
            if key not in ref_index.index:
                raise KeyError(
                    f"Missing original avg-util reference for {key}"
                )

            ref = ref_index.loc[key]

            if isinstance(ref, pd.DataFrame):
                if len(ref) != 1:
                    raise ValueError(
                        f"Duplicate original avg-util rows for {key}"
                    )
                ref = ref.iloc[0]

            rows.append(
                select_one(
                    search=search,
                    background_id=background_id,
                    policy=policy,
                    ref=ref,
                )
            )

        if i % 50 == 0 or i == len(shards):
            print(
                f"Selected {i:,}/{len(shards):,} backgrounds"
            )

    selected = pd.DataFrame(rows)

    expected_rows = len(shards) * len(POLICIES)
    if len(selected) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows:,} rows; got {len(selected):,}"
        )

    destination.mkdir(parents=True, exist_ok=True)

    selected_out = destination / "baseline_access_cells.csv"
    summary_out = destination / "selection_summary.csv"

    selected.to_csv(selected_out, index=False)

    summary = build_summary(selected)
    summary.to_csv(summary_out, index=False)

    print("\nSelection complete.")
    print(f"Selected cells: {selected_out}")
    print(f"Summary:        {summary_out}")
    print("\nDiagnostic summary (SEARCH seeds only; not final inference):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
