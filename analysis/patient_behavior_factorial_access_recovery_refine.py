#!/usr/bin/env python3
"""Targeted refinement for the 3x3 access-recovery analysis.

For each background and reservation-capable policy:

- Freeze the original average-utilization optimum U*.
- Restrict attention to genuinely active reservation cells:
      Q > 0 and window >= 2
- Enforce:
      U >= U* - 0.005
- Refine up to three DISTINCT local neighborhoods:
  1. access_recovery:
       maximize Class-1 served rate, then Class-2 served rate, then utilization.
  2. win_win_frontier:
       maximize min(dSR1_vs_baseline, dSR2_vs_baseline).  This is the best
       current win-win if one exists, otherwise the closest active cell to one.
  3. win_neutral_frontier:
       minimize violation of the Class-2 neutral band |dSR2| <= 0.005,
       then maximize Class-1 served-rate gain.  This is the best current
       C1-win/C2-neutral candidate if one exists, otherwise a nearby frontier
       cell that fine refinement could move into the neutral band.

Only reservation_only and both_flexible are refined here. Horizon-only already
uses an integer daily horizon grid, so there is no finer horizon resolution to
simulate. For both_flexible, rerunning selection after each refinement pass can
move the chosen horizon; a second pass can then refine Q/window at that horizon.

This script can be run with --dry-run to count the new simulations before any
rows are appended.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.patient_behavior_factorial import (
    SEARCH_KEY_COLUMNS,
    SEARCH_SEED_POOL,
    _filter_pending,
    _make_task,
    load_bank,
    q_fine_grid,
    run_tasks,
    window_fine_grid,
)

POLICIES = ("reservation_only", "both_flexible")
CELL_COLS = ["horizon_days", "Q", "window"]

UTIL_TOL = 0.005
NEUTRAL_BAND = 0.005
MIN_ACTIVE_WINDOW = 2


def _dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    x = frame.copy()
    rank = {
        "coarse": 0,
        "fine_average_utilization": 1,
        "fine_priority_weighted_utilization": 2,
        "fine_access_balance": 3,
        "fine_baseline_access": 4,
        "fine_access_recovery": 5,
    }

    x["_rank"] = x.get(
        "arm",
        pd.Series("", index=x.index),
    ).map(rank).fillna(99)

    return (
        x.sort_values(CELL_COLS + ["seed", "_rank"], kind="stable")
        .drop_duplicates(CELL_COLS + ["seed"], keep="first")
        .drop(columns="_rank")
    )


def _aggregate(
    frame: pd.DataFrame,
    expected_seeds: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    g = (
        _dedupe(frame)
        .groupby(CELL_COLS, as_index=False)
        .agg(
            average_utilization=("average_utilization", "mean"),
            class_1_percent_serviced=(
                "class_1_percent_serviced",
                "mean",
            ),
            class_2_percent_serviced=(
                "class_2_percent_serviced",
                "mean",
            ),
            n_seeds=("seed", "nunique"),
        )
    )

    return g[g["n_seeds"] == expected_seeds].copy()


def _baseline_means(
    search: pd.DataFrame,
    expected_seeds: int,
) -> tuple[float, float]:
    b = _dedupe(
        search[search["stage"] == "baseline"].copy()
    )

    if int(b["seed"].nunique()) != expected_seeds:
        raise ValueError(
            f"Baseline has {b['seed'].nunique()} seeds; "
            f"expected {expected_seeds}."
        )

    return (
        float(b["class_1_percent_serviced"].mean()),
        float(b["class_2_percent_serviced"].mean()),
    )


def _active_feasible(
    *,
    search: pd.DataFrame,
    policy: str,
    u_star: float,
    expected_seeds: int,
) -> pd.DataFrame:
    x = search[
        (search["stage"] == policy)
        & (search["Q"] > 0)
        & (search["window"] >= MIN_ACTIVE_WINDOW)
    ].copy()

    g = _aggregate(x, expected_seeds)

    if g.empty:
        return g

    g = g[
        g["average_utilization"]
        >= u_star - UTIL_TOL
    ].copy()

    if g.empty:
        return g

    b1, b2 = _baseline_means(search, expected_seeds)

    g["d1"] = (
        g["class_1_percent_serviced"] - b1
    )
    g["d2"] = (
        g["class_2_percent_serviced"] - b2
    )
    g["worst"] = np.minimum(g["d1"], g["d2"])
    g["neutral_violation"] = np.maximum(
        g["d2"].abs() - NEUTRAL_BAND,
        0.0,
    )

    return g


def _target_cells(
    feasible: pd.DataFrame,
) -> dict[str, tuple[int, int, int]]:
    if feasible.empty:
        return {}

    targets: dict[str, tuple[int, int, int]] = {}

    # 1. Primary access-recovery point.
    access = (
        feasible.sort_values(
            [
                "class_1_percent_serviced",
                "class_2_percent_serviced",
                "average_utilization",
                "Q",
                "window",
                "horizon_days",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                True,
                True,
            ],
            kind="stable",
        )
        .iloc[0]
    )

    targets["access_recovery"] = (
        int(access["horizon_days"]),
        int(access["Q"]),
        int(access["window"]),
    )

    # 2. Best current win-win / closest win-win frontier point.
    ww = (
        feasible.sort_values(
            [
                "worst",
                "d1",
                "d2",
                "average_utilization",
                "Q",
                "window",
                "horizon_days",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                True,
                True,
                True,
            ],
            kind="stable",
        )
        .iloc[0]
    )

    targets["win_win_frontier"] = (
        int(ww["horizon_days"]),
        int(ww["Q"]),
        int(ww["window"]),
    )

    # 3. Best current C1-win/C2-neutral / closest neutral frontier point.
    wn = (
        feasible.sort_values(
            [
                "neutral_violation",
                "d1",
                "average_utilization",
                "d2",
                "Q",
                "window",
                "horizon_days",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                True,
                True,
                True,
            ],
            kind="stable",
        )
        .iloc[0]
    )

    targets["win_neutral_frontier"] = (
        int(wn["horizon_days"]),
        int(wn["Q"]),
        int(wn["window"]),
    )

    return targets


def _make_refinement_tasks(
    *,
    bank_row: pd.Series,
    target: tuple[int, int, int],
    policy: str,
    seeds: tuple[int, ...],
    smoke: bool,
) -> list[dict]:
    h, q, w = target
    capacity = int(bank_row["slots_per_day"])

    cells: set[tuple[int, int, int]] = set()

    # Coordinate refinement in Q at fixed H,W.
    for q2 in q_fine_grid(q, capacity):
        q2 = int(q2)
        if q2 > 0:
            cells.add((h, q2, w))

    # Coordinate refinement in window at fixed H,Q.
    for w2 in window_fine_grid(w, h):
        w2 = int(w2)
        if w2 >= MIN_ACTIVE_WINDOW:
            cells.add((h, q, w2))

    tasks = []

    for h2, q2, w2 in sorted(cells):
        for seed in seeds:
            tasks.append(
                _make_task(
                    bank_row,
                    horizon=h2,
                    q=q2,
                    window=w2,
                    seed=seed,
                    stage=policy,
                    phase="fine_access_recovery",
                    smoke=smoke,
                )
            )

    # Remove duplicates inside this target.
    seen = set()
    unique = []

    for task in tasks:
        key = tuple(
            task["seed"]
            if c == "seed"
            else task["extra_cols"][c]
            for c in SEARCH_KEY_COLUMNS
        )

        if key not in seen:
            seen.add(key)
            unique.append(task)

    return unique


def _sharded_bank(
    bank: pd.DataFrame,
    shard_index: int,
    shard_count: int,
    smoke: bool,
) -> pd.DataFrame:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard index/count")

    rows = (
        bank.sort_values("background_id", kind="stable")
        .reset_index(drop=True)
        .iloc[shard_index::shard_count]
    )

    return rows.head(2) if smoke else rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.n_seeds <= len(SEARCH_SEED_POOL):
        raise ValueError("Invalid --n-seeds")

    seeds = SEARCH_SEED_POOL[
        : (2 if args.smoke else args.n_seeds)
    ]

    bank = load_bank(args.bank)
    rows = _sharded_bank(
        bank,
        args.shard_index,
        args.shard_count,
        args.smoke,
    )

    raw_dir = args.output_dir / "search" / "raw"
    candidates_path = (
        args.output_dir
        / "access_recovery_optimization"
        / "access_recovery_candidates.csv"
    )

    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Missing access-recovery selection: "
            f"{candidates_path}"
        )

    selected = pd.read_csv(candidates_path)
    selected = selected[
        (selected["candidate_type"] == "access_recovery")
        & (selected["policy"].isin(POLICIES))
    ].copy()

    selected_idx = selected.set_index(
        ["background_id", "policy"],
        drop=False,
    )

    total_planned = 0
    total_pending = 0
    backgrounds_with_pending = 0
    targeted_policy_backgrounds = 0
    distinct_target_cells = 0

    target_type_counts = {
        p: {
            "access_recovery": 0,
            "win_win_frontier": 0,
            "win_neutral_frontier": 0,
            "no_feasible_active_cell": 0,
        }
        for p in POLICIES
    }

    for number, (_, bank_row) in enumerate(
        rows.iterrows(),
        start=1,
    ):
        bg = str(bank_row["background_id"])
        raw_path = raw_dir / f"{bg}.csv"

        if not raw_path.exists():
            raise FileNotFoundError(
                f"Missing search shard: {raw_path}"
            )

        search = pd.read_csv(raw_path)

        bg_planned_tasks: list[dict] = []

        for policy in POLICIES:
            key = (bg, policy)

            if key not in selected_idx.index:
                raise KeyError(
                    f"Missing access-recovery row for {key}"
                )

            sr = selected_idx.loc[key]

            if isinstance(sr, pd.DataFrame):
                if len(sr) != 1:
                    raise ValueError(
                        f"Duplicate access-recovery rows for {key}"
                    )
                sr = sr.iloc[0]

            u_star = float(
                sr["avg_opt_average_utilization"]
            )

            feasible = _active_feasible(
                search=search,
                policy=policy,
                u_star=u_star,
                expected_seeds=len(seeds),
            )

            targets = _target_cells(feasible)

            if not targets:
                target_type_counts[
                    policy
                ]["no_feasible_active_cell"] += 1
                continue

            targeted_policy_backgrounds += 1

            # Count target purposes, then dedupe identical cells.
            for reason in targets:
                target_type_counts[policy][reason] += 1

            unique_cells = sorted(set(targets.values()))
            distinct_target_cells += len(unique_cells)

            policy_tasks: list[dict] = []

            for target in unique_cells:
                policy_tasks.extend(
                    _make_refinement_tasks(
                        bank_row=bank_row,
                        target=target,
                        policy=policy,
                        seeds=seeds,
                        smoke=args.smoke,
                    )
                )

            # Dedupe across multiple target neighborhoods in this policy/background.
            seen = set()
            unique_policy_tasks = []

            for task in policy_tasks:
                task_key = tuple(
                    task["seed"]
                    if c == "seed"
                    else task["extra_cols"][c]
                    for c in SEARCH_KEY_COLUMNS
                )

                if task_key not in seen:
                    seen.add(task_key)
                    unique_policy_tasks.append(task)

            bg_planned_tasks.extend(unique_policy_tasks)

        # Dedupe once more across the background, then filter already-run cells.
        seen = set()
        unique_bg_tasks = []

        for task in bg_planned_tasks:
            task_key = tuple(
                task["seed"]
                if c == "seed"
                else task["extra_cols"][c]
                for c in SEARCH_KEY_COLUMNS
            )

            if task_key not in seen:
                seen.add(task_key)
                unique_bg_tasks.append(task)

        pending = _filter_pending(
            unique_bg_tasks,
            raw_path,
            SEARCH_KEY_COLUMNS,
        )

        total_planned += len(unique_bg_tasks)
        total_pending += len(pending)
        backgrounds_with_pending += int(len(pending) > 0)

        if not args.dry_run and pending:
            print(
                f"{bg}: {len(unique_bg_tasks)} planned; "
                f"{len(pending)} pending"
            )

            run_tasks(
                pending,
                raw_path,
                args.workers,
            )

        if (
            args.dry_run
            and (number % 20 == 0 or number == len(rows))
        ):
            print(
                f"[{number}/{len(rows)}] "
                f"{len(unique_bg_tasks)} planned, "
                f"{len(pending)} pending"
            )

    print("\nAccess-recovery targeted refinement summary")
    print(
        f"Shard backgrounds:              {len(rows)}"
    )
    print(
        "Targeted policy-backgrounds:    "
        f"{targeted_policy_backgrounds}"
    )
    print(
        "Distinct target cells:          "
        f"{distinct_target_cells}"
    )
    print(
        f"Planned simulation runs:        {total_planned}"
    )
    print(
        f"Pending simulation runs:        {total_pending}"
    )
    print(
        "Backgrounds with pending work:  "
        f"{backgrounds_with_pending}"
    )

    print("\nTarget-purpose counts")
    for policy in POLICIES:
        print(f"\n{policy}")
        for reason, count in target_type_counts[policy].items():
            print(f"  {reason}: {count}")

    if args.dry_run:
        print(
            "\nDRY RUN ONLY: no rows were appended."
        )


if __name__ == "__main__":
    main()
