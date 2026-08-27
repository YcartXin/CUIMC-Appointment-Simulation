#!/usr/bin/env python3
"""Targeted favorable-frontier refinement for the 3x3 experiment.

For reservation_only and both_flexible, refine only policy-backgrounds whose
existing 5-search-seed active-reservation cells are plausibly close to a
favorable outcome relative to the matched no-policy baseline, while remaining
within 0.5 percentage points of the policy's frozen average-utilization optimum.

Active reservation requires Q > 0 and window >= 2.

Win-win frontier screen:
    best min(dSR1, dSR2) >= 0.0
Final win-win still requires both gains >= +0.005.

C1-win/C2-neutral screen:
    dSR1 >= +0.005
    |dSR2| <= 0.010
    dSR2 < +0.005
If an exact win-win already exists, keep the win-neutral frontier only when its
Class-1 gain exceeds the best win-win's Class-1 gain.

There is intentionally no C2-win/C1-neutral target.

Run with --dry-run first.
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
WIN_THRESHOLD = 0.005
NEUTRAL_BAND = 0.005
WN_SCREEN_BAND = 0.010
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
        "fine_favorable_frontier": 6,
    }

    if "arm" in x.columns:
        x["_rank"] = x["arm"].map(rank).fillna(99)
    else:
        x["_rank"] = 99

    return (
        x.sort_values(CELL_COLS + ["seed", "_rank"], kind="stable")
        .drop_duplicates(CELL_COLS + ["seed"], keep="first")
        .drop(columns="_rank")
    )


def _aggregate(frame: pd.DataFrame, expected_seeds: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    g = (
        _dedupe(frame)
        .groupby(CELL_COLS, as_index=False)
        .agg(
            average_utilization=("average_utilization", "mean"),
            class_1_percent_serviced=("class_1_percent_serviced", "mean"),
            class_2_percent_serviced=("class_2_percent_serviced", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )

    return g[g["n_seeds"] == expected_seeds].copy()


def _baseline_means(
    search: pd.DataFrame,
    expected_seeds: int,
) -> tuple[float, float]:
    b = _dedupe(search[search["stage"] == "baseline"].copy())

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
    active = search[
        (search["stage"] == policy)
        & (search["Q"] > 0)
        & (search["window"] >= MIN_ACTIVE_WINDOW)
    ].copy()

    g = _aggregate(active, expected_seeds)

    if g.empty:
        return g

    g = g[
        g["average_utilization"] >= u_star - UTIL_TOL
    ].copy()

    if g.empty:
        return g

    b1, b2 = _baseline_means(search, expected_seeds)

    g["d1"] = g["class_1_percent_serviced"] - b1
    g["d2"] = g["class_2_percent_serviced"] - b2
    g["worst"] = np.minimum(g["d1"], g["d2"])

    return g


def _select_targets(
    feasible: pd.DataFrame,
) -> tuple[dict[str, tuple[int, int, int]], dict[str, bool]]:
    flags = {
        "win_win_exact_exists": False,
        "win_neutral_skipped_due_to_win_win": False,
    }

    if feasible.empty:
        return {}, flags

    targets: dict[str, tuple[int, int, int]] = {}

    # Win-win frontier: closest/best balanced favorable point.
    ww_frontier = (
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
            ascending=[False, False, False, False, True, True, True],
            kind="stable",
        )
        .iloc[0]
    )

    # Screen only if the worse class is already nonnegative.
    if float(ww_frontier["worst"]) >= -1e-12:
        targets["win_win_frontier"] = (
            int(ww_frontier["horizon_days"]),
            int(ww_frontier["Q"]),
            int(ww_frontier["window"]),
        )

    # Best exact search-seed win-win for the user's dominance rule.
    exact_ww = feasible[
        (feasible["d1"] >= WIN_THRESHOLD - 1e-12)
        & (feasible["d2"] >= WIN_THRESHOLD - 1e-12)
    ].copy()

    best_exact_ww = None
    if not exact_ww.empty:
        flags["win_win_exact_exists"] = True
        best_exact_ww = (
            exact_ww.sort_values(
                [
                    "d1",
                    "d2",
                    "average_utilization",
                    "Q",
                    "window",
                    "horizon_days",
                ],
                ascending=[False, False, False, True, True, True],
                kind="stable",
            )
            .iloc[0]
        )

    # Plausible C1-win/C2-neutral frontier:
    # Class 1 already wins; Class 2 lies within 0.5pp of the final +/-0.5pp
    # neutral band. Exact win-wins are excluded from this pool.
    wn_pool = feasible[
        (feasible["d1"] >= WIN_THRESHOLD - 1e-12)
        & (feasible["d2"].abs() <= WN_SCREEN_BAND + 1e-12)
        & (feasible["d2"] < WIN_THRESHOLD - 1e-12)
    ].copy()

    if not wn_pool.empty:
        wn_pool["neutral_distance"] = wn_pool["d2"].abs()

        best_wn = (
            wn_pool.sort_values(
                [
                    "d1",
                    "neutral_distance",
                    "average_utilization",
                    "d2",
                    "Q",
                    "window",
                    "horizon_days",
                ],
                ascending=[False, True, False, False, True, True, True],
                kind="stable",
            )
            .iloc[0]
        )

        keep_wn = True

        if best_exact_ww is not None:
            keep_wn = bool(
                float(best_wn["d1"])
                > float(best_exact_ww["d1"]) + 1e-12
            )

        if keep_wn:
            targets["win_neutral_frontier"] = (
                int(best_wn["horizon_days"]),
                int(best_wn["Q"]),
                int(best_wn["window"]),
            )
        else:
            flags["win_neutral_skipped_due_to_win_win"] = True

    return targets, flags


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

    # Fine Q search at fixed horizon and reservation window.
    for q2 in q_fine_grid(q, capacity):
        q2 = int(q2)
        if q2 > 0:
            cells.add((h, q2, w))

    # Fine reservation-window search at fixed horizon and Q.
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
                    phase="fine_favorable_frontier",
                    smoke=smoke,
                )
            )

    # Remove duplicates inside this neighborhood.
    seen = set()
    unique = []

    for task in tasks:
        key = tuple(
            task["seed"] if c == "seed" else task["extra_cols"][c]
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

    seeds = SEARCH_SEED_POOL[: (2 if args.smoke else args.n_seeds)]

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
            f"Missing access-recovery selection: {candidates_path}"
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

    counts = {
        p: {
            "win_win_frontier": 0,
            "win_neutral_frontier": 0,
            "win_win_exact_exists": 0,
            "win_neutral_skipped_due_to_win_win": 0,
            "screened_out_no_close_favorable_cell": 0,
            "no_feasible_active_cell": 0,
        }
        for p in POLICIES
    }

    for number, (_, bank_row) in enumerate(rows.iterrows(), start=1):
        bg = str(bank_row["background_id"])
        raw_path = raw_dir / f"{bg}.csv"

        if not raw_path.exists():
            raise FileNotFoundError(f"Missing search shard: {raw_path}")

        search = pd.read_csv(raw_path)
        bg_tasks: list[dict] = []

        for policy in POLICIES:
            key = (bg, policy)

            if key not in selected_idx.index:
                raise KeyError(f"Missing access-recovery row for {key}")

            sr = selected_idx.loc[key]

            if isinstance(sr, pd.DataFrame):
                if len(sr) != 1:
                    raise ValueError(
                        f"Duplicate access-recovery rows for {key}"
                    )
                sr = sr.iloc[0]

            u_star = float(sr["avg_opt_average_utilization"])

            feasible = _active_feasible(
                search=search,
                policy=policy,
                u_star=u_star,
                expected_seeds=len(seeds),
            )

            if feasible.empty:
                counts[policy]["no_feasible_active_cell"] += 1
                continue

            targets, flags = _select_targets(feasible)

            if flags["win_win_exact_exists"]:
                counts[policy]["win_win_exact_exists"] += 1

            if flags["win_neutral_skipped_due_to_win_win"]:
                counts[policy]["win_neutral_skipped_due_to_win_win"] += 1

            if not targets:
                counts[policy]["screened_out_no_close_favorable_cell"] += 1
                continue

            targeted_policy_backgrounds += 1

            for reason in targets:
                counts[policy][reason] += 1

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

            # Dedupe across the two frontier purposes.
            seen = set()
            for task in policy_tasks:
                task_key = tuple(
                    task["seed"]
                    if c == "seed"
                    else task["extra_cols"][c]
                    for c in SEARCH_KEY_COLUMNS
                )

                if task_key not in seen:
                    seen.add(task_key)
                    bg_tasks.append(task)

        # Dedupe across the entire background, then skip already-run cells.
        seen = set()
        unique_bg_tasks = []

        for task in bg_tasks:
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
            run_tasks(pending, raw_path, args.workers)

        if args.dry_run and (
            number % 20 == 0 or number == len(rows)
        ):
            print(
                f"[{number}/{len(rows)}] "
                f"{len(unique_bg_tasks)} planned, "
                f"{len(pending)} pending"
            )

    print("\nFavorable-frontier targeted refinement summary")
    print(f"Shard backgrounds:              {len(rows)}")
    print(
        "Targeted policy-backgrounds:    "
        f"{targeted_policy_backgrounds}"
    )
    print(f"Distinct target cells:          {distinct_target_cells}")
    print(f"Planned simulation runs:        {total_planned}")
    print(f"Pending simulation runs:        {total_pending}")
    print(
        "Backgrounds with pending work:  "
        f"{backgrounds_with_pending}"
    )

    print("\nTarget-purpose counts")
    for policy in POLICIES:
        print(f"\n{policy}")
        for reason, count in counts[policy].items():
            print(f"  {reason}: {count}")

    if args.dry_run:
        print("\nDRY RUN ONLY: no rows were appended.")


if __name__ == "__main__":
    main()
