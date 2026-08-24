#!/usr/bin/env python3
"""Targeted refinement for baseline-referenced access optimization.

For each background and policy family (reservation_only, both_flexible):

1. Keep the original average-utilization optimum U* frozen.
2. Define the strict feasible set by U >= U* - 0.005.
3. Measure Class-1/Class-2 access relative to the matched no-policy baseline.
4. Treat a reservation as genuinely active only when Q > 0 and window >= 2.
   Under same-day release, window=1 is operationally equivalent to no reservation.
5. Refine ONE local active-reservation neighborhood when:
   a) the current baseline-access winner genuinely uses a reservation, OR
   b) the current winner is non-reservation and the best genuinely active
      alternative is within 0.001 (= 0.10 percentage points) in the
      max-min access objective.

This screening threshold is only for deciding where extra simulations are
worth running. Final selection still uses the exact 0.005 utilization floor
and no 0.001 access tolerance.
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
UTIL_TOL = 0.005
NONACTIVE_MAX_GAP = 0.001
MIN_ACTIVE_WINDOW = 2
CELL_COLS = ["horizon_days", "Q", "window"]


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


def _best_genuine_active(
    *,
    search: pd.DataFrame,
    policy: str,
    frozen_u_star: float,
    expected_seeds: int,
) -> pd.Series | None:
    x = search[
        (search["stage"] == policy)
        & (search["Q"] > 0)
        & (search["window"] >= MIN_ACTIVE_WINDOW)
    ].copy()

    x = _dedupe(x)

    if x.empty:
        return None

    g = (
        x.groupby(CELL_COLS, as_index=False)
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

    g = g[g["n_seeds"] == expected_seeds].copy()
    g = g[
        g["average_utilization"]
        >= frozen_u_star - UTIL_TOL
    ].copy()

    if g.empty:
        return None

    b1, b2 = _baseline_means(search, expected_seeds)

    g["d1"] = g["class_1_percent_serviced"] - b1
    g["d2"] = g["class_2_percent_serviced"] - b2
    g["worst"] = np.minimum(g["d1"], g["d2"])
    g["sumgain"] = g["d1"] + g["d2"]

    return (
        g.sort_values(
            [
                "worst",
                "average_utilization",
                "sumgain",
                "Q",
                "window",
                "horizon_days",
            ],
            ascending=[False, False, False, True, True, True],
            kind="stable",
        )
        .iloc[0]
    )


def _target_cell(
    *,
    search: pd.DataFrame,
    selected_row: pd.Series,
    expected_seeds: int,
) -> tuple[tuple[int, int, int] | None, str, float | None]:
    policy = str(selected_row["policy"])
    u_star = float(
        selected_row["frozen_avg_opt_average_utilization"]
    )

    best_active = _best_genuine_active(
        search=search,
        policy=policy,
        frozen_u_star=u_star,
        expected_seeds=expected_seeds,
    )

    if best_active is None:
        return None, "no_feasible_active_cell", None

    selected_is_active = (
        int(selected_row["selected_Q"]) > 0
        and int(selected_row["selected_window"]) >= MIN_ACTIVE_WINDOW
    )

    cell = (
        int(best_active["horizon_days"]),
        int(best_active["Q"]),
        int(best_active["window"]),
    )

    if selected_is_active:
        return cell, "current_winner_active", 0.0

    gap = (
        float(selected_row["search_worst_access_gain_vs_baseline"])
        - float(best_active["worst"])
    )

    if gap <= NONACTIVE_MAX_GAP + 1e-12:
        return cell, "close_active_alternative", gap

    return None, "active_alternative_not_close", gap


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

    # One-dimensional local refinement around the target cell.
    for q2 in q_fine_grid(q, capacity):
        q2 = int(q2)
        if q2 > 0:
            cells.add((h, q2, w))

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
                    phase="fine_baseline_access",
                    smoke=smoke,
                )
            )

    # Remove duplicates inside the task list.
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

    seeds = SEARCH_SEED_POOL[: (2 if args.smoke else args.n_seeds)]

    bank = load_bank(args.bank)
    rows = _sharded_bank(
        bank,
        args.shard_index,
        args.shard_count,
        args.smoke,
    )

    raw_dir = args.output_dir / "search" / "raw"
    selection_path = (
        args.output_dir
        / "baseline_access_optimization"
        / "baseline_access_cells.csv"
    )

    if not selection_path.exists():
        raise FileNotFoundError(
            f"Missing baseline-access selection: {selection_path}"
        )

    selected = pd.read_csv(selection_path)
    selected = selected[
        selected["policy"].isin(POLICIES)
    ].copy()
    selected_idx = selected.set_index(
        ["background_id", "policy"]
    )

    total_planned = 0
    total_pending = 0
    targeted_policy_backgrounds = 0
    backgrounds_with_pending = 0

    reason_counts = {
        p: {
            "current_winner_active": 0,
            "close_active_alternative": 0,
            "active_alternative_not_close": 0,
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

        bg_planned = 0
        bg_pending = 0

        for policy in POLICIES:
            key = (bg, policy)

            if key not in selected_idx.index:
                raise KeyError(
                    f"Missing baseline-access selection for {key}"
                )

            sr = selected_idx.loc[key]

            if isinstance(sr, pd.DataFrame):
                if len(sr) != 1:
                    raise ValueError(
                        f"Duplicate baseline-access selections for {key}"
                    )
                sr = sr.iloc[0]

            target, reason, gap = _target_cell(
                search=search,
                selected_row=sr,
                expected_seeds=len(seeds),
            )

            reason_counts[policy][reason] += 1

            if target is None:
                continue

            targeted_policy_backgrounds += 1

            tasks = _make_refinement_tasks(
                bank_row=bank_row,
                target=target,
                policy=policy,
                seeds=seeds,
                smoke=args.smoke,
            )

            pending = _filter_pending(
                tasks,
                raw_path,
                SEARCH_KEY_COLUMNS,
            )

            bg_planned += len(tasks)
            bg_pending += len(pending)

            if not args.dry_run and pending:
                gap_text = (
                    "n/a"
                    if gap is None
                    else f"{100 * gap:.4f} pp"
                )
                print(
                    f"{bg} / {policy}: {reason}; "
                    f"gap={gap_text}; target={target}; "
                    f"{len(tasks)} planned; "
                    f"{len(pending)} pending"
                )

                run_tasks(
                    pending,
                    raw_path,
                    args.workers,
                )
                search = pd.read_csv(raw_path)

        total_planned += bg_planned
        total_pending += bg_pending
        backgrounds_with_pending += int(bg_pending > 0)

        if (
            args.dry_run
            and (number % 20 == 0 or number == len(rows))
        ):
            print(
                f"[{number}/{len(rows)}] "
                f"{bg_planned} planned, {bg_pending} pending"
            )

    print("\nTargeted baseline-access refinement summary")
    print(f"Shard backgrounds:              {len(rows)}")
    print(
        "Targeted policy-backgrounds:    "
        f"{targeted_policy_backgrounds}"
    )
    print(f"Planned simulation runs:        {total_planned}")
    print(f"Pending simulation runs:        {total_pending}")
    print(
        "Backgrounds with pending work:  "
        f"{backgrounds_with_pending}"
    )

    print("\nScreening counts")
    for policy in POLICIES:
        print(f"\n{policy}")
        for reason, count in reason_counts[policy].items():
            print(f"  {reason}: {count}")

    if args.dry_run:
        print("\nDRY RUN ONLY: no rows were appended.")


if __name__ == "__main__":
    main()
