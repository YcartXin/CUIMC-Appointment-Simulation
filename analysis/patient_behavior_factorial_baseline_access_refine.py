#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.patient_behavior_factorial import (
    SEARCH_KEY_COLUMNS, SEARCH_SEED_POOL, _filter_pending, _make_task,
    load_bank, q_fine_grid, run_tasks, window_fine_grid,
)

POLICIES = ("reservation_only", "both_flexible")
UTIL_TOL = 0.005
REFINE_MARGIN = 0.0025
TOP_ACTIVE_CELLS = 3
CELL_COLS = ["horizon_days", "Q", "window"]


def complete_means(frame, expected_seeds):
    if frame.empty:
        return pd.DataFrame()
    x = frame[frame["Q"] > 0].copy()
    if x.empty:
        return pd.DataFrame()
    rank = {
        "coarse": 0,
        "fine_average_utilization": 1,
        "fine_priority_weighted_utilization": 2,
        "fine_access_balance": 3,
        "fine_baseline_access": 4,
    }
    x["_rank"] = x["arm"].map(rank).fillna(99)
    x = (x.sort_values(CELL_COLS + ["seed", "_rank"], kind="stable")
           .drop_duplicates(CELL_COLS + ["seed"], keep="first"))
    g = x.groupby(CELL_COLS, as_index=False).agg(
        average_utilization=("average_utilization", "mean"),
        class_1_percent_serviced=("class_1_percent_serviced", "mean"),
        class_2_percent_serviced=("class_2_percent_serviced", "mean"),
        n_seeds=("seed", "nunique"),
    )
    return g[g["n_seeds"] == expected_seeds].copy()


def baseline_means(search, expected_seeds):
    b = search[search["stage"] == "baseline"].copy()
    b = (b.sort_values(CELL_COLS + ["seed"], kind="stable")
          .drop_duplicates(CELL_COLS + ["seed"], keep="first"))
    if int(b["seed"].nunique()) != expected_seeds:
        raise ValueError("Incomplete baseline seeds")
    return (
        float(b["class_1_percent_serviced"].mean()),
        float(b["class_2_percent_serviced"].mean()),
    )


def top_active(search, policy, u_star, expected_seeds):
    cells = complete_means(search[search["stage"] == policy], expected_seeds)
    if cells.empty:
        return cells
    b1, b2 = baseline_means(search, expected_seeds)
    cells["d1"] = cells["class_1_percent_serviced"] - b1
    cells["d2"] = cells["class_2_percent_serviced"] - b2
    cells["worst"] = np.minimum(cells["d1"], cells["d2"])
    cells["sumgain"] = cells["d1"] + cells["d2"]
    cells = cells[
        cells["average_utilization"] >= u_star - UTIL_TOL - REFINE_MARGIN
    ].copy()
    if cells.empty:
        return cells
    return (cells.sort_values(
        ["worst", "average_utilization", "sumgain", "Q", "window", "horizon_days"],
        ascending=[False, False, False, True, True, True],
        kind="stable",
    ).head(TOP_ACTIVE_CELLS))


def refinement_tasks(bank_row, search, policy, u_star, seeds, smoke):
    top = top_active(search, policy, u_star, len(seeds))
    capacity = int(bank_row["slots_per_day"])
    tasks = []
    for r in top.itertuples(index=False):
        h, q, w = int(r.horizon_days), int(r.Q), int(r.window)
        cells = set()
        for q2 in q_fine_grid(q, capacity):
            cells.add((h, int(q2), w))
        for w2 in window_fine_grid(w, h):
            cells.add((h, q, int(w2)))
        for h2, q2, w2 in sorted(cells):
            for seed in seeds:
                tasks.append(_make_task(
                    bank_row, horizon=h2, q=q2, window=w2, seed=seed,
                    stage=policy, phase="fine_baseline_access", smoke=smoke,
                ))
    seen, unique = set(), []
    for task in tasks:
        key = tuple(
            task["seed"] if c == "seed" else task["extra_cols"][c]
            for c in SEARCH_KEY_COLUMNS
        )
        if key not in seen:
            seen.add(key)
            unique.append(task)
    return unique


def sharded_bank(bank, shard_index, shard_count, smoke):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard index/count")
    rows = bank.sort_values("background_id", kind="stable").reset_index(drop=True)
    rows = rows.iloc[shard_index::shard_count]
    return rows.head(2) if smoke else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = SEARCH_SEED_POOL[: (2 if args.smoke else args.n_seeds)]
    bank = load_bank(args.bank)
    rows = sharded_bank(bank, args.shard_index, args.shard_count, args.smoke)
    raw_dir = args.output_dir / "search" / "raw"
    selected_path = args.output_dir / "selection" / "selected_cells.csv"
    selected = pd.read_csv(selected_path)
    avg = selected[
        (selected["selection_objective"] == "average_utilization")
        & selected["policy"].isin(POLICIES)
    ].copy()
    refs = avg.set_index(["background_id", "policy"])

    total_pending = total_planned = backgrounds_with_work = 0

    for number, (_, bank_row) in enumerate(rows.iterrows(), start=1):
        bg = str(bank_row["background_id"])
        path = raw_dir / f"{bg}.csv"
        search = pd.read_csv(path)
        bg_pending = bg_planned = 0

        for policy in POLICIES:
            ref = refs.loc[(bg, policy)]
            if isinstance(ref, pd.DataFrame):
                ref = ref.iloc[0]
            u_star = float(ref["search_objective_mean"])
            tasks = refinement_tasks(bank_row, search, policy, u_star, seeds, args.smoke)
            pending = _filter_pending(tasks, path, SEARCH_KEY_COLUMNS)
            bg_planned += len(tasks)
            bg_pending += len(pending)
            if not args.dry_run and pending:
                print(f"{bg} / {policy}: {len(tasks)} planned; {len(pending)} pending")
                run_tasks(pending, path, args.workers)
                search = pd.read_csv(path)

        total_planned += bg_planned
        total_pending += bg_pending
        backgrounds_with_work += int(bg_pending > 0)

        if args.dry_run or number % 10 == 0 or number == len(rows):
            print(f"[{number}/{len(rows)}] {bg}: {bg_planned} planned, {bg_pending} pending")

    print("\nBaseline-access refinement summary")
    print(f"Shard backgrounds:       {len(rows)}")
    print(f"Planned simulation runs: {total_planned}")
    print(f"Pending simulation runs: {total_pending}")
    print(f"Backgrounds with work:   {backgrounds_with_work}")
    if args.dry_run:
        print("DRY RUN ONLY: no rows were appended.")


if __name__ == "__main__":
    main()
