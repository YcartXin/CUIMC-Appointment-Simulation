#!/usr/bin/env python3
"""Evaluate frozen access-recovery/favorable candidates on independent seeds.

This script does NOT search or re-optimize.

For each factorial background it builds the union of:
1. the original average-utilization-optimal cells for baseline, horizon_only,
   reservation_only, and both_flexible; and
2. all existing access-recovery/favorable candidate cells with
   candidate_exists=True:
      - access_recovery
      - best_win_win
      - c1_win_c2_neutral_if_better

It then evaluates only missing (H, Q, window, seed) combinations using the
experiment's independent EVALUATION_SEED_POOL. Existing evaluation rows are
reused automatically by _filter_pending.

Run with --dry-run first to count pending simulations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.patient_behavior_factorial import (
    EVALUATION_KEY_COLUMNS,
    EVALUATION_SEED_POOL,
    _filter_pending,
    _task_grid,
    load_bank,
    run_tasks,
    shard_path,
)

POLICIES = (
    "baseline",
    "horizon_only",
    "reservation_only",
    "both_flexible",
)

CANDIDATE_TYPES = (
    "access_recovery",
    "best_win_win",
    "c1_win_c2_neutral_if_better",
)


def _sharded_rows(
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


def _int_cell(
    row: pd.Series,
    *,
    h_col: str,
    q_col: str,
    w_col: str,
) -> tuple[int, int, int]:
    return (
        int(row[h_col]),
        int(row[q_col]),
        int(row[w_col]),
    )


def _cells_for_background(
    *,
    original_avg: pd.DataFrame,
    candidates: pd.DataFrame,
    background_id: str,
) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    cells: set[tuple[int, int, int]] = set()

    original = original_avg[
        original_avg["background_id"] == background_id
    ]

    if original.empty:
        raise ValueError(
            f"No original average-utilization selections for {background_id}"
        )

    for _, r in original.iterrows():
        cells.add(
            _int_cell(
                r,
                h_col="selected_horizon_days",
                q_col="selected_Q",
                w_col="selected_window",
            )
        )

    exists = candidates["candidate_exists"].astype(str).str.lower().eq("true")

    subset = candidates[
        (candidates["background_id"] == background_id)
        & exists
        & (candidates["candidate_type"].isin(CANDIDATE_TYPES))
    ]

    counts = {
        candidate_type: int(
            (subset["candidate_type"] == candidate_type).sum()
        )
        for candidate_type in CANDIDATE_TYPES
    }

    for _, r in subset.iterrows():
        cells.add(
            _int_cell(
                r,
                h_col="selected_horizon_days",
                q_col="selected_Q",
                w_col="selected_window",
            )
        )

    return sorted(cells), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.n_seeds <= len(EVALUATION_SEED_POOL):
        raise ValueError("Invalid --n-seeds")

    seeds = EVALUATION_SEED_POOL[
        : (2 if args.smoke else args.n_seeds)
    ]

    root = args.output_dir.resolve()

    candidates_path = (
        root
        / "access_recovery_optimization"
        / "access_recovery_candidates.csv"
    )

    selected_path = (
        root
        / "selection"
        / "selected_cells.csv"
    )

    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Missing candidate file: {candidates_path}"
        )

    if not selected_path.exists():
        raise FileNotFoundError(
            f"Missing original selection file: {selected_path}"
        )

    candidates = pd.read_csv(candidates_path)
    selected = pd.read_csv(selected_path)

    original_avg = selected[
        (selected["selection_objective"] == "average_utilization")
        & (selected["policy"].isin(POLICIES))
    ].copy()

    bank = load_bank(args.bank)
    rows = _sharded_rows(
        bank,
        args.shard_index,
        args.shard_count,
        args.smoke,
    )

    raw_dir = root / "evaluation" / "raw"

    total_unique_cells = 0
    total_planned_runs = 0
    total_pending_runs = 0
    backgrounds_with_pending = 0

    candidate_counts = {
        candidate_type: 0
        for candidate_type in CANDIDATE_TYPES
    }

    for number, (_, bank_row) in enumerate(
        rows.iterrows(),
        start=1,
    ):
        background_id = str(bank_row["background_id"])

        cells, counts = _cells_for_background(
            original_avg=original_avg,
            candidates=candidates,
            background_id=background_id,
        )

        for k, v in counts.items():
            candidate_counts[k] += v

        tasks = _task_grid(
            bank_row,
            stage="evaluation",
            phase="independent_access_recovery_evaluation",
            cells=cells,
            seeds=seeds,
            smoke=args.smoke,
        )

        path = shard_path(
            raw_dir,
            background_id,
        )

        pending = _filter_pending(
            tasks,
            path,
            EVALUATION_KEY_COLUMNS,
        )

        total_unique_cells += len(cells)
        total_planned_runs += len(tasks)
        total_pending_runs += len(pending)
        backgrounds_with_pending += int(bool(pending))

        if args.dry_run:
            if number % 20 == 0 or number == len(rows):
                print(
                    f"[{number}/{len(rows)}] "
                    f"{len(cells)} cells, "
                    f"{len(tasks)} planned, "
                    f"{len(pending)} pending"
                )
        elif pending:
            print(
                f"{background_id}: "
                f"{len(cells)} cells; "
                f"{len(pending)} pending"
            )
            run_tasks(
                pending,
                path,
                args.workers,
            )

    print("\nIndependent access-recovery evaluation summary")
    print(f"Shard backgrounds:              {len(rows)}")
    print(f"Unique cells across backgrounds:{total_unique_cells:>9}")
    print(f"Planned simulation runs:        {total_planned_runs}")
    print(f"Pending simulation runs:        {total_pending_runs}")
    print(
        "Backgrounds with pending work:  "
        f"{backgrounds_with_pending}"
    )

    print("\nFrozen candidate rows included")
    for candidate_type in CANDIDATE_TYPES:
        print(
            f"  {candidate_type}: "
            f"{candidate_counts[candidate_type]}"
        )

    if args.dry_run:
        print(
            "\nDRY RUN ONLY: no evaluation rows were appended."
        )


if __name__ == "__main__":
    main()
