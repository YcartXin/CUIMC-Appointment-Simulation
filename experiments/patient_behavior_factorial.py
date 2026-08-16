"""Run the 3x3 Class-1 balking x no-show factorial horizon-reservation experiment.

Workflow
--------
1. ``search``: evaluate the full coarse grid with five search seeds, then refine
   the top three coarse (Q, window) cells for each objective and horizon.
2. ``select``: freeze each policy regime's selected cell under average
   utilization and priority-weighted utilization. Also freeze one
   utilization-constrained priority cell per background.
3. ``evaluate``: run only the frozen cells on independent evaluation seeds.

The reservation variant is same-day release only.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from analysis.metrics import result_metrics_from_result  # noqa: E402
from experiments.hypothesis_common import build_config  # noqa: E402
from simulation.engine import ClinicAppointmentSimulation  # noqa: E402

DEFAULT_BANK = REPO_DIR / "outputs" / "hypotheses" / "patient_behavior_factorial_bank.csv"
DEFAULT_OUTPUT = Path("/scratch") / "unknown" / "patient_behavior_factorial"

HORIZON_VALUES = tuple(range(2, 27))
OPEN_HORIZON_DAYS = 100
RESERVATION_WINDOW_MAX = 26
Q_COARSE_STEP = 5
WINDOW_COARSE_STEP = 2
Q_REFINE_RADIUS = 5
WINDOW_REFINE_RADIUS = 2
TOP_COARSE_CELLS = 3

SEARCH_SEED_POOL = tuple(range(1000, 1020))
EVALUATION_SEED_POOL = tuple(range(2000, 2100))
RUN_ORDER_SEED = 20260731

OBJECTIVES = ("average_utilization", "priority_weighted_utilization")
POLICIES = ("baseline", "horizon_only", "reservation_only", "both_flexible")
STAGE_RANK = {"baseline": 0, "horizon_only": 1, "reservation_only": 2, "both_flexible": 3}
PRACTICAL_TOLERANCE = 0.005
CLASS2_CONSTRAINT_TOLERANCE = 0.01

SEARCH_KEY_COLUMNS = ["stage", "horizon_days", "Q", "window", "seed"]
EVALUATION_KEY_COLUMNS = ["horizon_days", "Q", "window", "seed"]


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


def load_bank(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Bank not found: {path}")
    bank = pd.read_csv(path)
    required = {
        "background_id", "horizon_days", "slots_per_day", "lambda_1", "lambda_2",
        "cancel_1", "cancel_2", "balk_threshold_1", "balk_low_1", "balk_high_1",
        "balk_threshold_2", "balk_low_2", "balk_high_2", "noshow_threshold_1",
        "noshow_low_1", "noshow_high_1", "noshow_threshold_2", "noshow_low_2",
        "noshow_high_2", "cap_thresholds_to_horizon",
    }
    missing = sorted(required - set(bank.columns))
    if missing:
        raise ValueError(f"Bank is missing columns: {missing}")
    return bank


def _row_bool(row: pd.Series, column: str, *, default: bool) -> bool:
    if column not in row.index or pd.isna(row[column]):
        return default
    value = row[column]
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        raise ValueError(f"Invalid boolean value for {column}: {value!r}")
    return bool(value)


def _row_config_kwargs(row: pd.Series) -> dict[str, Any]:
    return {
        "slots_per_day": int(row["slots_per_day"]),
        "lambda_1": float(row["lambda_1"]),
        "lambda_2": float(row["lambda_2"]),
        "cancel_1": float(row["cancel_1"]),
        "cancel_2": float(row["cancel_2"]),
        "balk_threshold_1": int(row["balk_threshold_1"]),
        "balk_low_1": float(row["balk_low_1"]),
        "balk_high_1": float(row["balk_high_1"]),
        "balk_threshold_2": int(row["balk_threshold_2"]),
        "balk_low_2": float(row["balk_low_2"]),
        "balk_high_2": float(row["balk_high_2"]),
        "noshow_threshold_1": int(row["noshow_threshold_1"]),
        "noshow_low_1": float(row["noshow_low_1"]),
        "noshow_high_1": float(row["noshow_high_1"]),
        "noshow_threshold_2": int(row["noshow_threshold_2"]),
        "noshow_low_2": float(row["noshow_low_2"]),
        "noshow_high_2": float(row["noshow_high_2"]),
        "cap_thresholds_to_horizon": _row_bool(
            row, "cap_thresholds_to_horizon", default=False
        ),
    }


def _reservation_kwargs(q: int, window: int) -> dict[str, Any]:
    if q <= 0:
        return {
            "reserved_class_id": None,
            "reserved_slots_per_day": 0,
            "reserved_window_days": None,
            "same_day_cancellation_enabled": True,
            "release_unused_reservation_same_day": False,
        }
    return {
        "reserved_class_id": 1,
        "reserved_slots_per_day": int(q),
        "reserved_window_days": int(window),
        "same_day_cancellation_enabled": True,
        "release_unused_reservation_same_day": True,
    }


def _smoke_overrides(smoke: bool) -> dict[str, int]:
    return {"burn_in_days": 5, "measure_days": 20, "cooldown_days": 5} if smoke else {}


def run_one(task: Mapping[str, Any]) -> dict[str, Any]:
    config = build_config(seed=int(task["seed"]), **task["config_kwargs"])
    result = ClinicAppointmentSimulation(config).run()
    metrics = result_metrics_from_result(result)
    c1 = result.class_metrics[1]
    c2 = result.class_metrics[2]
    total_slots = float(result.total_slots)

    def rate(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    row: dict[str, Any] = {
        **metrics,
        "seed": int(task["seed"]),
        # Each completed Class-1 visit receives twice the value of a Class-2 visit.
        # Dividing by 2S normalizes by the maximum attainable weighted slot value.
        "priority_weighted_utilization": (
            (2.0 * c1.served + c2.served) / (2.0 * total_slots)
            if total_slots else math.nan
        ),
        "class_1_arrivals": int(c1.arrivals),
        "class_2_arrivals": int(c2.arrivals),
        "class_1_served": int(c1.served),
        "class_2_served": int(c2.served),
        "class_1_no_show_rate": rate(c1.no_show, c1.arrivals),
        "class_2_no_show_rate": rate(c2.no_show, c2.arrivals),
        "class_1_no_offer_rate": rate(c1.no_offer, c1.arrivals),
        "class_2_no_offer_rate": rate(c2.no_offer, c2.arrivals),
        "reserved_slot_fill_rate": result.reserved_slot_fill_rate,
    }
    row.update(task["extra_cols"])
    return row


def q_coarse_grid(capacity: int) -> list[int]:
    values = list(range(Q_COARSE_STEP, int(capacity) + 1, Q_COARSE_STEP))
    if not values or values[-1] != int(capacity):
        values.append(int(capacity))
    return values


def window_coarse_grid(horizon: int) -> list[int]:
    # Reservations remain a near-term policy. Opening the booking horizon to
    # 100 days should not silently expand the reservation window to 100 days.
    limit = min(int(horizon), RESERVATION_WINDOW_MAX)
    values = list(range(1, limit + 1, WINDOW_COARSE_STEP))
    if values[-1] != limit:
        values.append(limit)
    return values


def q_fine_grid(best_q: int, capacity: int) -> list[int]:
    if best_q <= 0:
        return []
    lo = max(1, int(best_q) - Q_REFINE_RADIUS)
    hi = min(int(capacity), int(best_q) + Q_REFINE_RADIUS)
    return [q for q in range(lo, hi + 1) if q != int(best_q)]


def window_fine_grid(best_window: int, horizon: int) -> list[int]:
    if best_window <= 0:
        return []
    lo = max(1, int(best_window) - WINDOW_REFINE_RADIUS)
    hi = min(int(horizon), RESERVATION_WINDOW_MAX, int(best_window) + WINDOW_REFINE_RADIUS)
    return [w for w in range(lo, hi + 1) if w != int(best_window)]


def _make_task(
    row: pd.Series,
    *,
    horizon: int,
    q: int,
    window: int,
    seed: int,
    stage: str,
    phase: str,
    smoke: bool,
) -> dict[str, Any]:
    background_id = str(row["background_id"])
    return {
        "config_kwargs": {
            **_row_config_kwargs(row),
            "horizon_days": int(horizon),
            **_reservation_kwargs(q, window),
            **_smoke_overrides(smoke),
            # A horizon H can create bookings as far out as H-1 days.
            # Give every measured booking enough cooldown to resolve.
            "cooldown_days": max(14, int(horizon)),
        },
        "seed": int(seed),
        "extra_cols": {
            "source_background_id": background_id,
            "background_id": f"{background_id}_{stage}_H={horizon}_Q={q}_W={window}",
            "stage": stage,
            "arm": phase,
            "horizon_days": int(horizon),
            "Q": int(q),
            "window": int(window),
        },
    }


def _task_grid(
    row: pd.Series,
    *,
    stage: str,
    phase: str,
    cells: Iterable[tuple[int, int, int]],
    seeds: Sequence[int],
    smoke: bool,
) -> list[dict[str, Any]]:
    return [
        _make_task(
            row,
            horizon=h,
            q=q,
            window=w,
            seed=seed,
            stage=stage,
            phase=phase,
            smoke=smoke,
        )
        for h, q, w in cells
        for seed in seeds
    ]


def _coarse_tasks(row: pd.Series, seeds: Sequence[int], smoke: bool) -> list[dict[str, Any]]:
    native_h = int(row["horizon_days"])
    capacity = int(row["slots_per_day"])
    horizons = HORIZON_VALUES[:3] if smoke else HORIZON_VALUES

    baseline = _task_grid(
        row, stage="baseline", phase="exact", cells=[(native_h, 0, -1)], seeds=seeds, smoke=smoke
    )
    horizon_only = _task_grid(
        row,
        stage="horizon_only",
        phase="exact",
        cells=[(h, 0, -1) for h in horizons],
        seeds=seeds,
        smoke=smoke,
    )

    q_values = q_coarse_grid(capacity)
    if smoke:
        q_values = q_values[:2]
    native_windows = window_coarse_grid(native_h)
    if smoke:
        native_windows = native_windows[:2]
    reservation_cells = [(native_h, q, w) for q in q_values for w in native_windows]
    reservation_only = _task_grid(
        row,
        stage="reservation_only",
        phase="coarse",
        cells=reservation_cells,
        seeds=seeds,
        smoke=smoke,
    )

    both_cells: list[tuple[int, int, int]] = []
    for horizon in horizons:
        windows = window_coarse_grid(horizon)
        if smoke:
            windows = windows[:2]
        both_cells.extend((horizon, q, w) for q in q_values for w in windows)
    both = _task_grid(
        row,
        stage="both_flexible",
        phase="coarse",
        cells=both_cells,
        seeds=seeds,
        smoke=smoke,
    )
    return baseline + horizon_only + reservation_only + both


def _dedupe_tasks(tasks: Iterable[dict[str, Any]], key_columns: Sequence[str]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    output: list[dict[str, Any]] = []
    for task in tasks:
        key = tuple(
            task["seed"] if column == "seed" else task["extra_cols"][column]
            for column in key_columns
        )
        if key not in seen:
            seen.add(key)
            output.append(task)
    return output


def _top_cells(cells: pd.DataFrame, objective: str, n: int = TOP_COARSE_CELLS) -> list[tuple[int, int, int]]:
    if cells.empty:
        return []
    deduped = cells.drop_duplicates(["horizon_days", "Q", "window", "seed"])
    means = (
        deduped.groupby(["horizon_days", "Q", "window"], as_index=False)[objective]
        .mean()
        .sort_values(
            [objective, "Q", "window", "horizon_days"],
            ascending=[False, True, True, True],
            kind="stable",
        )
    )
    return [
        (int(record.horizon_days), int(record.Q), int(record.window))
        for record in means.head(n).itertuples(index=False)
    ]


def _fine_tasks(
    row: pd.Series,
    search_rows: pd.DataFrame,
    seeds: Sequence[int],
    smoke: bool,
) -> list[dict[str, Any]]:
    native_h = int(row["horizon_days"])
    capacity = int(row["slots_per_day"])
    tasks: list[dict[str, Any]] = []

    baseline = search_rows[search_rows["stage"] == "baseline"]
    reservation_coarse = search_rows[
        (search_rows["stage"] == "reservation_only") & (search_rows["arm"] == "coarse")
    ]
    horizon_only = search_rows[search_rows["stage"] == "horizon_only"]
    both_coarse = search_rows[
        (search_rows["stage"] == "both_flexible") & (search_rows["arm"] == "coarse")
    ]

    for objective in OBJECTIVES:
        reservation_candidates = pd.concat([reservation_coarse, baseline], ignore_index=True)
        for horizon, q, window in _top_cells(reservation_candidates, objective):
            if q <= 0:
                continue
            cells = [(native_h, q2, window) for q2 in q_fine_grid(q, capacity)]
            cells += [(native_h, q, w2) for w2 in window_fine_grid(window, native_h)]
            tasks.extend(
                _task_grid(
                    row,
                    stage="reservation_only",
                    phase=f"fine_{objective}",
                    cells=cells,
                    seeds=seeds,
                    smoke=smoke,
                )
            )

        both_candidates = pd.concat([both_coarse, horizon_only], ignore_index=True)
        for h, q, window in _top_cells(both_candidates, objective):
            if q <= 0:
                continue
            cells = [(h, q2, window) for q2 in q_fine_grid(q, capacity)]
            cells += [(h, q, w2) for w2 in window_fine_grid(window, h)]
            tasks.extend(
                _task_grid(
                    row,
                    stage="both_flexible",
                    phase=f"fine_{objective}",
                    cells=cells,
                    seeds=seeds,
                    smoke=smoke,
                )
            )

    return _dedupe_tasks(tasks, SEARCH_KEY_COLUMNS)


def shard_path(raw_dir: Path, background_id: str) -> Path:
    return raw_dir / f"{background_id}.csv"


def _load_completed(path: Path, key_columns: Sequence[str]) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    existing = pd.read_csv(path, usecols=list(key_columns))
    return set(existing[list(key_columns)].itertuples(index=False, name=None))


def _filter_pending(
    tasks: list[dict[str, Any]], path: Path, key_columns: Sequence[str]
) -> list[dict[str, Any]]:
    completed = _load_completed(path, key_columns)
    pending = []
    for task in tasks:
        key = tuple(
            task["seed"] if column == "seed" else task["extra_cols"][column]
            for column in key_columns
        )
        if key not in completed:
            pending.append(task)
    return pending


def _task_keys(
    tasks: Iterable[dict[str, Any]], key_columns: Sequence[str]
) -> set[tuple[Any, ...]]:
    return {
        tuple(
            task["seed"] if column == "seed" else task["extra_cols"][column]
            for column in key_columns
        )
        for task in tasks
    }


def _validate_search_complete(
    row: pd.Series,
    search: pd.DataFrame,
    *,
    seeds: Sequence[int],
    smoke: bool,
) -> None:
    if search.duplicated(SEARCH_KEY_COLUMNS).any():
        raise ValueError(f"Duplicate search keys for {row['background_id']}")
    expected_coarse = _coarse_tasks(row, seeds, smoke)
    expected_fine = _fine_tasks(row, search, seeds, smoke)
    expected = _task_keys(expected_coarse + expected_fine, SEARCH_KEY_COLUMNS)
    observed = set(search[SEARCH_KEY_COLUMNS].itertuples(index=False, name=None))
    missing = expected - observed
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(
            f"Incomplete search for {row['background_id']}: "
            f"{len(missing):,} expected runs are missing; examples={examples}"
        )


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def run_tasks(tasks: list[dict[str, Any]], path: Path, workers: int, flush_every: int = 500) -> None:
    if not tasks:
        print("  Nothing new to run.")
        return
    buffer: list[dict[str, Any]] = []
    executor = None
    try:
        if workers <= 1:
            iterator = map(run_one, tasks)
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            iterator = executor.map(run_one, tasks, chunksize=4)
        for index, result in enumerate(iterator, start=1):
            buffer.append(result)
            if len(buffer) >= flush_every:
                _append_rows(path, buffer)
                buffer.clear()
            if index % 5000 == 0 or index == len(tasks):
                print(f"  Completed {index:,}/{len(tasks):,} runs")
        _append_rows(path, buffer)
    finally:
        if executor is not None:
            executor.shutdown()


def _sharded_rows(
    bank: pd.DataFrame,
    *,
    shard_index: int,
    shard_count: int,
    smoke: bool,
) -> pd.DataFrame:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard index/count")
    rows = bank.sample(frac=1.0, random_state=RUN_ORDER_SEED).reset_index(drop=True)
    rows = rows.iloc[shard_index::shard_count]
    return rows.head(4) if smoke else rows


def run_search(
    *,
    bank_path: Path,
    output_dir: Path,
    workers: int,
    n_seeds: int,
    shard_index: int,
    shard_count: int,
    smoke: bool,
) -> None:
    if not 1 <= n_seeds <= len(SEARCH_SEED_POOL):
        raise ValueError("Invalid number of search seeds")
    seeds = SEARCH_SEED_POOL[: (2 if smoke else n_seeds)]
    bank = load_bank(bank_path)
    rows = _sharded_rows(
        bank, shard_index=shard_index, shard_count=shard_count, smoke=smoke
    )
    raw_dir = output_dir / "search" / "raw"
    started = time.monotonic()

    for number, (_, row) in enumerate(rows.iterrows(), start=1):
        background_id = str(row["background_id"])
        path = shard_path(raw_dir, background_id)
        elapsed = time.monotonic() - started
        eta = ""
        if number > 1:
            per_background = elapsed / (number - 1)
            remaining = per_background * (len(rows) - number + 1) / 3600
            eta = f"; approx. {remaining:.1f} h remaining"
        print(f"\nSearch {number:,}/{len(rows):,}: {background_id}{eta}")

        coarse = _coarse_tasks(row, seeds, smoke)
        pending = _filter_pending(coarse, path, SEARCH_KEY_COLUMNS)
        print(f"  Coarse/exact: {len(coarse):,} total; {len(pending):,} pending")
        run_tasks(pending, path, workers)

        if not path.exists():
            raise RuntimeError(f"Search shard was not created: {path}")
        search_rows = pd.read_csv(path)
        fine = _fine_tasks(row, search_rows, seeds, smoke)
        pending = _filter_pending(fine, path, SEARCH_KEY_COLUMNS)
        print(f"  Top-{TOP_COARSE_CELLS} refinement: {len(fine):,} total; {len(pending):,} pending")
        run_tasks(pending, path, workers)

    print(f"Search output: {raw_dir}")


def _dedupe_cells(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.sort_values(
        ["horizon_days", "Q", "window", "seed", "stage"], kind="stable"
    ).drop_duplicates(["horizon_days", "Q", "window", "seed"], keep="first")


def _policy_candidates(search: pd.DataFrame, policy: str) -> pd.DataFrame:
    baseline = search[search["stage"] == "baseline"].copy()
    horizon = search[search["stage"] == "horizon_only"].copy()
    if policy == "baseline":
        return _dedupe_cells(baseline)
    if policy == "horizon_only":
        return _dedupe_cells(horizon)
    if policy == "reservation_only":
        return _dedupe_cells(
            pd.concat([search[search["stage"] == "reservation_only"], baseline], ignore_index=True)
        )
    if policy == "both_flexible":
        return _dedupe_cells(
            pd.concat([search[search["stage"] == "both_flexible"], horizon], ignore_index=True)
        )
    raise ValueError(f"Unknown policy: {policy}")


def _select_cell(cells: pd.DataFrame, objective: str) -> pd.Series:
    means = cells.groupby(["horizon_days", "Q", "window"], as_index=False)[objective].mean()
    best = means.sort_values(
        [objective, "Q", "window", "horizon_days"],
        ascending=[False, True, True, True],
        kind="stable",
    ).iloc[0]
    chosen = cells[
        (cells["horizon_days"] == best["horizon_days"])
        & (cells["Q"] == best["Q"])
        & (cells["window"] == best["window"])
    ]
    source_stage = chosen["stage"].mode().iloc[0]
    return pd.Series(
        {
            "selected_horizon_days": int(best["horizon_days"]),
            "selected_Q": int(best["Q"]),
            "selected_window": int(best["window"]),
            "selected_source_stage": source_stage,
            "search_objective_mean": float(best[objective]),
            "search_n_seeds": int(chosen["seed"].nunique()),
        }
    )


def _all_unique_candidates(search: pd.DataFrame) -> pd.DataFrame:
    frame = search.copy()
    frame["stage_rank"] = frame["stage"].map(STAGE_RANK).fillna(99)
    frame = frame.sort_values(
        ["horizon_days", "Q", "window", "seed", "stage_rank"], kind="stable"
    )
    return frame.drop_duplicates(["horizon_days", "Q", "window", "seed"], keep="first")


def _canonical_policy(native_horizon: int, horizon: int, q: int) -> str:
    if q == 0 and horizon == native_horizon:
        return "baseline"
    if q == 0:
        return "horizon_only"
    if horizon == native_horizon:
        return "reservation_only"
    return "both_flexible"


def _select_constrained(search: pd.DataFrame, native_horizon: int) -> dict[str, Any]:
    cells = _all_unique_candidates(search)
    group_cols = ["horizon_days", "Q", "window"]
    means = cells.groupby(group_cols, as_index=False).agg(
        average_utilization=("average_utilization", "mean"),
        priority_weighted_utilization=("priority_weighted_utilization", "mean"),
        class_2_percent_serviced=("class_2_percent_serviced", "mean"),
    )
    baseline_c2 = float(
        cells[
            (cells["horizon_days"] == native_horizon)
            & (cells["Q"] == 0)
        ]["class_2_percent_serviced"].mean()
    )
    max_average = float(means["average_utilization"].max())
    eligible = means[
        (means["average_utilization"] >= max_average - PRACTICAL_TOLERANCE)
        & (
            means["class_2_percent_serviced"]
            >= baseline_c2 - CLASS2_CONSTRAINT_TOLERANCE
        )
    ].copy()
    feasible = not eligible.empty
    if feasible:
        best = eligible.sort_values(
            ["priority_weighted_utilization", "Q", "window", "horizon_days"],
            ascending=[False, True, True, True],
            kind="stable",
        ).iloc[0]
        failure_reason = ""
    else:
        # The two constraints can conflict. Preserve the background in the
        # output by falling back to baseline, while clearly marking that no
        # policy cell met both constraints in the search sample.
        baseline_rows = means[
            (means["horizon_days"] == native_horizon) & (means["Q"] == 0)
        ]
        if baseline_rows.empty:
            raise RuntimeError("Baseline cell missing from constrained selection")
        best = baseline_rows.iloc[0]
        failure_reason = "no_cell_met_both_constraints"
    h, q, w = int(best.horizon_days), int(best.Q), int(best.window)
    return {
        "selected_horizon_days": h,
        "selected_Q": q,
        "selected_window": w,
        "selected_policy": _canonical_policy(native_horizon, h, q),
        "search_constraints_feasible": bool(feasible),
        "search_failure_reason": failure_reason,
        "search_average_utilization": float(best.average_utilization),
        "search_priority_weighted_utilization": float(best.priority_weighted_utilization),
        "search_class_2_percent_serviced": float(best.class_2_percent_serviced),
        "search_max_average_utilization": max_average,
        "search_baseline_class_2_percent_serviced": baseline_c2,
    }


def select_cells(
    *,
    bank_path: Path,
    output_dir: Path,
    n_seeds: int,
    smoke: bool,
) -> None:
    if not 1 <= n_seeds <= len(SEARCH_SEED_POOL):
        raise ValueError("Invalid number of search seeds for selection")
    seeds = SEARCH_SEED_POOL[: (2 if smoke else n_seeds)]
    bank = load_bank(bank_path).set_index("background_id")
    raw_dir = output_dir / "search" / "raw"
    shards = sorted(raw_dir.glob("*.csv"))
    if not shards:
        raise FileNotFoundError(f"No search shards found under {raw_dir}")

    selected_rows: list[dict[str, Any]] = []
    constrained_rows: list[dict[str, Any]] = []
    for index, path in enumerate(shards, start=1):
        search = pd.read_csv(path)
        background_id = str(search["source_background_id"].iloc[0])
        bank_row = bank.loc[background_id].copy()
        bank_row["background_id"] = background_id
        _validate_search_complete(bank_row, search, seeds=seeds, smoke=smoke)
        for objective in OBJECTIVES:
            for policy in POLICIES:
                selection = _select_cell(_policy_candidates(search, policy), objective)
                selected_rows.append(
                    {
                        "background_id": background_id,
                        "selection_objective": objective,
                        "policy": policy,
                        **selection.to_dict(),
                    }
                )
        constrained_rows.append(
            {
                "background_id": background_id,
                **_select_constrained(search, int(bank_row["horizon_days"])),
            }
        )
        if index % 100 == 0 or index == len(shards):
            print(f"Selected {index:,}/{len(shards):,} backgrounds")

    selection_dir = output_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.DataFrame(selected_rows)
    constrained = pd.DataFrame(constrained_rows)
    selected.to_csv(selection_dir / "selected_cells.csv", index=False)
    constrained.to_csv(selection_dir / "constrained_priority_cells.csv", index=False)
    print(f"Selected policy cells: {len(selected):,}")
    print(f"Constrained-priority cells: {len(constrained):,}")


def _evaluation_cells_for_background(
    selected: pd.DataFrame,
    constrained: pd.DataFrame,
    background_id: str,
) -> list[tuple[int, int, int]]:
    subset = selected[selected["background_id"] == background_id]
    cells = {
        (int(row.selected_horizon_days), int(row.selected_Q), int(row.selected_window))
        for row in subset.itertuples(index=False)
    }
    crow = constrained[constrained["background_id"] == background_id]
    if len(crow) != 1:
        raise ValueError(f"Expected one constrained row for {background_id}")
    record = crow.iloc[0]
    cells.add(
        (int(record["selected_horizon_days"]), int(record["selected_Q"]), int(record["selected_window"]))
    )
    return sorted(cells)


def run_evaluation(
    *,
    bank_path: Path,
    output_dir: Path,
    workers: int,
    n_seeds: int,
    shard_index: int,
    shard_count: int,
    smoke: bool,
) -> None:
    if not 1 <= n_seeds <= len(EVALUATION_SEED_POOL):
        raise ValueError("Invalid number of evaluation seeds")
    seeds = EVALUATION_SEED_POOL[: (2 if smoke else n_seeds)]
    bank = load_bank(bank_path)
    rows = _sharded_rows(
        bank, shard_index=shard_index, shard_count=shard_count, smoke=smoke
    )
    selection_dir = output_dir / "selection"
    selected = pd.read_csv(selection_dir / "selected_cells.csv")
    constrained = pd.read_csv(selection_dir / "constrained_priority_cells.csv")
    raw_dir = output_dir / "evaluation" / "raw"

    for number, (_, row) in enumerate(rows.iterrows(), start=1):
        background_id = str(row["background_id"])
        path = shard_path(raw_dir, background_id)
        cells = _evaluation_cells_for_background(selected, constrained, background_id)
        tasks = _task_grid(
            row,
            stage="evaluation",
            phase="independent_evaluation",
            cells=cells,
            seeds=seeds,
            smoke=smoke,
        )
        pending = _filter_pending(tasks, path, EVALUATION_KEY_COLUMNS)
        print(
            f"Evaluation {number:,}/{len(rows):,}: {background_id}; "
            f"{len(cells)} cells, {len(pending):,} pending runs"
        )
        run_tasks(pending, path, workers)

    print(f"Evaluation output: {raw_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("search", "select", "evaluate"))
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-seeds", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "search":
        run_search(
            bank_path=args.bank,
            output_dir=args.output_dir,
            workers=args.workers,
            n_seeds=args.n_seeds or 5,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            smoke=args.smoke,
        )
    elif args.command == "select":
        select_cells(
            bank_path=args.bank,
            output_dir=args.output_dir,
            n_seeds=args.n_seeds or 5,
            smoke=args.smoke,
        )
    else:
        run_evaluation(
            bank_path=args.bank,
            output_dir=args.output_dir,
            workers=args.workers,
            n_seeds=args.n_seeds or 10,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            smoke=args.smoke,
        )


if __name__ == "__main__":
    main()
