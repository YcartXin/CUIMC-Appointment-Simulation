"""Hypothesis 1: short-horizon reservation for the shorter-no-show-threshold class.

Claim: when Class 1 has a shorter no-show threshold than Class 2, reserving
slots for Class 1 only in near-term horizon days (a reserved_window_days
bounded reservation, not the whole calendar) raises utilization by keeping
more of Class 1's offered delays under their own no-show threshold.

This consumes the shared background-scenario bank from
experiments/hypothesis_scenario_bank.py, run against every background in
the bank (no pre-filtering to backgrounds that satisfy H1's stated
condition -- the condition itself is one of the things earlier stages of
this project tested, not assumed).

WHAT THIS SCRIPT ANSWERS
-------------------------
For each background, and separately for each of two reservation-release
variants (--variant strict|release, see below), this finds the optimal H1
policy under four different flexibility regimes and compares them:

    baseline           No reservation, booking horizon fixed at the
                        background's own bank-assigned value. The plain
                        FCFS outcome -- no policy intervention at all.
    horizon_only        No reservation (Q=0), but booking horizon is
                        swept across H1_HORIZON_VALUES to find the best
                        horizon alone.
    reservation_only    Booking horizon fixed at the background's own
                        bank-assigned value, but Q and the reservation
                        window are optimized (via the coarse-to-fine
                        search described below).
    both_flexible       Booking horizon, Q, and window are all jointly
                        optimized -- horizon swept across
                        H1_HORIZON_VALUES, with a coarse-to-fine (Q,
                        window) search performed at each horizon.

Both_flexible's search space weakly contains the other three regimes'
(same horizon values, same Q/window ranges, Q=0 always a candidate via
horizon_only's own rows -- see "Sharing Q=0 across conditions" below), so
its optimum should never be meaningfully worse than the other three. The
summary step reports how often that dominance actually holds as a sanity
check on the search itself.

The optimization objective is selectable with --objective. By default,
weighted_utilization is used for backward compatibility; use
--objective average_utilization to choose policies that maximize completed
appointments as a share of measured capacity. classify() reports each
condition's optimum under the selected objective, both utilization metrics
at that optimum, and all six policy comparisons among baseline,
horizon_only, reservation_only, and both_flexible.

STRICT VS RELEASE VARIANTS
----------------------------
--variant strict    Reserved capacity is never available to any class
                     other than Class 1, at any residual day (the
                     original reservation behavior).
--variant release    At residual day r = 0 (the day of service) only,
                     idle reserved capacity is pooled with general
                     capacity, so Class 2 can take a reserved slot Class
                     1 hasn't filled -- see
                     SimulationConfig.release_unused_reservation_same_day.

Both variants run with same_day_cancellation_enabled=True uniformly --
across every condition, including baseline -- so that same-day
cancellations (which is what makes idle reserved capacity actually
possible to observe and release on the day of, on top of Class 1 simply
not showing up) are a constant background capability in both variants,
and the only thing that differs between a strict run and a release run is
whether idle reserved capacity actually gets released. Run this script
twice, once per --variant, as two independent passes (e.g. two separate
grid job submissions); each pass's output is fully self-contained.

Both balk and no-show thresholds are dynamically capped at horizon_days -
1 wherever a config is built (see hypothesis_common.build_config), so
every threshold is guaranteed to sit within whatever horizon is in play.

COARSE-TO-FINE (Q, WINDOW) SEARCH
------------------------------------
Q now spans 1..capacity in steps of 1 (up to 50 values) and window spans
1..horizon in steps of 1 -- searching that full grid at every horizon,
for every background, in both variants, is roughly 7,850 simulations per
background per seed (see the design discussion this was decided in).
Instead, each (Q, window) search is done in two phases:

    coarse    Q at steps of Q_COARSE_STEP, window at steps of
              WINDOW_COARSE_STEP, evaluated jointly (every coarse Q
              paired with every coarse window).
    fine      A "+"-shaped refinement around the coarse winner: Q at
              step 1 within Q_REFINE_RADIUS of the winning Q (holding
              window at its coarse-winning value), and window at step 1
              within WINDOW_REFINE_RADIUS of the winning window (holding
              Q at its coarse-winning value). Not a full 2-D refine grid
              (that would multiply the two radii together) -- this is
              coordinate-wise, which is cheaper and still recovers full
              step-1 resolution in the neighborhood immediately around
              the coarse winner, since each radius exactly matches its
              dimension's coarse step size.

The coarse and fine phases both simply get logged as additional
evaluated cells; classify() takes the true argmax over every cell
actually evaluated (coarse union fine) using the selected --objective.
Both metrics are recorded at every cell. Re-running with a different
objective against the same raw-output directory reuses all exact and
coarse cells and adds only any missing fine cells around the new coarse
winner.

If Q = 0 wins in the coarse phase (i.e. no reservation already beats
every coarse Q > 0 candidate), the fine phase is skipped for that
condition/horizon entirely -- there's no local neighborhood to refine
around a "no reservation" answer.

SHARING Q=0 ACROSS CONDITIONS
--------------------------------
reservation_only's own Q=0 cell is never separately simulated: baseline
IS exactly reservation_only's Q=0 case (same horizon, same everything),
so baseline's rows are folded in as reservation_only's Q=0 candidate at
classification time. Likewise, both_flexible never separately simulates
Q=0 at each horizon: horizon_only's own per-horizon rows (Q=0, horizon
swept) are folded in as both_flexible's Q=0 candidates at each horizon.
This avoids redundant simulation while still letting every condition's
search legitimately discover "no reservation is best here."

SCALE AND STORAGE
--------------------
Raw output is sharded one CSV per background (raw/{background_id}.csv)
rather than one monolithic file, since a full run across the entire bank
is tens of millions of rows -- a single flat file would make every
resume check reload the whole dataset. Resuming is evaluated per shard
(cheap: each shard is a few thousand rows), so an interrupted run only
re-derives work for backgrounds it hadn't finished.

Run from the repository root:

    python experiments/hypothesis_scenario_bank.py           # build the bank once
    python experiments/h1_short_horizon_reservation.py all --variant strict
    python experiments/h1_short_horizon_reservation.py all --variant release

Use --smoke for a fast end-to-end check before a full run.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_DIR = Path(__file__).resolve().parents[1]
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

from experiments.hypothesis_common import (  # noqa: E402
    PRACTICAL_TOLERANCE,
    STAGE1_SEEDS,
    WEIGHTED_UTILIZATION_W1,
    WEIGHTED_UTILIZATION_W2,
    classify_effect,
    default_workers,
    load_completed_keys,
    paired_delta_ci,
    run_one,
    write_markdown,
)
from experiments.hypothesis_scenario_bank import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_BANK_PATH,
    HORIZON_VALUES,
    generate_background_bank,
)

REPO_DIR = _REPO_DIR
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "hypotheses" / "h1_short_horizon_reservation"

KEY_COLUMNS = ["stage", "background_id", "arm", "seed"]

H1_HORIZON_VALUES = HORIZON_VALUES

VARIANT_STRICT = "strict"
VARIANT_RELEASE = "release"
VARIANTS = (VARIANT_STRICT, VARIANT_RELEASE)

STAGE_BASELINE = "baseline"
STAGE_HORIZON_ONLY = "horizon_only"
STAGE_RESERVATION_ONLY = "reservation_only"
STAGE_BOTH_FLEXIBLE = "both_flexible"
STAGES = (STAGE_BASELINE, STAGE_HORIZON_ONLY, STAGE_RESERVATION_ONLY, STAGE_BOTH_FLEXIBLE)

PHASE_EXACT = "exact"
PHASE_COARSE = "coarse"
PHASE_FINE = "fine"

# Coarse-to-fine search resolution. The fine radius on each dimension
# exactly matches that dimension's coarse step, so coarse+fine together
# give full step-1 coverage in the neighborhood immediately surrounding
# the coarse winner, right out to its two nearest coarse neighbors.
Q_COARSE_STEP = 5
WINDOW_COARSE_STEP = 2
Q_REFINE_RADIUS = 5
WINDOW_REFINE_RADIUS = 2

VALUE_COLS: dict[str, tuple[str, str | None]] = {
    "average_utilization": ("average_utilization", "positive"),
    "weighted_utilization": ("weighted_utilization", "positive"),
}
OPTIMIZATION_OBJECTIVES = tuple(VALUE_COLS)
DEFAULT_OPTIMIZATION_OBJECTIVE = "weighted_utilization"


def _validate_objective(objective: str) -> str:
    """Validate and return the metric used to choose policy optima."""
    if objective not in OPTIMIZATION_OBJECTIVES:
        raise ValueError(
            f"objective must be one of {OPTIMIZATION_OBJECTIVES}, got {objective!r}"
        )
    return objective


def load_bank(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Background bank not found: {path}. Run "
            "`python experiments/hypothesis_scenario_bank.py` first."
        )
    return pd.read_csv(path)


def _row_config_kwargs(row: pd.Series) -> dict[str, Any]:
    """Every background parameter except horizon_days.

    horizon_days is supplied separately by callers, since every H1
    condition here treats it as either fixed-at-native or swept, never
    read implicitly from the row.
    """
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
    }


def _reservation_kwargs(q: int, window: int, variant: str) -> dict[str, Any]:
    """Reservation-related SimulationConfig kwargs for a given Q.

    Q = 0 always means "no reservation" regardless of variant (release
    only matters once there's reserved capacity to release). same_day_
    cancellation_enabled is True unconditionally, in every condition and
    every variant, per the module docstring.
    """
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
        "release_unused_reservation_same_day": variant == VARIANT_RELEASE,
    }


# Number of seeds per cell for non-smoke runs. Settable via --n-seeds
# (see set_n_seeds); always a PREFIX of STAGE1_SEEDS, so rows already
# simulated under a larger seed count remain valid, resumable work --
# a 10-seed run's keys are a strict subset of a 20-seed run's keys.
_ACTIVE_N_SEEDS = len(STAGE1_SEEDS)

# Fixed shuffle seed for background processing order (see run()). The
# bank is stored sorted by horizon stratum, so processing it in file
# order would leave any partially-completed run biased toward short
# horizons; a deterministic shuffle makes any prefix (or any
# --shard-index subset) a representative sample of the whole bank,
# while staying identical across jobs and reruns so resume/sharding
# stay consistent.
RUN_ORDER_SEED = 20260720


def set_n_seeds(n: int) -> None:
    global _ACTIVE_N_SEEDS
    n = int(n)
    if not (1 <= n <= len(STAGE1_SEEDS)):
        raise ValueError(
            f"--n-seeds must be between 1 and {len(STAGE1_SEEDS)}, got {n}"
        )
    _ACTIVE_N_SEEDS = n


def _seeds(smoke: bool) -> tuple[int, ...]:
    return STAGE1_SEEDS[:2] if smoke else STAGE1_SEEDS[:_ACTIVE_N_SEEDS]


def _smoke_overrides(smoke: bool) -> dict[str, Any]:
    if not smoke:
        return {}
    return {"burn_in_days": 5, "measure_days": 20, "cooldown_days": 5}


def _smoke_bank(bank: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return bank.sample(n=min(n, len(bank)), random_state=0)


# ---------------------------------------------------------------------
# Coarse-to-fine (Q, window) grid helpers
# ---------------------------------------------------------------------

def q_coarse_grid(capacity: int) -> list[int]:
    """Coarse Q candidates, strictly positive: Q_COARSE_STEP, 2 *
    Q_COARSE_STEP, ..., always including capacity itself as the extreme
    point. Q = 0 is never included here -- it's supplied by folding in
    baseline/horizon_only's own rows at classification time instead (see
    the module docstring's "Sharing Q=0 across conditions").
    """
    capacity = int(capacity)
    values = list(range(Q_COARSE_STEP, capacity + 1, Q_COARSE_STEP))
    if not values or values[-1] != capacity:
        values.append(capacity)
    return values


def window_coarse_grid(horizon: int) -> list[int]:
    """Coarse window candidates: 1, 1 + WINDOW_COARSE_STEP, ..., always
    including horizon itself.
    """
    horizon = int(horizon)
    values = list(range(1, horizon + 1, WINDOW_COARSE_STEP))
    if values[-1] != horizon:
        values.append(horizon)
    return values


def q_fine_grid(best_q: int, capacity: int) -> list[int]:
    """Fine Q candidates around a coarse winner: every integer within
    Q_REFINE_RADIUS, clipped to [1, capacity], excluding best_q itself
    (already evaluated in the coarse phase). Empty if best_q <= 0 -- a
    "no reservation" coarse winner has no neighborhood to refine.
    """
    best_q = int(best_q)
    if best_q <= 0:
        return []
    lo = max(1, best_q - Q_REFINE_RADIUS)
    hi = min(int(capacity), best_q + Q_REFINE_RADIUS)
    return [q for q in range(lo, hi + 1) if q != best_q]


def window_fine_grid(best_window: int, horizon: int) -> list[int]:
    """Fine window candidates around a coarse winner, same shape as
    q_fine_grid. best_window <= 0 should never occur (window is only
    meaningful when Q > 0, and callers skip fine search entirely when
    the coarse winner was Q = 0), but is handled defensively.
    """
    best_window = int(best_window)
    if best_window <= 0:
        return []
    lo = max(1, best_window - WINDOW_REFINE_RADIUS)
    hi = min(int(horizon), best_window + WINDOW_REFINE_RADIUS)
    return [w for w in range(lo, hi + 1) if w != best_window]


# ---------------------------------------------------------------------
# Per-condition task generation
# ---------------------------------------------------------------------

def _make_task(
    *,
    base_kwargs: dict[str, Any],
    horizon: int,
    q: int,
    window: int,
    variant: str,
    smoke: bool,
    stage: str,
    phase: str,
    background_id: str,
    source_background_id: str,
    seed: int,
) -> dict[str, Any]:
    config_kwargs = {
        **base_kwargs,
        "horizon_days": horizon,
        **_reservation_kwargs(q, window, variant),
        **_smoke_overrides(smoke),
    }
    return {
        "config_kwargs": config_kwargs,
        "seed": seed,
        "extra_cols": {
            "stage": stage,
            "arm": phase,
            "background_id": background_id,
            "source_background_id": source_background_id,
            "seed": seed,
            "variant": variant,
            "horizon_days": horizon,
            "Q": q,
            "window": window,
        },
    }


def baseline_tasks(row: pd.Series, variant: str, smoke: bool) -> list[dict[str, Any]]:
    bg = row["background_id"]
    horizon = int(row["horizon_days"])
    base_kwargs = _row_config_kwargs(row)
    return [
        _make_task(
            base_kwargs=base_kwargs,
            horizon=horizon,
            q=0,
            window=-1,
            variant=variant,
            smoke=smoke,
            stage=STAGE_BASELINE,
            phase=PHASE_EXACT,
            background_id=f"{bg}_baseline",
            source_background_id=bg,
            seed=seed,
        )
        for seed in _seeds(smoke)
    ]


def horizon_only_tasks(row: pd.Series, variant: str, smoke: bool) -> list[dict[str, Any]]:
    bg = row["background_id"]
    base_kwargs = _row_config_kwargs(row)
    horizons = H1_HORIZON_VALUES[:2] if smoke else H1_HORIZON_VALUES
    tasks = []
    for horizon in horizons:
        for seed in _seeds(smoke):
            tasks.append(
                _make_task(
                    base_kwargs=base_kwargs,
                    horizon=horizon,
                    q=0,
                    window=-1,
                    variant=variant,
                    smoke=smoke,
                    stage=STAGE_HORIZON_ONLY,
                    phase=PHASE_EXACT,
                    background_id=f"{bg}_honly_H={horizon}",
                    source_background_id=bg,
                    seed=seed,
                )
            )
    return tasks


def reservation_only_coarse_tasks(row: pd.Series, variant: str, smoke: bool) -> list[dict[str, Any]]:
    bg = row["background_id"]
    horizon = int(row["horizon_days"])
    base_kwargs = _row_config_kwargs(row)
    q_values = q_coarse_grid(int(row["slots_per_day"]))
    window_values = window_coarse_grid(horizon)
    if smoke:
        q_values, window_values = q_values[:2], window_values[:2]

    tasks = []
    for q in q_values:
        for window in window_values:
            for seed in _seeds(smoke):
                tasks.append(
                    _make_task(
                        base_kwargs=base_kwargs,
                        horizon=horizon,
                        q=q,
                        window=window,
                        variant=variant,
                        smoke=smoke,
                        stage=STAGE_RESERVATION_ONLY,
                        phase=PHASE_COARSE,
                        background_id=f"{bg}_resv_Q={q}_w={window}",
                        source_background_id=bg,
                        seed=seed,
                    )
                )
    return tasks


def reservation_only_fine_tasks(
    row: pd.Series, variant: str, smoke: bool, *, best_q: int, best_window: int
) -> list[dict[str, Any]]:
    bg = row["background_id"]
    horizon = int(row["horizon_days"])
    base_kwargs = _row_config_kwargs(row)
    capacity = int(row["slots_per_day"])

    tasks = []
    for q in q_fine_grid(best_q, capacity):
        for seed in _seeds(smoke):
            tasks.append(
                _make_task(
                    base_kwargs=base_kwargs,
                    horizon=horizon,
                    q=q,
                    window=best_window,
                    variant=variant,
                    smoke=smoke,
                    stage=STAGE_RESERVATION_ONLY,
                    phase=PHASE_FINE,
                    background_id=f"{bg}_resv_Q={q}_w={best_window}",
                    source_background_id=bg,
                    seed=seed,
                )
            )
    for window in window_fine_grid(best_window, horizon):
        for seed in _seeds(smoke):
            tasks.append(
                _make_task(
                    base_kwargs=base_kwargs,
                    horizon=horizon,
                    q=best_q,
                    window=window,
                    variant=variant,
                    smoke=smoke,
                    stage=STAGE_RESERVATION_ONLY,
                    phase=PHASE_FINE,
                    background_id=f"{bg}_resv_Q={best_q}_w={window}",
                    source_background_id=bg,
                    seed=seed,
                )
            )
    return tasks


def both_flexible_coarse_tasks(row: pd.Series, variant: str, smoke: bool) -> list[dict[str, Any]]:
    bg = row["background_id"]
    base_kwargs = _row_config_kwargs(row)
    capacity = int(row["slots_per_day"])
    horizons = H1_HORIZON_VALUES[:2] if smoke else H1_HORIZON_VALUES

    tasks = []
    for horizon in horizons:
        q_values = q_coarse_grid(capacity)
        window_values = window_coarse_grid(horizon)
        if smoke:
            q_values, window_values = q_values[:2], window_values[:2]
        for q in q_values:
            for window in window_values:
                for seed in _seeds(smoke):
                    tasks.append(
                        _make_task(
                            base_kwargs=base_kwargs,
                            horizon=horizon,
                            q=q,
                            window=window,
                            variant=variant,
                            smoke=smoke,
                            stage=STAGE_BOTH_FLEXIBLE,
                            phase=PHASE_COARSE,
                            background_id=f"{bg}_both_H={horizon}_Q={q}_w={window}",
                            source_background_id=bg,
                            seed=seed,
                        )
                    )
    return tasks


def both_flexible_fine_tasks(
    row: pd.Series, variant: str, smoke: bool, *, winners: dict[int, tuple[int, int]]
) -> list[dict[str, Any]]:
    """winners maps horizon -> (best_q, best_window) found in the coarse
    phase for that horizon (only for horizons whose coarse winner had
    Q > 0; horizons resolved to Q = 0 are skipped, same rule as
    reservation_only).
    """
    bg = row["background_id"]
    base_kwargs = _row_config_kwargs(row)
    capacity = int(row["slots_per_day"])

    tasks = []
    for horizon, (best_q, best_window) in winners.items():
        for q in q_fine_grid(best_q, capacity):
            for seed in _seeds(smoke):
                tasks.append(
                    _make_task(
                        base_kwargs=base_kwargs,
                        horizon=horizon,
                        q=q,
                        window=best_window,
                        variant=variant,
                        smoke=smoke,
                        stage=STAGE_BOTH_FLEXIBLE,
                        phase=PHASE_FINE,
                        background_id=f"{bg}_both_H={horizon}_Q={q}_w={best_window}",
                        source_background_id=bg,
                        seed=seed,
                    )
                )
        for window in window_fine_grid(best_window, horizon):
            for seed in _seeds(smoke):
                tasks.append(
                    _make_task(
                        base_kwargs=base_kwargs,
                        horizon=horizon,
                        q=best_q,
                        window=window,
                        variant=variant,
                        smoke=smoke,
                        stage=STAGE_BOTH_FLEXIBLE,
                        phase=PHASE_FINE,
                        background_id=f"{bg}_both_H={horizon}_Q={best_q}_w={window}",
                        source_background_id=bg,
                        seed=seed,
                    )
                )
    return tasks


# ---------------------------------------------------------------------
# Sharded resumable execution
# ---------------------------------------------------------------------

def shard_path(raw_dir: Path, background_id: str) -> Path:
    return raw_dir / f"{background_id}.csv"


def load_shard_completed_keys(raw_dir: Path, background_id: str) -> set[tuple[Any, ...]]:
    return load_completed_keys(shard_path(raw_dir, background_id), KEY_COLUMNS)


def _append_shard(raw_dir: Path, background_id: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = shard_path(raw_dir, background_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_sharded_tasks(
    tasks: list[dict[str, Any]], *, raw_dir: Path, workers: int, flush_every: int = 500
) -> None:
    """Like hypothesis_common.run_tasks, but routes each result row into
    a per-source_background_id shard file instead of one monolithic
    file. At tens of millions of total rows, a single flat CSV would
    make every resume check reload the entire dataset; sharding by
    background keeps each resume check (and each downstream classify
    read) down to a few thousand rows at a time.
    """
    if not tasks:
        print("Nothing to run: all requested tasks are already completed.")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    executor = None
    try:
        if workers <= 1:
            iterator = map(run_one, tasks)
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            iterator = executor.map(run_one, tasks, chunksize=4)

        for index, row in enumerate(iterator, start=1):
            bg = row["source_background_id"]
            buffers[bg].append(row)
            if len(buffers[bg]) >= flush_every:
                _append_shard(raw_dir, bg, buffers[bg])
                buffers[bg] = []
            if index % 5000 == 0 or index == len(tasks):
                print(f"Completed {index:,}/{len(tasks):,} new runs")
        for bg, rows in buffers.items():
            _append_shard(raw_dir, bg, rows)
    finally:
        if executor is not None:
            executor.shutdown()


def _filter_pending(tasks: list[dict[str, Any]], raw_dir: Path) -> list[dict[str, Any]]:
    completed_by_bg: dict[str, set[tuple[Any, ...]]] = {}
    pending = []
    for t in tasks:
        bg = t["extra_cols"]["source_background_id"]
        if bg not in completed_by_bg:
            completed_by_bg[bg] = load_shard_completed_keys(raw_dir, bg)
        key = tuple(t["extra_cols"][c] for c in KEY_COLUMNS)
        if key not in completed_by_bg[bg]:
            pending.append(t)
    return pending


# ---------------------------------------------------------------------
# Run: two batches, coarse+exact first, then fine
# ---------------------------------------------------------------------

def run(
    *,
    variant: str,
    bank_path: Path,
    output_dir: Path,
    workers: int,
    smoke: bool,
    resume: bool,
    n_seeds: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    objective: str = DEFAULT_OPTIMIZATION_OBJECTIVE,
) -> Path:
    objective = _validate_objective(objective)
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if shard_count < 1:
        raise ValueError(f"--shard-count must be >= 1, got {shard_count}")
    if not (0 <= shard_index < shard_count):
        raise ValueError(
            f"--shard-index must be in [0, {shard_count - 1}], got {shard_index}"
        )
    if n_seeds is not None:
        set_n_seeds(n_seeds)

    bank = load_bank(bank_path)
    rows = _smoke_bank(bank) if smoke else bank

    # Deterministic shuffle, then stride-slice for this job's shard. The
    # shuffle (fixed RUN_ORDER_SEED, identical across jobs/reruns) makes
    # any prefix of any shard a representative sample of the whole bank
    # rather than a run of same-horizon rows; the stride slice gives each
    # of N concurrent jobs a disjoint background subset, so multiple
    # invocations can safely share one output tree (shard files are
    # per-background, so disjoint backgrounds means no write collisions).
    rows = rows.sample(frac=1.0, random_state=RUN_ORDER_SEED).reset_index(drop=True)
    rows = rows.iloc[shard_index::shard_count]

    raw_dir = output_dir / variant / "raw"

    if not resume and raw_dir.exists():
        # Scope deletion to THIS job's backgrounds only: with concurrent
        # sharded jobs sharing one output tree, deleting every shard file
        # here would destroy other jobs' completed work.
        for bg in rows["background_id"]:
            shard = shard_path(raw_dir, bg)
            if shard.exists():
                shard.unlink()

    # Process one background at a time. The original implementation built
    # every task for every background in one in-memory list; the full bank
    # creates millions of nested task dictionaries and can exhaust RAM before
    # the first simulation starts. Per-background processing preserves the
    # existing shard/resume behavior while bounding memory use -- and,
    # combined with the shuffled order above, means an interrupted or
    # deadline-truncated run leaves behind fully-completed, representative
    # backgrounds that classify() can use as-is.
    shard_label = f" shard {shard_index + 1}/{shard_count}" if shard_count > 1 else ""
    run_started = time.monotonic()
    for row_number, (_, row) in enumerate(rows.iterrows(), start=1):
        bg = row["background_id"]
        elapsed = time.monotonic() - run_started
        if row_number > 1 and elapsed > 0:
            per_bg = elapsed / (row_number - 1)
            remaining_h = per_bg * (len(rows) - row_number + 1) / 3600
            eta = f"; ~{per_bg / 60:.1f} min/bg, est {remaining_h:.1f}h remaining"
        else:
            eta = ""
        print(f"\nH1 [{variant}]{shard_label} background {row_number:,}/{len(rows):,}: {bg}{eta}")

        # Batch 1: baseline + horizon_only (exact) + reservation_only and
        # both_flexible coarse phases for this background only.
        batch1: list[dict[str, Any]] = []
        batch1.extend(baseline_tasks(row, variant, smoke))
        batch1.extend(horizon_only_tasks(row, variant, smoke))
        batch1.extend(reservation_only_coarse_tasks(row, variant, smoke))
        batch1.extend(both_flexible_coarse_tasks(row, variant, smoke))

        pending1 = _filter_pending(batch1, raw_dir) if resume else batch1
        print(
            "  batch 1 (baseline/horizon_only/coarse): "
            f"total={len(batch1):,}; "
            f"completed={len(batch1) - len(pending1):,}; "
            f"to run={len(pending1):,}"
        )
        run_sharded_tasks(pending1, raw_dir=raw_dir, workers=workers)

        # Batch 2: derive fine-phase winners from this background's now-complete
        # coarse shard, then run only this background's fine tasks.
        shard = shard_path(raw_dir, bg)
        if not shard.exists():
            print(f"  No shard found for {bg}; skipping fine phase.")
            continue

        shard_df = pd.read_csv(shard)
        batch2: list[dict[str, Any]] = []

        resv_coarse = shard_df[
            (shard_df["stage"] == STAGE_RESERVATION_ONLY)
            & (shard_df["arm"] == PHASE_COARSE)
        ]
        baseline_zero = shard_df[shard_df["stage"] == STAGE_BASELINE]

        best_q, best_window = _best_qw(
            pd.concat([resv_coarse, baseline_zero], ignore_index=True),
            objective,
        )
        if best_q is not None and best_q > 0:
            batch2.extend(
                reservation_only_fine_tasks(
                    row, variant, smoke, best_q=best_q, best_window=best_window
                )
            )

        both_coarse = shard_df[
            (shard_df["stage"] == STAGE_BOTH_FLEXIBLE)
            & (shard_df["arm"] == PHASE_COARSE)
        ]
        winners: dict[int, tuple[int, int]] = {}
        for horizon, group in both_coarse.groupby("horizon_days"):
            horizon_zero = shard_df[
                (shard_df["stage"] == STAGE_HORIZON_ONLY)
                & (shard_df["horizon_days"] == horizon)
            ]

            bq, bw = _best_qw(
                pd.concat([group, horizon_zero], ignore_index=True),
                objective,
            )
            if bq is not None and bq > 0:
                winners[int(horizon)] = (bq, bw)
        if winners:
            batch2.extend(both_flexible_fine_tasks(row, variant, smoke, winners=winners))

        pending2 = _filter_pending(batch2, raw_dir) if resume else batch2
        print(
            "  batch 2 (fine): "
            f"total={len(batch2):,}; "
            f"completed={len(batch2) - len(pending2):,}; "
            f"to run={len(pending2):,}"
        )
        run_sharded_tasks(pending2, raw_dir=raw_dir, workers=workers)

    print(f"Raw results (sharded): {raw_dir}")
    return raw_dir


def _best_qw(
    cells: pd.DataFrame,
    objective: str = DEFAULT_OPTIMIZATION_OBJECTIVE,
) -> tuple[int | None, int | None]:
    """Select the coarse (Q, window) winner using the requested objective."""
    objective = _validate_objective(objective)
    if cells.empty:
        return None, None
    means = cells.groupby(["Q", "window"], as_index=False)[objective].mean()
    best = means.loc[means[objective].idxmax()]
    return int(best["Q"]), int(best["window"])


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------

def _condition_optimum(
    shard_df: pd.DataFrame,
    stage: str,
    *,
    objective: str = DEFAULT_OPTIMIZATION_OBJECTIVE,
    extra_zero_rows: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Return seed-level rows for the best evaluated policy cell.

    The winner is chosen by the mean requested objective across seeds.
    For horizon-swept conditions (horizon_only and both_flexible), the
    optimum is selected across the entire flexible horizon search space.
    """
    objective = _validate_objective(objective)
    cells = shard_df[shard_df["stage"] == stage]
    if extra_zero_rows is not None and not extra_zero_rows.empty:
        cells = pd.concat([cells, extra_zero_rows], ignore_index=True)
    if cells.empty:
        return None

    # The coarse and fine grids can evaluate the same policy cell under
    # different phase labels (for example, a neighboring coarse Q can also
    # appear in the fine refinement).  Keep one deterministic result per
    # policy cell and seed so duplicated phase rows do not overweight a cell
    # or create duplicate seed labels in paired comparisons.
    cells = cells.drop_duplicates(
        subset=["horizon_days", "Q", "window", "seed"],
        keep="first",
    )

    group_cols = ["horizon_days", "Q", "window"]
    means = cells.groupby(group_cols, as_index=False)[objective].mean()
    best = means.loc[means[objective].idxmax()]
    mask = (
        (cells["horizon_days"] == best["horizon_days"])
        & (cells["Q"] == best["Q"])
        & (cells["window"] == best["window"])
    )
    return cells[mask].sort_values("seed")


def _delta_row(
    label: str,
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    seed_key: tuple[Any, ...],
) -> dict[str, Any]:
    """Paired-seed bootstrap delta of condition a vs condition b, for
    both average_utilization and weighted_utilization.
    """
    # Collapse any repeated phase rows to one value per seed before pairing.
    # This is necessary because the coarse and fine grids can contain the same
    # (horizon, Q, window) cell, producing duplicate seed rows for an otherwise
    # identical policy.
    metric_columns = [column for column, _ in VALUE_COLS.values()]
    a_idx = a.groupby("seed", sort=True)[metric_columns].mean()
    b_idx = b.groupby("seed", sort=True)[metric_columns].mean()
    paired_seeds = sorted(set(a_idx.index) & set(b_idx.index))
    row: dict[str, Any] = {"comparison": label, "n_paired_seeds": len(paired_seeds)}
    if not paired_seeds:
        return row
    for metric, (column, expected_sign) in VALUE_COLS.items():
        mean, low, high, _ = paired_delta_ci(
            a_idx.loc[paired_seeds, column].tolist(),
            b_idx.loc[paired_seeds, column].tolist(),
            seed=abs(hash((*seed_key, label, metric))) % (2**31),
        )
        row[f"delta_{metric}"] = mean
        row[f"delta_{metric}_ci_low"] = low
        row[f"delta_{metric}_ci_high"] = high
        row[f"{metric}_status"] = classify_effect(mean, low, high, expected_sign=expected_sign)
    return row


def classify(
    *,
    output_dir: Path,
    bank_path: Path,
    variant: str,
    objective: str = DEFAULT_OPTIMIZATION_OBJECTIVE,
) -> None:
    objective = _validate_objective(objective)
    bank = load_bank(bank_path)
    raw_dir = output_dir / variant / "raw"
    summary_name = (
        "summary"
        if objective == DEFAULT_OPTIMIZATION_OBJECTIVE
        else f"summary_{objective}"
    )
    summary_dir = output_dir / variant / summary_name
    summary_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(raw_dir.glob("*.csv"))
    if not shards:
        print(f"No raw shards found under {raw_dir}; run the experiment first.")
        return

    optimum_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []

    for shard in shards:
        shard_df = pd.read_csv(shard)
        if shard_df.empty:
            continue
        background_id = str(shard_df["source_background_id"].iloc[0])

        baseline = shard_df[shard_df["stage"] == STAGE_BASELINE]
        horizon_only = _condition_optimum(
            shard_df,
            STAGE_HORIZON_ONLY,
            objective=objective,
        )
        reservation_only = _condition_optimum(
            shard_df,
            STAGE_RESERVATION_ONLY,
            objective=objective,
            extra_zero_rows=baseline,
        )
        both_flexible = _condition_optimum(
            shard_df,
            STAGE_BOTH_FLEXIBLE,
            objective=objective,
            extra_zero_rows=horizon_only,
        )

        conditions = {
            STAGE_BASELINE: baseline if not baseline.empty else None,
            STAGE_HORIZON_ONLY: horizon_only,
            STAGE_RESERVATION_ONLY: reservation_only,
            STAGE_BOTH_FLEXIBLE: both_flexible,
        }
        if any(v is None for v in conditions.values()):
            continue

        opt_row: dict[str, Any] = {
            "background_id": background_id,
            "variant": variant,
            "optimization_objective": objective,
        }
        for stage, cells in conditions.items():
            opt_row[f"{stage}_horizon_days"] = int(cells["horizon_days"].iloc[0])
            opt_row[f"{stage}_Q"] = int(cells["Q"].iloc[0])
            opt_row[f"{stage}_window"] = int(cells["window"].iloc[0])
            opt_row[f"{stage}_average_utilization"] = float(cells["average_utilization"].mean())
            opt_row[f"{stage}_weighted_utilization"] = float(cells["weighted_utilization"].mean())
            opt_row[f"{stage}_n_seeds"] = int(cells["seed"].nunique())
        optimum_rows.append(opt_row)

        if objective == DEFAULT_OPTIMIZATION_OBJECTIVE:
            # Preserve the original weighted-objective output contract so the
            # existing summary files and regression tests remain unchanged.
            comparisons = (
                (
                    "both_flexible_vs_baseline",
                    conditions[STAGE_BOTH_FLEXIBLE],
                    conditions[STAGE_BASELINE],
                ),
                (
                    "both_flexible_vs_reservation_only",
                    conditions[STAGE_BOTH_FLEXIBLE],
                    conditions[STAGE_RESERVATION_ONLY],
                ),
                (
                    "both_flexible_vs_horizon_only",
                    conditions[STAGE_BOTH_FLEXIBLE],
                    conditions[STAGE_HORIZON_ONLY],
                ),
            )
        else:
            comparisons = (
                (
                    "horizon_only_vs_baseline",
                    conditions[STAGE_HORIZON_ONLY],
                    conditions[STAGE_BASELINE],
                ),
                (
                    "reservation_only_vs_baseline",
                    conditions[STAGE_RESERVATION_ONLY],
                    conditions[STAGE_BASELINE],
                ),
                (
                    "both_flexible_vs_baseline",
                    conditions[STAGE_BOTH_FLEXIBLE],
                    conditions[STAGE_BASELINE],
                ),
                (
                    "both_flexible_vs_horizon_only",
                    conditions[STAGE_BOTH_FLEXIBLE],
                    conditions[STAGE_HORIZON_ONLY],
                ),
                (
                    "both_flexible_vs_reservation_only",
                    conditions[STAGE_BOTH_FLEXIBLE],
                    conditions[STAGE_RESERVATION_ONLY],
                ),
                (
                    "reservation_only_vs_horizon_only",
                    conditions[STAGE_RESERVATION_ONLY],
                    conditions[STAGE_HORIZON_ONLY],
                ),
            )

        for label, first, second in comparisons:
            drow = _delta_row(
                label,
                first,
                second,
                seed_key=(background_id, objective),
            )
            drow["background_id"] = background_id
            drow["variant"] = variant
            drow["optimization_objective"] = objective
            delta_rows.append(drow)

    optimum_table = pd.DataFrame(optimum_rows)
    delta_table = pd.DataFrame(delta_rows)
    if not optimum_table.empty:
        bank_cols = [
            "background_id",
            "horizon_days",
            "rho",
            "class1_share",
            "slots_per_day",
            "noshow_threshold_1",
            "noshow_threshold_2",
        ]
        optimum_table = optimum_table.merge(bank[bank_cols], on="background_id", how="left")

    optimum_table.to_csv(summary_dir / "condition_optima.csv", index=False)
    delta_table.to_csv(summary_dir / "condition_deltas.csv", index=False)

    print(f"Backgrounds classified: {len(optimum_table):,}")
    print(f"Condition optima: {summary_dir / 'condition_optima.csv'}")
    print(f"Condition deltas: {summary_dir / 'condition_deltas.csv'}")
    if not delta_table.empty:
        for metric in ("average_utilization", "weighted_utilization"):
            col = f"{metric}_status"
            if col in delta_table.columns:
                print(f"\n{metric} status by comparison:")
                print(
                    delta_table.groupby(["comparison", col]).size().rename("n_backgrounds").to_string()
                )

    _write_summary(optimum_table, delta_table, summary_dir, variant, objective)


def _write_summary(
    optimum_table: pd.DataFrame,
    delta_table: pd.DataFrame,
    summary_dir: Path,
    variant: str,
    objective: str,
) -> None:
    dominance_note = "n/a (no rows classified)"
    if not delta_table.empty:
        dominance_comparisons = {
            "both_flexible_vs_baseline",
            "both_flexible_vs_reservation_only",
            "both_flexible_vs_horizon_only",
        }
        dominance_rows = delta_table[
            delta_table["comparison"].isin(dominance_comparisons)
            & delta_table["delta_average_utilization"].notna()
        ]
        both_ge_all = dominance_rows.groupby("background_id")[
            "delta_average_utilization"
        ].min()
        share_dominant = float((both_ge_all >= -PRACTICAL_TOLERANCE).mean()) if len(both_ge_all) else float("nan")
        dominance_note = (
            f"{share_dominant:.1%} of backgrounds had both_flexible's average_utilization "
            f"at or above all three other conditions (within the {PRACTICAL_TOLERANCE} "
            "practical-equivalence tolerance) -- the expected weak-dominance property, "
            "since both_flexible's search space contains the other three."
        )
    delta_description = (
        "condition_deltas.csv has the original three both-flexible comparisons."
        if objective == DEFAULT_OPTIMIZATION_OBJECTIVE
        else "condition_deltas.csv has all six paired-seed-bootstrap policy comparisons."
    )
    lines = [
        f"# H1 Short-Horizon Reservation ({variant}): Summary",
        "",
        f"Optimization objective: {objective}",
        f"Practical-equivalence tolerance: {PRACTICAL_TOLERANCE}",
        f"Weighted-utilization weights: w1={WEIGHTED_UTILIZATION_W1}, w2={WEIGHTED_UTILIZATION_W2}",
        f"Backgrounds classified: {len(optimum_table):,}",
        "",
        "This is an auto-generated data summary, not the narrative report.",
        "condition_optima.csv has one row per background with each of the four",
        "conditions' optimal (horizon, Q, window) and both utilization metrics.",
        delta_description,
        "For reservation_only_vs_horizon_only, a positive delta means the",
        "reservation-only policy is higher; a negative delta means horizon-only is higher.",
        "",
        f"Dominance check: {dominance_note}",
    ]
    write_markdown(lines, summary_dir / "h1_summary.md")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "classify", "all"])
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument(
        "--objective",
        choices=OPTIMIZATION_OBJECTIVES,
        default=DEFAULT_OPTIMIZATION_OBJECTIVE,
        help=(
            "Metric used to choose coarse-search winners and final policy optima. "
            "Both metrics are still reported for the selected policy."
        ),
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=len(STAGE1_SEEDS),
        help=(
            "Seeds per cell (prefix of STAGE1_SEEDS). Lower = faster with "
            "wider CIs; rows already run at a higher seed count remain "
            "valid and are skipped on resume."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="This job's shard number, 0-based. See --shard-count.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help=(
            "Split the (shuffled) bank across N concurrent jobs; each job "
            "handles backgrounds where position %% N == its --shard-index. "
            "All jobs may share one --output-dir: shard files are "
            "per-background, so disjoint subsets never collide."
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.bank.exists():
        print(f"Background bank not found at {args.bank}; generating the default bank now.")
        bank = generate_background_bank()
        args.bank.parent.mkdir(parents=True, exist_ok=True)
        bank.to_csv(args.bank, index=False)

    if args.command in {"run", "all"}:
        run(
            variant=args.variant,
            bank_path=args.bank,
            output_dir=args.output_dir,
            workers=args.workers,
            smoke=args.smoke,
            resume=not args.no_resume,
            n_seeds=args.n_seeds,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            objective=args.objective,
        )
    if args.command in {"classify", "all"}:
        classify(
            output_dir=args.output_dir,
            bank_path=args.bank,
            variant=args.variant,
            objective=args.objective,
        )


if __name__ == "__main__":
    main()
