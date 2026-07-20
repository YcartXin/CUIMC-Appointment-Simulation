"""Hypothesis 1: short-horizon reservation for the shorter-no-show-threshold class.

Claim: when Class 1 has a shorter no-show threshold than Class 2, reserving
slots for Class 1 only in near-term horizon days (a reserved_window_days
bounded reservation, not the whole calendar) raises utilization by keeping
more of Class 1's offered delays under their own no-show threshold.

This consumes the shared background-scenario bank from
experiments/hypothesis_scenario_bank.py, which spans the full user-specified
range on every dimension (rho, horizon, class mix, capacity, cancellation,
balking, no-show), per-class asymmetric wherever that's physically
meaningful, without pre-filtering to backgrounds that satisfy H1's stated
condition. This is deliberate: the condition (threshold_1 < threshold_2) is
one of the things being tested, not assumed.

Booking horizon is treated as a genuine policy lever for H1 (clinics can
choose how far out to let patients book), not just a fixed background
fact -- see H1_HORIZON_VALUES. The reservation window is swept from 1 day
up to the entire booking horizon in steps of 1, replacing the old fixed
{3,4,5,6,7} candidate set, and both balk and no-show thresholds are
dynamically capped at horizon_days - 1 wherever a config is built (see
experiments/hypothesis_common.py's build_config), so every threshold is
guaranteed to sit within whatever horizon is in play for a given task.

Three stages:

    screen          Broad on/off test across every background in the
                    bank, at each background's own bank-assigned horizon
                    (horizon itself is NOT swept here, only the window --
                    see the module-level compute-scale note above
                    H1_HORIZON_VALUES). Q is fixed at STANDARD_Q; window
                    sweeps 1..horizon, and classify_screen picks each
                    background's own best window before comparing on vs
                    off. Answers whether the effect exists at all, and
                    whether it concentrates in backgrounds that satisfy
                    H1's stated condition or shows up elsewhere too.
    grid            Full (horizon, Q, window) grid -- horizon swept
                    across H1_HORIZON_VALUES, Q up to half of that
                    background's own capacity in steps of 5, window
                    swept 1..that horizon -- at a small curated set of 12
                    backgrounds spanning condition-satisfying and
                    condition-violating cases. Answers the optimal-vs-none
                    question per (background, horizon).
    condition_grid  The same full (horizon, Q, window) grid, but at a much
                    larger, condition-balanced background set (~50
                    backgrounds, all with rho > 1.2) instead of 12.
                    Purpose-built to answer "does the threshold-gap
                    condition help under each background's own OPTIMAL
                    policy" with real sample size, rather than the screen
                    stage's answer to the same question at a fixed Q.
                    Not part of the default "all" stage set -- request it
                    explicitly.

Run from the repository root:

    python experiments/hypothesis_scenario_bank.py          # build the bank once
    python experiments/h1_short_horizon_reservation.py all --stage all
    python experiments/h1_short_horizon_reservation.py classify --stage all
    python experiments/h1_short_horizon_reservation.py all --stage condition_grid

Use --smoke for a fast end-to-end check before a full run.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_DIR = Path(__file__).resolve().parents[1]
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

from experiments.hypothesis_common import (  # noqa: E402
    PRACTICAL_TOLERANCE,
    STAGE1_SEEDS,
    best_and_near_tie,
    classify_effect,
    default_workers,
    load_completed_keys,
    paired_delta_ci,
    run_tasks,
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

STANDARD_Q = 5

# Booking horizon is treated as its own tested policy lever in the grid and
# condition_grid stages (compared directly, holding a background's other
# parameters fixed), separate from whatever horizon_days each background
# happens to carry in the bank. The screen stage does NOT sweep horizon
# itself -- only the reservation window, at each background's own native
# horizon -- purely for compute-scale reasons: sweeping horizon too across
# all 480 screen backgrounds would multiply screen's task count roughly
# 30x (see the module docstring's stage descriptions).
H1_HORIZON_VALUES = HORIZON_VALUES
N_DEEP_BACKGROUNDS_PER_BUCKET = 3

# condition_grid stage: a larger, condition-balanced background set for
# testing the threshold-gap condition under each background's own optimal
# policy, rather than one fixed (Q, window). Only requires rho above the
# demand floor identified in the screen stage; no additional filtering on
# Class 1's own volume, since including Q=0 in the search already reveals
# when a background can't support any positive reservation.
CONDITION_GRID_MIN_RHO = 1.2
CONDITION_GRID_N_PER_BUCKET = 25
CONDITION_GRID_SEED = 20260713


def load_bank(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Background bank not found: {path}. Run "
            "`python experiments/hypothesis_scenario_bank.py` first."
        )
    return pd.read_csv(path)


def _row_config_kwargs(row: pd.Series) -> dict[str, Any]:
    """Every background parameter except horizon_days.

    horizon_days is deliberately left out here: H1 treats it as a tested
    policy lever supplied separately per task (see H1_HORIZON_VALUES),
    rather than reading whatever value the bank happened to assign to
    this row. Callers must add "horizon_days" to the returned dict
    themselves before passing it to build_config.
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


def _reservation_kwargs(on: bool, q: int, window: int) -> dict[str, Any]:
    if not on:
        return {"reserved_class_id": None, "reserved_slots_per_day": 0, "reserved_window_days": None}
    return {"reserved_class_id": 1, "reserved_slots_per_day": q, "reserved_window_days": window}


def _seeds(smoke: bool) -> tuple[int, ...]:
    return STAGE1_SEEDS[:2] if smoke else STAGE1_SEEDS


def _smoke_overrides(smoke: bool) -> dict[str, Any]:
    if not smoke:
        return {}
    return {"burn_in_days": 5, "measure_days": 20, "cooldown_days": 5}


def q_grid_for_capacity(capacity: int) -> list[int]:
    return list(range(5, capacity // 2 + 1, 5))


def window_grid_for_horizon(horizon_days: int) -> list[int]:
    """Reservation window candidates: 1 day up to the entire booking
    horizon, in steps of 1. Replaces the old fixed {3,4,5,6,7} (capped at
    Class 1's no-show threshold) now that the window is swept against
    whatever horizon is actually in play for a given task.
    """
    return list(range(1, int(horizon_days) + 1))


# ---------------------------------------------------------------------
# Screen: broad on/off test across the whole bank
# ---------------------------------------------------------------------

def screen_tasks(bank: pd.DataFrame, smoke: bool) -> list[dict[str, Any]]:
    """Broad on/off test across the whole bank.

    Horizon is NOT swept here (see H1_HORIZON_VALUES's comment above) --
    each background keeps its own bank-assigned horizon_days. What's new
    is the on-arm window sweep: instead of one fixed window=3, every
    background now gets an on-arm cell for every window from 1 to its own
    horizon_days, so classify_screen can pick each background's own best
    window before comparing on vs off (see classify_screen).
    """
    rows = bank.sample(n=min(40, len(bank)), random_state=0) if smoke else bank
    tasks = []
    for _, row in rows.iterrows():
        background_id = row["background_id"]
        horizon_days = int(row["horizon_days"])
        base_kwargs = {**_row_config_kwargs(row), "horizon_days": horizon_days}
        window_values = window_grid_for_horizon(horizon_days)
        if smoke:
            window_values = window_values[:2]

        for seed in _seeds(smoke):
            tasks.append(
                {
                    "config_kwargs": {
                        **base_kwargs,
                        **_reservation_kwargs(False, 0, 0),
                        **_smoke_overrides(smoke),
                    },
                    "seed": seed,
                    "extra_cols": {
                        "stage": "screen",
                        "background_id": background_id,
                        "arm": "off",
                        "seed": seed,
                        "source_background_id": background_id,
                        "horizon_days": horizon_days,
                        "Q": 0,
                        "window": -1,
                    },
                }
            )
        for window in window_values:
            for seed in _seeds(smoke):
                tasks.append(
                    {
                        "config_kwargs": {
                            **base_kwargs,
                            **_reservation_kwargs(True, STANDARD_Q, window),
                            **_smoke_overrides(smoke),
                        },
                        "seed": seed,
                        "extra_cols": {
                            "stage": "screen",
                            "background_id": f"{background_id}_w={window}",
                            "arm": "on",
                            "seed": seed,
                            "source_background_id": background_id,
                            "horizon_days": horizon_days,
                            "Q": STANDARD_Q,
                            "window": window,
                        },
                    }
                )
    return tasks


# ---------------------------------------------------------------------
# Grid: full (Q, window) sweep at a curated subset of backgrounds
# ---------------------------------------------------------------------

def select_deep_backgrounds(bank: pd.DataFrame) -> pd.DataFrame:
    """Pick a small, labeled set of backgrounds spanning H1's condition
    (threshold_1 < threshold_2) and its violations, so the deep grid
    covers both rather than only the "expected" region.
    """
    gap = bank["noshow_threshold_2"] - bank["noshow_threshold_1"]
    high_demand = bank["rho"] >= 2.0

    buckets = {
        "condition_satisfied_strong_gap": bank[(gap > 0) & high_demand].nlargest(
            N_DEEP_BACKGROUNDS_PER_BUCKET, "rho"
        ),
        "condition_violated_no_gap": bank[(gap <= 0) & high_demand].head(N_DEEP_BACKGROUNDS_PER_BUCKET),
        "condition_violated_low_demand": bank[(gap > 0) & (bank["rho"] < 1.2)].head(
            N_DEEP_BACKGROUNDS_PER_BUCKET
        ),
        # Diversify by capacity only: every selected background is tested at
        # every H1_HORIZON_VALUES horizon in grid_tasks regardless of the
        # bank row's own horizon_days, so diversifying this bucket by
        # horizon_days would no longer add coverage the way it used to.
        "diverse_capacity": bank[(gap > 0) & high_demand]
        .drop_duplicates(subset=["slots_per_day"])
        .head(N_DEEP_BACKGROUNDS_PER_BUCKET),
    }
    selected = []
    for label, subset in buckets.items():
        subset = subset.copy()
        subset["deep_bucket"] = label
        selected.append(subset)
    return pd.concat(selected, ignore_index=True).drop_duplicates(subset="background_id")


def select_condition_comparison_backgrounds(bank: pd.DataFrame) -> pd.DataFrame:
    """A larger, condition-balanced background set for testing whether the
    threshold-gap condition still matters once the policy is tuned to each
    background's own optimum, instead of held at one fixed (Q, window).
    """
    eligible = bank[bank["rho"] > CONDITION_GRID_MIN_RHO].copy()
    gap = eligible["noshow_threshold_2"] - eligible["noshow_threshold_1"]
    satisfied = eligible[gap > 0]
    violated = eligible[gap <= 0]

    n_sat = min(CONDITION_GRID_N_PER_BUCKET, len(satisfied))
    n_viol = min(CONDITION_GRID_N_PER_BUCKET, len(violated))
    satisfied = satisfied.sample(n=n_sat, random_state=CONDITION_GRID_SEED).copy()
    violated = violated.sample(n=n_viol, random_state=CONDITION_GRID_SEED).copy()
    satisfied["deep_bucket"] = "condition_satisfied"
    violated["deep_bucket"] = "condition_violated"
    return pd.concat([satisfied, violated], ignore_index=True)


def grid_tasks(deep_backgrounds: pd.DataFrame, smoke: bool, stage_label: str = "grid") -> list[dict[str, Any]]:
    """Full (horizon, Q, window) sweep at a curated set of backgrounds.

    Horizon is swept explicitly here (H1_HORIZON_VALUES), holding a
    background's other parameters fixed -- this is where booking horizon
    is actually tested as a policy lever, rather than left at whatever
    value the bank assigned. Each tested horizon gets its own baseline
    (Q=0) cell and its own window range (1..that horizon), since both
    the "no reservation" comparison point and the window candidates are
    horizon-dependent.
    """
    tasks = []
    rows = deep_backgrounds.head(2) if smoke else deep_backgrounds
    horizons = H1_HORIZON_VALUES[:2] if smoke else H1_HORIZON_VALUES
    for _, row in rows.iterrows():
        background_id = row["background_id"]
        base_kwargs = _row_config_kwargs(row)
        q_values = q_grid_for_capacity(int(row["slots_per_day"]))
        if smoke:
            q_values = q_values[:2]

        for horizon in horizons:
            horizon_kwargs = {**base_kwargs, "horizon_days": horizon}
            window_values = window_grid_for_horizon(horizon)
            if smoke:
                window_values = window_values[:2]

            for seed in _seeds(smoke):
                tasks.append(
                    {
                        "config_kwargs": {
                            **horizon_kwargs,
                            **_reservation_kwargs(False, 0, 0),
                            **_smoke_overrides(smoke),
                        },
                        "seed": seed,
                        "extra_cols": {
                            "stage": stage_label,
                            "background_id": f"{background_id}_H={horizon}_Q=0",
                            "arm": stage_label,
                            "seed": seed,
                            "source_background_id": background_id,
                            "horizon_days": horizon,
                            "Q": 0,
                            "window": -1,
                        },
                    }
                )
            for q in q_values:
                for window in window_values:
                    for seed in _seeds(smoke):
                        tasks.append(
                            {
                                "config_kwargs": {
                                    **horizon_kwargs,
                                    **_reservation_kwargs(True, q, window),
                                    **_smoke_overrides(smoke),
                                },
                                "seed": seed,
                                "extra_cols": {
                                    "stage": stage_label,
                                    "background_id": f"{background_id}_H={horizon}_Q={q}_w={window}",
                                    "arm": stage_label,
                                    "seed": seed,
                                    "source_background_id": background_id,
                                    "horizon_days": horizon,
                                    "Q": q,
                                    "window": window,
                                },
                            }
                        )
    return tasks


def build_tasks(stages: list[str], bank: pd.DataFrame, smoke: bool) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    if "screen" in stages:
        tasks.extend(screen_tasks(bank, smoke))
    if "grid" in stages:
        deep_backgrounds = select_deep_backgrounds(bank)
        tasks.extend(grid_tasks(deep_backgrounds, smoke))
    if "condition_grid" in stages:
        condition_backgrounds = select_condition_comparison_backgrounds(bank)
        tasks.extend(grid_tasks(condition_backgrounds, smoke, stage_label="condition_grid"))
    return tasks


def run(
    *, stages: list[str], bank_path: Path, output_dir: Path, workers: int, smoke: bool, resume: bool
) -> Path:
    bank = load_bank(bank_path)
    raw_path = output_dir / "raw" / "h1_raw.csv"
    tasks = build_tasks(stages, bank, smoke)

    completed: set[tuple[Any, ...]] = set()
    if resume:
        completed = load_completed_keys(raw_path, KEY_COLUMNS)
    elif raw_path.exists():
        raw_path.unlink()

    pending = [t for t in tasks if tuple(t["extra_cols"][c] for c in KEY_COLUMNS) not in completed]
    print(f"H1 stages: {stages}; backgrounds in bank: {len(bank)}")
    print(f"Total tasks: {len(tasks):,}; already completed: {len(completed):,}; to run: {len(pending):,}")
    run_tasks(pending, raw_path=raw_path, workers=workers)
    print(f"Raw results: {raw_path}")
    return raw_path


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------

VALUE_COLS: dict[str, tuple[str, str | None]] = {
    "utilization": ("average_utilization", "positive"),
    "class_1_served_rate": ("class_1_percent_serviced", "positive"),
    "class_2_served_rate": ("class_2_percent_serviced", None),
    "overall_served_rate": ("overall_percent_serviced", None),
    "mean_offered_delay": ("mean_offered_booking_delay", None),
}


def classify_screen(raw: pd.DataFrame, bank: pd.DataFrame) -> pd.DataFrame:
    """Per background: pick the on-arm window with the best mean
    utilization (Q held at STANDARD_Q), then paired-bootstrap that best
    window against the off arm. Grouping is on source_background_id, not
    background_id, since each on-arm window cell now has its own
    background_id (e.g. "BG00001_w=5") to keep resumable-run keys unique
    per window -- see screen_tasks.
    """
    screen = raw[raw["stage"] == "screen"]
    rows: list[dict[str, Any]] = []
    for background_id, group in screen.groupby("source_background_id", sort=False):
        off = group[group["arm"] == "off"].sort_values("seed").set_index("seed")
        on_all = group[group["arm"] == "on"]
        if off.empty or on_all.empty:
            continue

        window_means = on_all.groupby("window")["average_utilization"].mean()
        if window_means.empty:
            continue
        best_window = int(window_means.idxmax())
        on = on_all[on_all["window"] == best_window].sort_values("seed").set_index("seed")

        paired_seeds = sorted(set(on.index) & set(off.index))
        if not paired_seeds:
            continue
        row: dict[str, Any] = {
            "background_id": background_id,
            "n_paired_seeds": len(paired_seeds),
            "best_window": best_window,
        }
        row["reserved_slot_fill_rate_on_arm"] = float(on.loc[paired_seeds, "reserved_slot_fill_rate"].mean())
        for prefix, (column, expected_sign) in VALUE_COLS.items():
            mean, low, high, _ = paired_delta_ci(
                on.loc[paired_seeds, column].tolist(),
                off.loc[paired_seeds, column].tolist(),
                seed=abs(hash((background_id, prefix))) % (2**31),
            )
            row[f"delta_{prefix}"] = mean
            row[f"delta_{prefix}_ci_low"] = low
            row[f"delta_{prefix}_ci_high"] = high
            if expected_sign is not None:
                row[f"{prefix}_status"] = classify_effect(mean, low, high, expected_sign=expected_sign)
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    bank_cols = [
        "background_id",
        "horizon_days",
        "rho",
        "class1_share",
        "slots_per_day",
        "noshow_threshold_1",
        "noshow_threshold_2",
    ]
    table = table.merge(bank[bank_cols], on="background_id", how="left")
    table["threshold_gap"] = table["noshow_threshold_2"] - table["noshow_threshold_1"]
    table["condition_satisfied"] = table["threshold_gap"] > 0
    return table


def classify_grid(raw: pd.DataFrame, stage_label: str = "grid") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (background, horizon): best (Q, window) vs. no reservation.

    Horizon is now one of the swept dimensions (see grid_tasks), so the
    grid summary has one row per background PER TESTED HORIZON, not one
    row per background. There is no more single "naive" (Q, window) cell
    to compare against -- the old fixed Q=5/window=3 comparison point
    doesn't generalize once window sweeps 1..horizon and horizon itself
    varies -- so this only reports optimal-vs-none, not naive-vs-optimal.
    """
    grid = raw[raw["stage"] == stage_label]
    cell_means = grid.groupby(["source_background_id", "horizon_days", "Q", "window"], as_index=False)[
        "average_utilization"
    ].mean()

    positive_q = cell_means[cell_means["Q"] > 0]
    best_table = best_and_near_tie(
        positive_q,
        group_cols=["source_background_id", "horizon_days"],
        param_cols=["Q", "window"],
        value_col="average_utilization",
        tolerance=0.01,
    )
    none_table = cell_means[cell_means["Q"] == 0][
        ["source_background_id", "horizon_days", "average_utilization"]
    ].rename(columns={"average_utilization": "none_utilization"})

    combined = best_table.rename(columns={"best_value": "optimal_utilization"})
    combined = combined.merge(none_table, on=["source_background_id", "horizon_days"], how="left")
    combined["optimal_minus_none"] = combined["optimal_utilization"] - combined["none_utilization"]
    return combined, cell_means


def _optimal_vs_none_status(
    raw_stage: pd.DataFrame, combined: pd.DataFrame, bank: pd.DataFrame
) -> pd.DataFrame:
    """Seed-level paired bootstrap of each background's best (Q, window)
    cell against Q=0, per tested horizon, so the condition_grid comparison
    has a real supported / contradicted / inconclusive verdict for each
    (background, horizon) combination, not just a bare mean of already
    seed-averaged cells.
    """
    gap = bank["noshow_threshold_2"] - bank["noshow_threshold_1"]
    bank_cond = bank.assign(condition_satisfied=gap > 0)[
        ["background_id", "condition_satisfied", "rho", "class1_share", "slots_per_day"]
    ]

    rows: list[dict[str, Any]] = []
    for _, r in combined.iterrows():
        bg = r["source_background_id"]
        horizon = int(r["horizon_days"])
        match = re.match(r"Q=(-?\d+),window=(-?\d+)", str(r["best_params"]))
        if not match:
            continue
        best_q, best_w = int(match.group(1)), int(match.group(2))
        bg_raw = raw_stage[
            (raw_stage["source_background_id"] == bg) & (raw_stage["horizon_days"] == horizon)
        ]
        best_rows = bg_raw[(bg_raw["Q"] == best_q) & (bg_raw["window"] == best_w)].sort_values("seed")
        none_rows = bg_raw[bg_raw["Q"] == 0].sort_values("seed")
        paired_seeds = sorted(set(best_rows["seed"]) & set(none_rows["seed"]))
        if not paired_seeds:
            continue
        best_idx = best_rows.set_index("seed")
        none_idx = none_rows.set_index("seed")
        mean, low, high, _ = paired_delta_ci(
            best_idx.loc[paired_seeds, "average_utilization"].tolist(),
            none_idx.loc[paired_seeds, "average_utilization"].tolist(),
            seed=abs(hash((bg, horizon, "optimal_vs_none"))) % (2**31),
        )
        rows.append(
            {
                "background_id": bg,
                "horizon_days": horizon,
                "best_q": best_q,
                "best_window": best_w,
                "delta_optimal_vs_none": mean,
                "delta_optimal_vs_none_ci_low": low,
                "delta_optimal_vs_none_ci_high": high,
                "optimal_status": classify_effect(mean, low, high, expected_sign="positive"),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.merge(bank_cond, on="background_id", how="left")


def classify(*, raw_path: Path, bank_path: Path, output_dir: Path) -> None:
    raw = pd.read_csv(raw_path)
    bank = load_bank(bank_path)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if (raw["stage"] == "screen").any():
        screen_table = classify_screen(raw, bank)
        screen_table.to_csv(summary_dir / "screen_by_background.csv", index=False)

        if not screen_table.empty:
            by_condition = (
                screen_table.groupby(["condition_satisfied", "utilization_status"])
                .size()
                .rename("n_backgrounds")
                .reset_index()
            )
            by_condition.to_csv(summary_dir / "screen_by_condition.csv", index=False)
            print(f"Screen: {summary_dir / 'screen_by_background.csv'}")
            print(f"Screen by condition: {summary_dir / 'screen_by_condition.csv'}")
            print(by_condition.to_string(index=False))
        else:
            print("Screen: no paired on/off rows found; check --stage and raw output.")

    if (raw["stage"] == "grid").any():
        combined, cell_means = classify_grid(raw, stage_label="grid")
        combined.to_csv(summary_dir / "grid_policy_summary.csv", index=False)
        cell_means.to_csv(summary_dir / "grid_cell_means.csv", index=False)
        print(f"Grid: {summary_dir / 'grid_policy_summary.csv'}")

    if (raw["stage"] == "condition_grid").any():
        raw_cg = raw[raw["stage"] == "condition_grid"]
        combined_cg, cell_means_cg = classify_grid(raw, stage_label="condition_grid")
        combined_cg.to_csv(summary_dir / "condition_grid_policy_summary.csv", index=False)
        cell_means_cg.to_csv(summary_dir / "condition_grid_cell_means.csv", index=False)

        status_table = _optimal_vs_none_status(raw_cg, combined_cg, bank)
        if not status_table.empty:
            status_table.to_csv(summary_dir / "condition_grid_optimal_status.csv", index=False)
            by_condition_cg = (
                status_table.groupby(["condition_satisfied", "optimal_status"])
                .size()
                .rename("n_backgrounds")
                .reset_index()
            )
            by_condition_cg.to_csv(summary_dir / "condition_grid_by_condition.csv", index=False)
            print(f"Condition grid: {summary_dir / 'condition_grid_optimal_status.csv'}")
            print(by_condition_cg.to_string(index=False))
            print(
                status_table.groupby("condition_satisfied")["delta_optimal_vs_none"]
                .agg(["mean", "median", "count"])
                .to_string()
            )

    _write_summary(raw, summary_dir)


def _write_summary(raw: pd.DataFrame, summary_dir: Path) -> None:
    lines = [
        "# H1 Short-Horizon Reservation: Summary",
        "",
        f"Practical-equivalence tolerance: {PRACTICAL_TOLERANCE}",
        f"Rows in raw results: {len(raw):,}",
        "",
        "This is an auto-generated data summary, not the narrative report.",
        "See screen_by_condition.csv for whether the threshold-gap condition",
        "is empirically necessary at each background's own best reservation",
        "window (Q=5, window swept 1..horizon), grid_policy_summary.csv for",
        "the optimal-vs-none comparison per (background, horizon) at 12",
        "curated backgrounds, and condition_grid_by_condition.csv for",
        "whether the condition matters under each background's own optimal",
        "policy at a larger, condition-balanced background set.",
    ]
    write_markdown(lines, summary_dir / "h1_summary.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "classify", "all"])
    parser.add_argument("--stage", default="all", help="screen, grid, condition_grid, or 'all' (screen+grid)")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _resolve_stages(spec: str) -> list[str]:
    if spec == "all":
        return ["screen", "grid"]
    return [s.strip().lower() for s in spec.split(",") if s.strip()]


def main() -> None:
    args = build_parser().parse_args()
    if not args.bank.exists():
        print(f"Background bank not found at {args.bank}; generating the default bank now.")
        bank = generate_background_bank()
        args.bank.parent.mkdir(parents=True, exist_ok=True)
        bank.to_csv(args.bank, index=False)

    stages = _resolve_stages(args.stage)
    raw_path = args.output_dir / "raw" / "h1_raw.csv"

    if args.command in {"run", "all"}:
        raw_path = run(
            stages=stages,
            bank_path=args.bank,
            output_dir=args.output_dir,
            workers=args.workers,
            smoke=args.smoke,
            resume=not args.no_resume,
        )
    if args.command in {"classify", "all"}:
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw H1 results not found: {raw_path}. Run the experiment first.")
        classify(raw_path=raw_path, bank_path=args.bank, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
