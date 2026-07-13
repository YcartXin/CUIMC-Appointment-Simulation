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

Two stages:

    screen  Broad on/off test at a fixed modest policy (Q=5, window=3,
            always valid given the bank's constraints) across every
            background in the bank. Answers whether the effect exists at
            all, and whether it concentrates in backgrounds that satisfy
            H1's stated condition or shows up elsewhere too.
    grid    Full (Q, window) grid -- Q up to half of that background's own
            capacity in steps of 5, window in {3,4,5,6,7} filtered to
            <= that background's noshow_threshold_1 -- at a small curated
            set of backgrounds spanning condition-satisfying and
            condition-violating cases. Answers the optimal-vs-naive-vs-
            none question.

Run from the repository root:

    python experiments/hypothesis_scenario_bank.py          # build the bank once
    python experiments/h1_short_horizon_reservation.py all --stage all
    python experiments/h1_short_horizon_reservation.py classify --stage all

Use --smoke for a fast end-to-end check before a full run.
"""

from __future__ import annotations

import argparse
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
    generate_background_bank,
)

REPO_DIR = _REPO_DIR
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "hypotheses" / "h1_short_horizon_reservation"

KEY_COLUMNS = ["stage", "background_id", "arm", "seed"]

STANDARD_Q = 5
STANDARD_WINDOW = 3
WINDOW_CANDIDATES = (3, 4, 5, 6, 7)
N_DEEP_BACKGROUNDS_PER_BUCKET = 3


def load_bank(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Background bank not found: {path}. Run "
            "`python experiments/hypothesis_scenario_bank.py` first."
        )
    return pd.read_csv(path)


def _row_config_kwargs(row: pd.Series) -> dict[str, Any]:
    return {
        "horizon_days": int(row["horizon_days"]),
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


def window_grid_for_threshold(threshold_1: int) -> list[int]:
    return [w for w in WINDOW_CANDIDATES if w <= threshold_1]


# ---------------------------------------------------------------------
# Screen: broad on/off test across the whole bank
# ---------------------------------------------------------------------

def screen_tasks(bank: pd.DataFrame, smoke: bool) -> list[dict[str, Any]]:
    rows = bank.sample(n=min(40, len(bank)), random_state=0) if smoke else bank
    tasks = []
    for _, row in rows.iterrows():
        background_id = row["background_id"]
        base_kwargs = _row_config_kwargs(row)
        for arm in ("off", "on"):
            for seed in _seeds(smoke):
                tasks.append(
                    {
                        "config_kwargs": {
                            **base_kwargs,
                            **_reservation_kwargs(arm == "on", STANDARD_Q, STANDARD_WINDOW),
                            **_smoke_overrides(smoke),
                        },
                        "seed": seed,
                        "extra_cols": {
                            "stage": "screen",
                            "background_id": background_id,
                            "arm": arm,
                            "seed": seed,
                            "source_background_id": background_id,
                            "Q": STANDARD_Q,
                            "window": STANDARD_WINDOW,
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
        "diverse_capacity_horizon": bank[(gap > 0) & high_demand]
        .drop_duplicates(subset=["horizon_days", "slots_per_day"])
        .head(N_DEEP_BACKGROUNDS_PER_BUCKET),
    }
    selected = []
    for label, subset in buckets.items():
        subset = subset.copy()
        subset["deep_bucket"] = label
        selected.append(subset)
    return pd.concat(selected, ignore_index=True).drop_duplicates(subset="background_id")


def grid_tasks(deep_backgrounds: pd.DataFrame, smoke: bool) -> list[dict[str, Any]]:
    tasks = []
    rows = deep_backgrounds.head(2) if smoke else deep_backgrounds
    for _, row in rows.iterrows():
        background_id = row["background_id"]
        base_kwargs = _row_config_kwargs(row)
        q_values = q_grid_for_capacity(int(row["slots_per_day"]))
        window_values = window_grid_for_threshold(int(row["noshow_threshold_1"]))
        if smoke:
            q_values = q_values[:2]
            window_values = window_values[:2]

        for seed in _seeds(smoke):
            tasks.append(
                {
                    "config_kwargs": {**base_kwargs, **_reservation_kwargs(False, 0, 0), **_smoke_overrides(smoke)},
                    "seed": seed,
                    "extra_cols": {
                        "stage": "grid",
                        "background_id": f"{background_id}_Q=0",
                        "arm": "grid",
                        "seed": seed,
                        "source_background_id": background_id,
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
                                **base_kwargs,
                                **_reservation_kwargs(True, q, window),
                                **_smoke_overrides(smoke),
                            },
                            "seed": seed,
                            "extra_cols": {
                                "stage": "grid",
                                "background_id": f"{background_id}_Q={q}_w={window}",
                                "arm": "grid",
                                "seed": seed,
                                "source_background_id": background_id,
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
    screen = raw[raw["stage"] == "screen"]
    rows: list[dict[str, Any]] = []
    for background_id, group in screen.groupby("background_id", sort=False):
        on = group[group["arm"] == "on"].sort_values("seed").set_index("seed")
        off = group[group["arm"] == "off"].sort_values("seed").set_index("seed")
        paired_seeds = sorted(set(on.index) & set(off.index))
        if not paired_seeds:
            continue
        row: dict[str, Any] = {"background_id": background_id, "n_paired_seeds": len(paired_seeds)}
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


def classify_grid(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = raw[raw["stage"] == "grid"]
    cell_means = grid.groupby(["source_background_id", "Q", "window"], as_index=False)[
        "average_utilization"
    ].mean()

    positive_q = cell_means[cell_means["Q"] > 0]
    best_table = best_and_near_tie(
        positive_q,
        group_cols=["source_background_id"],
        param_cols=["Q", "window"],
        value_col="average_utilization",
        tolerance=0.01,
    )
    none_table = cell_means[cell_means["Q"] == 0][["source_background_id", "average_utilization"]].rename(
        columns={"average_utilization": "none_utilization"}
    )
    naive = cell_means[
        (cell_means["Q"] == STANDARD_Q) & (cell_means["window"] == STANDARD_WINDOW)
    ][["source_background_id", "average_utilization"]].rename(columns={"average_utilization": "naive_utilization"})

    combined = best_table.rename(columns={"background_id": "source_background_id", "best_value": "optimal_utilization"})
    combined = combined.merge(none_table, on="source_background_id", how="left").merge(
        naive, on="source_background_id", how="left"
    )
    combined["naive_minus_none"] = combined["naive_utilization"] - combined["none_utilization"]
    combined["optimal_minus_naive"] = combined["optimal_utilization"] - combined["naive_utilization"]
    combined["optimal_minus_none"] = combined["optimal_utilization"] - combined["none_utilization"]
    return combined, cell_means


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
        combined, cell_means = classify_grid(raw)
        combined.to_csv(summary_dir / "grid_policy_summary.csv", index=False)
        cell_means.to_csv(summary_dir / "grid_cell_means.csv", index=False)
        print(f"Grid: {summary_dir / 'grid_policy_summary.csv'}")

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
        "is empirically necessary, and grid_policy_summary.csv for the",
        "optimal-vs-naive-vs-none comparison at the curated backgrounds.",
    ]
    write_markdown(lines, summary_dir / "h1_summary.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "classify", "all"])
    parser.add_argument("--stage", default="all", help="screen, grid, or 'all'")
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
