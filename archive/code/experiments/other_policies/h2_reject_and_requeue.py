"""Hypothesis 2: standby/requeue for long-horizon, wait-sensitive, high-demand clinics.

Claim: on long booking horizons (>=21 days), letting patients reject a
far-out offer and sit in a per-class standby queue for a possible earlier
opening (instead of immediately balking) raises utilization, chiefly by
avoiding no-shows that would otherwise occur on offers patients accept but
are unlikely to attend given how far out they are.

This consumes the shared background-scenario bank from
experiments/hypothesis_scenario_bank.py, unfiltered by H2's stated
condition (horizon_days >= 21 and rho >= 2.0) -- the condition itself is
part of what's being tested, not assumed.

Two stages:

    screen  Broad on/off test at a fixed standby policy (standby_prob=0.8
            both classes, no eligibility delay, no max_standby_days cap)
            across every background in the bank. Answers whether the
            effect exists at all, and whether it concentrates in
            backgrounds that satisfy H2's stated condition.
    dose    standby_prob dose-response sweep {0.0, 0.1, ..., 0.8} at a
            curated set of backgrounds (48, 12 per condition bucket)
            spanning condition-satisfying and condition-violating cases
            (violating via horizon alone, via rho alone, and via both).

standby_eligible_after_days is deliberately left unset (None) in this
version: it is not required by the engine (which treats an unset value
as 0, i.e. immediate eligibility) and was found to add a second
confound on top of the FIFO/cancellation-rate bottleneck already
limiting recalls, making it hard to tell whether a weak result came
from that bottleneck or from the extra delay. Removing it isolates the
standby_prob dose-response question.

Run from the repository root:

    python experiments/hypothesis_scenario_bank.py          # build the bank once
    python experiments/h2_reject_and_requeue.py all --stage all
    python experiments/h2_reject_and_requeue.py classify --stage all

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
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "hypotheses" / "h2_reject_and_requeue"

KEY_COLUMNS = ["stage", "background_id", "arm", "seed"]

STANDARD_STANDBY_PROB = 0.8
STANDBY_ELIGIBLE_AFTER_DAYS = None
DOSE_VALUES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
N_DEEP_BACKGROUNDS_PER_BUCKET = 12

STANDBY_DIAGNOSTIC_COLS = [
    "class_1_standby_joined",
    "class_2_standby_joined",
    "class_1_standby_recalled",
    "class_2_standby_recalled",
    "class_1_standby_expired",
    "class_2_standby_expired",
    "class_1_standby_recall_rate",
    "class_2_standby_recall_rate",
    "class_1_mean_standby_wait_days",
    "class_2_mean_standby_wait_days",
    "class_1_mean_original_offered_delay_recalled",
    "class_2_mean_original_offered_delay_recalled",
]


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


def _standby_kwargs(prob: float) -> dict[str, Any]:
    # max_standby_days left at None (uncapped) both classes: the user's own
    # lean, and the simpler baseline -- a cap is a second lever that would
    # confound the dose-response read on standby_prob alone. Worth a
    # follow-up sweep later if the uncapped queue shows pathological
    # buildup (very old entries with 0% recall probability sitting in
    # queue and biasing mean_standby_wait_days upward without ever
    # resolving), but the engine's own expiry-at-cooldown-boundary logic
    # already bounds a queue's lifetime to at most the horizon.
    #
    # standby_eligible_after_days_* is also left unset (None -> the engine
    # treats this as 0, immediate eligibility): it is not required by the
    # engine and, in the first version of this sweep, added a second gate
    # on top of the FIFO/cancellation-rate bottleneck that was already
    # suppressing recalls, making the two hard to tell apart.
    return {
        "standby_prob_1": prob,
        "standby_prob_2": prob,
        "standby_eligible_after_days_1": STANDBY_ELIGIBLE_AFTER_DAYS,
        "standby_eligible_after_days_2": STANDBY_ELIGIBLE_AFTER_DAYS,
        "max_standby_days_1": None,
        "max_standby_days_2": None,
    }


def _seeds(smoke: bool) -> tuple[int, ...]:
    return STAGE1_SEEDS[:2] if smoke else STAGE1_SEEDS


def _smoke_overrides(smoke: bool) -> dict[str, Any]:
    if not smoke:
        return {}
    return {"burn_in_days": 5, "measure_days": 20, "cooldown_days": 5}


# ---------------------------------------------------------------------
# Screen: broad on/off test across the whole bank
# ---------------------------------------------------------------------

def screen_tasks(bank: pd.DataFrame, smoke: bool) -> list[dict[str, Any]]:
    rows = bank.sample(n=min(40, len(bank)), random_state=0) if smoke else bank
    tasks = []
    for _, row in rows.iterrows():
        background_id = row["background_id"]
        base_kwargs = _row_config_kwargs(row)
        for arm, prob in (("off", 0.0), ("on", STANDARD_STANDBY_PROB)):
            for seed in _seeds(smoke):
                tasks.append(
                    {
                        "config_kwargs": {
                            **base_kwargs,
                            **_standby_kwargs(prob),
                            **_smoke_overrides(smoke),
                        },
                        "seed": seed,
                        "extra_cols": {
                            "stage": "screen",
                            "background_id": background_id,
                            "arm": arm,
                            "seed": seed,
                            "source_background_id": background_id,
                            "standby_prob": prob,
                        },
                    }
                )
    return tasks


# ---------------------------------------------------------------------
# Dose: standby_prob sweep at a curated subset of backgrounds
# ---------------------------------------------------------------------

def select_deep_backgrounds(bank: pd.DataFrame) -> pd.DataFrame:
    """Pick a small, labeled set of backgrounds spanning H2's condition
    (horizon_days >= 21 and rho >= 2.0), each dimension violated
    separately as well as jointly satisfied, so the dose sweep can show
    whether either condition alone is doing the work.
    """
    long_horizon = bank["horizon_days"] >= 21
    high_demand = bank["rho"] >= 2.0

    buckets = {
        "condition_satisfied": bank[long_horizon & high_demand].nlargest(
            N_DEEP_BACKGROUNDS_PER_BUCKET, "rho"
        ),
        "violated_short_horizon_high_demand": bank[~long_horizon & high_demand].head(
            N_DEEP_BACKGROUNDS_PER_BUCKET
        ),
        "violated_long_horizon_low_demand": bank[long_horizon & ~high_demand].head(
            N_DEEP_BACKGROUNDS_PER_BUCKET
        ),
        "violated_both": bank[~long_horizon & ~high_demand].head(N_DEEP_BACKGROUNDS_PER_BUCKET),
    }
    selected = []
    for label, subset in buckets.items():
        subset = subset.copy()
        subset["deep_bucket"] = label
        selected.append(subset)
    return pd.concat(selected, ignore_index=True).drop_duplicates(subset="background_id")


def dose_tasks(deep_backgrounds: pd.DataFrame, smoke: bool) -> list[dict[str, Any]]:
    tasks = []
    rows = deep_backgrounds.head(2) if smoke else deep_backgrounds
    dose_values = DOSE_VALUES[:2] if smoke else DOSE_VALUES
    for _, row in rows.iterrows():
        background_id = row["background_id"]
        base_kwargs = _row_config_kwargs(row)
        for prob in dose_values:
            for seed in _seeds(smoke):
                tasks.append(
                    {
                        "config_kwargs": {
                            **base_kwargs,
                            **_standby_kwargs(prob),
                            **_smoke_overrides(smoke),
                        },
                        "seed": seed,
                        "extra_cols": {
                            "stage": "dose",
                            "background_id": f"{background_id}_p={prob}",
                            "arm": "dose",
                            "seed": seed,
                            "source_background_id": background_id,
                            "standby_prob": prob,
                        },
                    }
                )
    return tasks


def build_tasks(stages: list[str], bank: pd.DataFrame, smoke: bool) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    if "screen" in stages:
        tasks.extend(screen_tasks(bank, smoke))
    if "dose" in stages:
        deep_backgrounds = select_deep_backgrounds(bank)
        tasks.extend(dose_tasks(deep_backgrounds, smoke))
    return tasks


def run(
    *, stages: list[str], bank_path: Path, output_dir: Path, workers: int, smoke: bool, resume: bool
) -> Path:
    bank = load_bank(bank_path)
    raw_path = output_dir / "raw" / "h2_raw.csv"
    tasks = build_tasks(stages, bank, smoke)

    completed: set[tuple[Any, ...]] = set()
    if resume:
        completed = load_completed_keys(raw_path, KEY_COLUMNS)
    elif raw_path.exists():
        raw_path.unlink()

    pending = [t for t in tasks if tuple(t["extra_cols"][c] for c in KEY_COLUMNS) not in completed]
    print(f"H2 stages: {stages}; backgrounds in bank: {len(bank)}")
    print(f"Total tasks: {len(tasks):,}; already completed: {len(completed):,}; to run: {len(pending):,}")
    run_tasks(pending, raw_path=raw_path, workers=workers)
    print(f"Raw results: {raw_path}")
    return raw_path


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------

VALUE_COLS: dict[str, tuple[str, str | None]] = {
    "utilization": ("average_utilization", "positive"),
    "class_1_no_show_rate": ("class_1_no_show_rate", "negative"),
    "class_2_no_show_rate": ("class_2_no_show_rate", "negative"),
    "mean_accepted_delay": ("mean_accepted_booking_delay", None),
    "class_1_served_rate": ("class_1_percent_serviced", None),
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
        for col in STANDBY_DIAGNOSTIC_COLS:
            row[f"{col}_on_arm"] = float(on.loc[paired_seeds, col].mean())
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
    bank_cols = ["background_id", "horizon_days", "rho", "class1_share", "slots_per_day"]
    table = table.merge(bank[bank_cols], on="background_id", how="left")
    table["condition_satisfied"] = (table["horizon_days"] >= 21) & (table["rho"] >= 2.0)
    return table


def classify_dose(raw: pd.DataFrame) -> pd.DataFrame:
    dose = raw[raw["stage"] == "dose"]
    cell_means = dose.groupby(["source_background_id", "standby_prob"], as_index=False)[
        ["average_utilization", "class_1_no_show_rate", "class_2_no_show_rate"]
    ].mean()
    return cell_means


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

    if (raw["stage"] == "dose").any():
        dose_table = classify_dose(raw)
        dose_table.to_csv(summary_dir / "dose_response.csv", index=False)
        print(f"Dose response: {summary_dir / 'dose_response.csv'}")

    _write_summary(raw, summary_dir)


def _write_summary(raw: pd.DataFrame, summary_dir: Path) -> None:
    lines = [
        "# H2 Reject-and-Requeue: Summary",
        "",
        f"Practical-equivalence tolerance: {PRACTICAL_TOLERANCE}",
        f"Rows in raw results: {len(raw):,}",
        "",
        "This is an auto-generated data summary, not the narrative report.",
        "See screen_by_condition.csv for whether the horizon/demand condition",
        "is empirically necessary, and dose_response.csv for the standby_prob",
        "dose-response curve at the curated backgrounds.",
    ]
    write_markdown(lines, summary_dir / "h2_summary.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "classify", "all"])
    parser.add_argument("--stage", default="all", help="screen, dose, or 'all'")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _resolve_stages(spec: str) -> list[str]:
    if spec == "all":
        return ["screen", "dose"]
    return [s.strip().lower() for s in spec.split(",") if s.strip()]


def main() -> None:
    args = build_parser().parse_args()
    if not args.bank.exists():
        print(f"Background bank not found at {args.bank}; generating the default bank now.")
        bank = generate_background_bank()
        args.bank.parent.mkdir(parents=True, exist_ok=True)
        bank.to_csv(args.bank, index=False)

    stages = _resolve_stages(args.stage)
    raw_path = args.output_dir / "raw" / "h2_raw.csv"

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
            raise FileNotFoundError(f"Raw H2 results not found: {raw_path}. Run the experiment first.")
        classify(raw_path=raw_path, bank_path=args.bank, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
