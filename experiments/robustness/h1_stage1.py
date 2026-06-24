"""Run and classify Stage 1 robustness tests for H1.

H1: A class with higher cancellation probability can shorten offered delay
under high demand, but lowers its own served rate.

Run from the repository root:

    python experiments/robustness/h1_stage1.py run
    python experiments/robustness/h1_stage1.py classify
    python experiments/robustness/h1_stage1.py all

Use ``--smoke`` before the full run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from experiments.robustness.scenario_space import (  # noqa: E402
    CANCEL_VALUES,
    PARAMETER_COLUMNS,
    STAGE1_SEEDS,
)
DEFAULT_BASE_CONFIG = REPO_DIR / "configs" / "baseline.yaml"

DEFAULT_SCENARIOS = REPO_DIR / "outputs" / "robustness" / "scenarios" / "all_stage1_scenarios.csv"
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "robustness" / "h1"
DEFAULT_RAW_PATH = DEFAULT_OUTPUT_DIR / "raw" / "h1_stage1_raw.csv"
DEFAULT_BACKGROUND_PATH = DEFAULT_OUTPUT_DIR / "design" / "h1_background_scenarios.csv"
DEFAULT_EFFECTS_PATH = DEFAULT_OUTPUT_DIR / "summary" / "h1_scenario_effects.csv"

SERVED_THRESHOLD = 0.005
DELAY_THRESHOLD_DAYS = 0.25
HIGH_DEMAND_MIN = 3.1
LOW_DEMAND_MAX = 1.25
FOCAL_LEVELS = tuple(float(x) for x in CANCEL_VALUES)
FOCAL_LOW = min(FOCAL_LEVELS)
FOCAL_HIGH = max(FOCAL_LEVELS)


def _clean_string(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def prepare_h1_backgrounds(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Overwrite H1's focal parameter conceptually and deduplicate backgrounds."""
    required = set(PARAMETER_COLUMNS) | {
        "scenario_id",
        "scenario_type",
        "parent_scenario_id",
        "design_note",
        "asymmetric_dimensions",
    }
    missing = required.difference(scenarios.columns)
    if missing:
        raise ValueError(f"Scenario CSV is missing columns: {sorted(missing)}")

    df = scenarios.copy()
    df["source_scenario_id"] = df["scenario_id"].astype(str)
    # Class 1 cancellation is focal for H1, so it is excluded from the
    # background signature and overwritten during simulation.
    signature_columns = [c for c in PARAMETER_COLUMNS if c != "cancel_class1"]

    grouped_rows: list[dict[str, Any]] = []
    for _, group in df.groupby(signature_columns, dropna=False, sort=False):
        first = group.iloc[0].to_dict()
        first["source_scenario_ids"] = ";".join(group["source_scenario_id"].astype(str))
        first["source_scenario_count"] = int(len(group))
        grouped_rows.append(first)

    out = pd.DataFrame(grouped_rows)
    out = out.reset_index(drop=True)
    out.insert(0, "background_id", [f"H1B{i:04d}" for i in range(1, len(out) + 1)])
    out["cancel_class1"] = np.nan
    return out


def _task_payloads(
    backgrounds: pd.DataFrame,
    seeds: Sequence[int],
    completed: set[tuple[str, float, int]],
    base_config_path: str | Path,
) -> Iterable[dict[str, Any]]:
    for row in backgrounds.to_dict(orient="records"):
        background_id = str(row["background_id"])
        for cancel in FOCAL_LEVELS:
            for seed in seeds:
                key = (background_id, float(cancel), int(seed))
                if key in completed:
                    continue
                yield {
                    "row": row,
                    "background_id": background_id,
                    "cancel_class1_focal": float(cancel),
                    "seed": int(seed),
                    "base_config_path": str(base_config_path),
                }


def _run_task(task: Mapping[str, Any]) -> dict[str, Any]:
    # Imported lazily so analysis-only commands and unit tests do not require
    # the simulation engine to be imported.
    from experiments.robustness.simulation_adapter import run_scenario

    row = dict(task["row"])
    cancel = float(task["cancel_class1_focal"])
    seed = int(task["seed"])
    metrics = run_scenario(
        row,
        seed=seed,
        base_config_path=task["base_config_path"],
        overrides={"cancel_class1": cancel},
    )
    return {
        "background_id": task["background_id"],
        "source_scenario_ids": _clean_string(row.get("source_scenario_ids")),
        "scenario_type": row["scenario_type"],
        "rho": float(row["rho"]),
        "class1_share": float(row["class1_share"]),
        "slots_per_day": int(row["slots_per_day"]),
        "horizon_class1": int(row["horizon_class1"]),
        "horizon_class2": int(row["horizon_class2"]),
        "cancel_class1_focal": cancel,
        "cancel_class2_background": float(row["cancel_class2"]),
        **metrics,
    }


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_stage1(
    *,
    scenarios_path: Path,
    output_dir: Path,
    base_config_path: Path,
    workers: int,
    smoke: bool,
    max_scenarios: int | None,
    resume: bool,
) -> Path:
    scenarios = pd.read_csv(scenarios_path)
    backgrounds = prepare_h1_backgrounds(scenarios)
    if smoke:
        backgrounds = backgrounds.head(2).copy()
        seeds = tuple(STAGE1_SEEDS[:2])
    else:
        seeds = STAGE1_SEEDS
    if max_scenarios is not None:
        backgrounds = backgrounds.head(max_scenarios).copy()

    background_path = output_dir / "design" / "h1_background_scenarios.csv"
    raw_path = output_dir / "raw" / "h1_stage1_raw.csv"
    background_path.parent.mkdir(parents=True, exist_ok=True)
    backgrounds.to_csv(background_path, index=False)

    completed: set[tuple[str, float, int]] = set()
    if resume and raw_path.exists():
        old = pd.read_csv(raw_path, usecols=["background_id", "cancel_class1_focal", "seed"])
        completed = {
            (str(r.background_id), float(r.cancel_class1_focal), int(r.seed))
            for r in old.itertuples(index=False)
        }
    elif raw_path.exists():
        raw_path.unlink()

    tasks = list(_task_payloads(backgrounds, seeds, completed, base_config_path))
    total_expected = len(backgrounds) * len(FOCAL_LEVELS) * len(seeds)
    print(f"H1 backgrounds: {len(backgrounds)}")
    print(f"H1 focal levels: {FOCAL_LEVELS}")
    print(f"Seeds: {len(seeds)}")
    print(f"Expected rows after completion: {total_expected}")
    print(f"Rows already completed: {len(completed)}")
    print(f"Rows to run now: {len(tasks)}")

    buffer: list[dict[str, Any]] = []
    flush_every = 100
    if workers <= 1:
        iterator = map(_run_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_run_task, tasks, chunksize=4)

    try:
        for index, result in enumerate(iterator, start=1):
            buffer.append(result)
            if len(buffer) >= flush_every:
                _append_rows(raw_path, buffer)
                buffer.clear()
            if index % 500 == 0 or index == len(tasks):
                print(f"Completed {index:,}/{len(tasks):,} new runs")
        _append_rows(raw_path, buffer)
    finally:
        if executor is not None:
            executor.shutdown()

    print(f"Raw results: {raw_path}")
    return raw_path


def _paired_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = int(len(clean))
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(clean.mean())
    if n == 1 or float(clean.std(ddof=1)) == 0.0:
        return mean, mean, mean, n
    sem = float(stats.sem(clean))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    half = critical * sem
    return mean, mean - half, mean + half, n


def _component_status(
    mean: float,
    low: float,
    high: float,
    *,
    expected: str,
    practical_threshold: float,
) -> str:
    if any(math.isnan(x) for x in (mean, low, high)):
        return "inconclusive"
    if expected == "negative":
        if mean <= -practical_threshold and high < 0:
            return "supported"
        if mean >= practical_threshold and low > 0:
            return "reversed"
    elif expected == "positive":
        if mean >= practical_threshold and low > 0:
            return "supported"
        if mean <= -practical_threshold and high < 0:
            return "reversed"
    else:
        raise ValueError(f"Unsupported expected direction: {expected}")
    return "inconclusive"


def _monotonic_fraction(group: pd.DataFrame, metric: str, expected: str) -> float:
    pivot = group.pivot(index="seed", columns="cancel_class1_focal", values=metric)
    required = list(FOCAL_LEVELS)
    pivot = pivot.dropna(subset=required)
    if pivot.empty:
        return math.nan
    diffs = np.diff(pivot[required].to_numpy(dtype=float), axis=1)
    if expected == "decreasing":
        per_seed = np.all(diffs <= 0, axis=1)
    elif expected == "increasing":
        per_seed = np.all(diffs >= 0, axis=1)
    else:
        raise ValueError(expected)
    return float(np.mean(per_seed))


def classify_stage1(*, raw_path: Path, output_dir: Path) -> Path:
    raw = pd.read_csv(raw_path)
    required = {
        "background_id",
        "seed",
        "cancel_class1_focal",
        "rho",
        "class_1_percent_serviced",
        "class_2_percent_serviced",
        "mean_offered_booking_delay",
        "average_utilization",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Raw H1 file is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for background_id, group in raw.groupby("background_id", sort=False):
        pivot = group.pivot(index="seed", columns="cancel_class1_focal")
        try:
            delta_r1 = (
                pivot["class_1_percent_serviced"][FOCAL_HIGH]
                - pivot["class_1_percent_serviced"][FOCAL_LOW]
            )
            delta_r2 = (
                pivot["class_2_percent_serviced"][FOCAL_HIGH]
                - pivot["class_2_percent_serviced"][FOCAL_LOW]
            )
            delta_delay = (
                pivot["mean_offered_booking_delay"][FOCAL_HIGH]
                - pivot["mean_offered_booking_delay"][FOCAL_LOW]
            )
            delta_util = (
                pivot["average_utilization"][FOCAL_HIGH]
                - pivot["average_utilization"][FOCAL_LOW]
            )
        except KeyError:
            continue

        r1_mean, r1_low, r1_high, n = _paired_ci(delta_r1)
        r2_mean, r2_low, r2_high, _ = _paired_ci(delta_r2)
        d_mean, d_low, d_high, _ = _paired_ci(delta_delay)
        u_mean, u_low, u_high, _ = _paired_ci(delta_util)

        r1_status = _component_status(
            r1_mean,
            r1_low,
            r1_high,
            expected="negative",
            practical_threshold=SERVED_THRESHOLD,
        )
        delay_status = _component_status(
            d_mean,
            d_low,
            d_high,
            expected="negative",
            practical_threshold=DELAY_THRESHOLD_DAYS,
        )

        rho = float(group["rho"].iloc[0])
        if rho >= HIGH_DEMAND_MIN:
            demand_regime = "high"
            if "reversed" in {r1_status, delay_status}:
                overall = "reversed"
            elif r1_status == "supported" and delay_status == "supported":
                overall = "supported"
            else:
                overall = "inconclusive"
        elif rho <= LOW_DEMAND_MAX:
            demand_regime = "low"
            if r1_status == "reversed":
                overall = "reversed"
            elif r1_status == "supported" and delay_status != "reversed":
                overall = "inactive"
            else:
                overall = "inconclusive"
        else:
            demand_regime = "boundary"
            if "reversed" in {r1_status, delay_status}:
                overall = "reversed"
            elif r1_status == "supported" and delay_status == "supported":
                overall = "supported"
            else:
                overall = "inconclusive"

        first = group.iloc[0]
        rows.append(
            {
                "background_id": background_id,
                "source_scenario_ids": first.get("source_scenario_ids", ""),
                "scenario_type": first["scenario_type"],
                "rho": rho,
                "demand_regime": demand_regime,
                "class1_share": float(first["class1_share"]),
                "slots_per_day": int(first["slots_per_day"]),
                "horizon_class1": int(first["horizon_class1"]),
                "horizon_class2": int(first["horizon_class2"]),
                "cancel_class2_background": float(first["cancel_class2_background"]),
                "n_paired_seeds": n,
                "delta_class1_served_rate": r1_mean,
                "delta_class1_served_rate_ci_low": r1_low,
                "delta_class1_served_rate_ci_high": r1_high,
                "class1_served_component": r1_status,
                "delta_class2_served_rate": r2_mean,
                "delta_class2_served_rate_ci_low": r2_low,
                "delta_class2_served_rate_ci_high": r2_high,
                "delta_mean_offered_delay": d_mean,
                "delta_mean_offered_delay_ci_low": d_low,
                "delta_mean_offered_delay_ci_high": d_high,
                "offered_delay_component": delay_status,
                "delta_average_utilization": u_mean,
                "delta_average_utilization_ci_low": u_low,
                "delta_average_utilization_ci_high": u_high,
                "class1_served_monotonic_fraction": _monotonic_fraction(
                    group, "class_1_percent_serviced", "decreasing"
                ),
                "offered_delay_monotonic_fraction": _monotonic_fraction(
                    group, "mean_offered_booking_delay", "decreasing"
                ),
                "classification": overall,
            }
        )

    effects = pd.DataFrame(rows)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    effects_path = summary_dir / "h1_scenario_effects.csv"
    effects.to_csv(effects_path, index=False)

    counts = (
        effects.groupby(["demand_regime", "classification"], dropna=False)
        .size()
        .rename("n_scenarios")
        .reset_index()
    )
    counts.to_csv(summary_dir / "h1_classification_counts.csv", index=False)

    failures = effects[
        effects["classification"].isin(["reversed", "inconclusive"])
    ].copy()
    failures.to_csv(summary_dir / "h1_failure_candidates.csv", index=False)

    stage2 = effects[
        (effects["classification"] == "reversed")
        | (
            (effects["classification"] == "inconclusive")
            & (effects["demand_regime"].isin(["high", "boundary"]))
        )
    ].copy()
    stage2.to_csv(summary_dir / "h1_stage2_candidates.csv", index=False)

    _write_summary_markdown(effects, summary_dir / "h1_stage1_summary.md")
    print(f"Scenario effects: {effects_path}")
    print(f"Stage 2 candidates: {summary_dir / 'h1_stage2_candidates.csv'}")
    return effects_path


def _write_summary_markdown(effects: pd.DataFrame, path: Path) -> None:
    lines = [
        "# H1 Stage 1 Robustness Summary",
        "",
        f"Background scenarios classified: **{len(effects)}**",
        "",
        "## Classification counts",
        "",
    ]
    if effects.empty:
        lines.append("No complete scenarios were available for classification.")
    else:
        table = (
            effects.groupby(["demand_regime", "classification"])
            .size()
            .unstack(fill_value=0)
        )
        lines.append(table.to_markdown())
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- High-demand support requires both a material reduction in Class 1 served rate and a material reduction in mean offered delay.",
                "- Low-demand scenarios can be classified as inactive when Class 1 served rate falls but the offered-delay rebooking mechanism does not activate.",
                "- Reversed and high-demand inconclusive scenarios are exported for Stage 2 confirmation with 100 new seeds.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "classify", "all"])
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_path = args.output_dir / "raw" / "h1_stage1_raw.csv"
    if args.command in {"run", "all"}:
        raw_path = run_stage1(
            scenarios_path=args.scenarios,
            output_dir=args.output_dir,
            base_config_path=args.base_config,
            workers=args.workers,
            smoke=args.smoke,
            max_scenarios=args.max_scenarios,
            resume=not args.no_resume,
        )
    if args.command in {"classify", "all"}:
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw H1 results not found: {raw_path}. Run the experiment first."
            )
        classify_stage1(raw_path=raw_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
