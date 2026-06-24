"""Run and classify Stage 1 robustness tests for H8.

H8: Holding Class 1's post-threshold balking probability fixed, an equal
increase in the between-class post-threshold balking gap has a larger absolute
effect on Class 1 served rate than an equal increase in Class 1's within-class
balking step.

Each background is assigned one balanced starting cell:

    S1 = b11 - b01
    G1 = b11 - b12

with b11 fixed at 0.50 and S1, G1 drawn cyclically from
{0.0, 0.1, 0.2, 0.3, 0.4}. Three paired configurations are run:

    baseline
    step_up: S1 increases by 0.10
    gap_up:  G1 increases by 0.10

The primary comparison is:

    D8 = |R1(gap_up) - R1(baseline)|
         - |R1(step_up) - R1(baseline)|

Run from the repository root:

    py -3 -m experiments.robustness.h8_stage1 all --smoke --workers 1 --no-resume
    py -3 -m experiments.robustness.h8_stage1 all --workers 4
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
    PARAMETER_COLUMNS,
    STAGE1_SEEDS,
)

DEFAULT_BASE_CONFIG = REPO_DIR / "configs" / "baseline.yaml"
DEFAULT_SCENARIOS = (
    REPO_DIR / "outputs" / "robustness" / "scenarios" / "all_stage1_scenarios.csv"
)
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "robustness" / "h8"

CLASS1_POST_FIXED = 0.50
CLASS2_PRE_FIXED = 0.00
INCREMENT = 0.10
START_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4)

EFFECT_THRESHOLD = 0.0025
EXPOSURE_THRESHOLD = 0.01

FOCAL_COLUMNS = {
    "balk_low_class1",
    "balk_high_class1",
    "balk_low_class2",
    "balk_high_class2",
}


def _start_cells() -> tuple[tuple[float, float], ...]:
    return tuple(
        (float(within_step), float(post_gap))
        for within_step in START_LEVELS
        for post_gap in START_LEVELS
    )


def prepare_h8_backgrounds(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Remove H8 focal rates, deduplicate, and assign balanced start cells."""
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
    signature_columns = [
        column for column in PARAMETER_COLUMNS if column not in FOCAL_COLUMNS
    ]

    rows: list[dict[str, Any]] = []
    for _, group in df.groupby(signature_columns, dropna=False, sort=False):
        first = group.iloc[0].to_dict()
        first["source_scenario_ids"] = ";".join(
            group["source_scenario_id"].astype(str)
        )
        first["source_scenario_count"] = int(len(group))
        rows.append(first)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out.insert(
        0,
        "background_id",
        [f"H8B{i:04d}" for i in range(1, len(out) + 1)],
    )

    cells = _start_cells()
    start_within: list[float] = []
    start_gap: list[float] = []
    c1_low: list[float] = []
    c2_high: list[float] = []

    for index in range(len(out)):
        within_step, post_gap = cells[index % len(cells)]
        start_within.append(within_step)
        start_gap.append(post_gap)
        c1_low.append(round(CLASS1_POST_FIXED - within_step, 2))
        c2_high.append(round(CLASS1_POST_FIXED - post_gap, 2))

    out["start_within_step"] = start_within
    out["start_post_gap"] = start_gap
    out["baseline_class1_low"] = c1_low
    out["baseline_class1_high"] = CLASS1_POST_FIXED
    out["baseline_class2_low"] = CLASS2_PRE_FIXED
    out["baseline_class2_high"] = c2_high
    out["h8_design_active"] = True
    out["h8_design_inactive_reason"] = ""

    # Placeholders keep the shared scenario adapter valid. Every H8 run
    # overwrites all four focal rates.
    out["balk_low_class1"] = out["baseline_class1_low"]
    out["balk_high_class1"] = CLASS1_POST_FIXED
    out["balk_low_class2"] = CLASS2_PRE_FIXED
    out["balk_high_class2"] = out["baseline_class2_high"]
    return out


def _arm_overrides(row: Mapping[str, Any], arm: str) -> dict[str, float]:
    c1_low = float(row["baseline_class1_low"])
    c2_high = float(row["baseline_class2_high"])

    overrides = {
        "balk_low_class1": c1_low,
        "balk_high_class1": CLASS1_POST_FIXED,
        "balk_low_class2": CLASS2_PRE_FIXED,
        "balk_high_class2": c2_high,
    }
    if arm == "step_up":
        overrides["balk_low_class1"] = round(c1_low - INCREMENT, 2)
    elif arm == "gap_up":
        overrides["balk_high_class2"] = round(c2_high - INCREMENT, 2)
    elif arm != "baseline":
        raise ValueError(f"Unknown H8 arm: {arm}")

    for key, value in overrides.items():
        if value < -1e-12 or value > 1.0 + 1e-12:
            raise ValueError(f"Invalid probability for {key}: {value}")
    if overrides["balk_high_class1"] < overrides["balk_low_class1"] - 1e-12:
        raise ValueError("Class 1 high balking rate is below its low rate.")
    if overrides["balk_high_class2"] < overrides["balk_low_class2"] - 1e-12:
        raise ValueError("Class 2 high balking rate is below its low rate.")

    return overrides


def _task_payloads(
    backgrounds: pd.DataFrame,
    seeds: Sequence[int],
    completed: set[tuple[str, str, int]],
    base_config_path: str | Path,
) -> Iterable[dict[str, Any]]:
    for row in backgrounds.to_dict(orient="records"):
        background_id = str(row["background_id"])
        for arm in ("baseline", "step_up", "gap_up"):
            overrides = _arm_overrides(row, arm)
            for seed in seeds:
                key = (background_id, arm, int(seed))
                if key in completed:
                    continue
                yield {
                    "row": row,
                    "background_id": background_id,
                    "arm": arm,
                    "overrides": overrides,
                    "seed": int(seed),
                    "base_config_path": str(base_config_path),
                }


def _delay_regime_shares(
    counts_json: Any,
    *,
    threshold: int,
    total_offered: float,
) -> tuple[float, float]:
    if total_offered <= 0:
        return math.nan, math.nan
    if isinstance(counts_json, dict):
        counts = counts_json
    else:
        counts = json.loads(str(counts_json))

    low_count = 0
    high_count = 0
    for raw_delay, raw_count in counts.items():
        delay = int(raw_delay)
        count = int(raw_count)
        if delay <= int(threshold):
            low_count += count
        else:
            high_count += count
    return low_count / total_offered, high_count / total_offered


def _run_task(task: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.robustness.simulation_adapter import (
        run_scenario_with_offered_delay_counts,
    )

    row = dict(task["row"])
    arm = str(task["arm"])
    seed = int(task["seed"])
    overrides = dict(task["overrides"])

    metrics = run_scenario_with_offered_delay_counts(
        row,
        seed=seed,
        base_config_path=task["base_config_path"],
        overrides=overrides,
    )

    c1_low_share, c1_high_share = _delay_regime_shares(
        metrics["class_1_offered_delay_counts_json"],
        threshold=int(row["balk_threshold_class1"]),
        total_offered=float(metrics["class_1_offered"]),
    )
    c2_low_share, c2_high_share = _delay_regime_shares(
        metrics["class_2_offered_delay_counts_json"],
        threshold=int(row["balk_threshold_class2"]),
        total_offered=float(metrics["class_2_offered"]),
    )

    return {
        "background_id": task["background_id"],
        "source_scenario_ids": str(row.get("source_scenario_ids", "")),
        "scenario_type": row["scenario_type"],
        "rho": float(row["rho"]),
        "class1_share": float(row["class1_share"]),
        "slots_per_day": int(row["slots_per_day"]),
        "horizon_class1": int(row["horizon_class1"]),
        "horizon_class2": int(row["horizon_class2"]),
        "start_within_step": float(row["start_within_step"]),
        "start_post_gap": float(row["start_post_gap"]),
        "h8_arm": arm,
        "class1_balk_low_focal": float(overrides["balk_low_class1"]),
        "class1_balk_high_focal": float(overrides["balk_high_class1"]),
        "class2_balk_low_focal": float(overrides["balk_low_class2"]),
        "class2_balk_high_focal": float(overrides["balk_high_class2"]),
        "class1_low_regime_offer_share": c1_low_share,
        "class1_high_regime_offer_share": c1_high_share,
        "class2_low_regime_offer_share": c2_low_share,
        "class2_high_regime_offer_share": c2_high_share,
        "seed": seed,
        **metrics,
    }


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def _select_backgrounds(
    scenarios_path: Path,
    *,
    smoke: bool,
    max_scenarios: int | None,
) -> pd.DataFrame:
    backgrounds = prepare_h8_backgrounds(pd.read_csv(scenarios_path))
    if smoke:
        backgrounds = backgrounds.head(2).copy()
    if max_scenarios is not None:
        backgrounds = backgrounds.head(max_scenarios).copy()
    return backgrounds


def run_stage1(
    *,
    scenarios_path: Path,
    output_dir: Path,
    base_config_path: Path,
    workers: int,
    smoke: bool,
    max_scenarios: int | None,
    resume: bool,
) -> tuple[Path, Path]:
    backgrounds = _select_backgrounds(
        scenarios_path,
        smoke=smoke,
        max_scenarios=max_scenarios,
    )
    seeds = tuple(STAGE1_SEEDS[:2]) if smoke else STAGE1_SEEDS

    design_path = output_dir / "design" / "h8_background_scenarios.csv"
    raw_path = output_dir / "raw" / "h8_stage1_raw.csv"
    design_path.parent.mkdir(parents=True, exist_ok=True)
    backgrounds.to_csv(design_path, index=False)

    completed: set[tuple[str, str, int]] = set()
    if resume and raw_path.exists():
        old = pd.read_csv(
            raw_path,
            usecols=["background_id", "h8_arm", "seed"],
        )
        completed = {
            (str(row.background_id), str(row.h8_arm), int(row.seed))
            for row in old.itertuples(index=False)
        }
    elif raw_path.exists():
        raw_path.unlink()

    tasks = list(
        _task_payloads(
            backgrounds,
            seeds,
            completed,
            base_config_path,
        )
    )
    expected = len(backgrounds) * 3 * len(seeds)

    print(f"H8 backgrounds in this run: {len(backgrounds)}")
    print("Arms: baseline, step_up, gap_up")
    print(f"Stage 1 seeds: {len(seeds)}")
    print(f"Expected rows after completion: {expected:,}")
    print(f"Rows already completed: {len(completed):,}")
    print(f"Rows to run now: {len(tasks):,}")

    buffer: list[dict[str, Any]] = []
    flush_every = 100

    if workers <= 1:
        iterator: Iterable[dict[str, Any]] = map(_run_task, tasks)
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
    return design_path, raw_path


def _paired_ci(values: pd.Series) -> tuple[float, float, float, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = int(len(clean))
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(clean.mean())
    if n == 1 or float(clean.std(ddof=1)) == 0.0:
        return mean, mean, mean, n
    sem = float(stats.sem(clean))
    critical = float(stats.t.ppf(0.975, df=n - 1))
    half = critical * sem
    return mean, mean - half, mean + half, n


def _classification(
    mean: float,
    low: float,
    high: float,
) -> str:
    if any(math.isnan(value) for value in (mean, low, high)):
        return "inconclusive"
    if mean >= EFFECT_THRESHOLD and low > 0:
        return "supported"
    if mean <= -EFFECT_THRESHOLD and high < 0:
        return "reversed"
    return "inconclusive"


def _scenario_effect_rows(
    design: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for design_row in design.to_dict(orient="records"):
        background_id = str(design_row["background_id"])
        metadata = {
            "background_id": background_id,
            "source_scenario_ids": design_row.get("source_scenario_ids", ""),
            "scenario_type": design_row.get("scenario_type", ""),
            **{column: design_row[column] for column in PARAMETER_COLUMNS},
            "start_within_step": float(design_row["start_within_step"]),
            "start_post_gap": float(design_row["start_post_gap"]),
            "baseline_class1_low": float(design_row["baseline_class1_low"]),
            "baseline_class1_high": float(design_row["baseline_class1_high"]),
            "baseline_class2_low": float(design_row["baseline_class2_low"]),
            "baseline_class2_high": float(design_row["baseline_class2_high"]),
        }

        group = raw[raw["background_id"].astype(str) == background_id]
        baseline = group[group["h8_arm"] == "baseline"].set_index("seed")
        step_up = group[group["h8_arm"] == "step_up"].set_index("seed")
        gap_up = group[group["h8_arm"] == "gap_up"].set_index("seed")
        common = baseline.index.intersection(step_up.index).intersection(gap_up.index)

        if len(common) == 0:
            rows.append(
                {
                    **metadata,
                    "n_paired_seeds": 0,
                    "failure_component": "no_complete_paired_seeds",
                    "classification": "inconclusive",
                }
            )
            continue

        baseline = baseline.loc[common]
        step_up = step_up.loc[common]
        gap_up = gap_up.loc[common]

        class1_low_exposure = float(
            pd.concat(
                [
                    baseline["class1_low_regime_offer_share"],
                    step_up["class1_low_regime_offer_share"],
                ]
            ).mean()
        )
        class1_high_exposure = float(
            pd.concat(
                [
                    baseline["class1_high_regime_offer_share"],
                    step_up["class1_high_regime_offer_share"],
                ]
            ).mean()
        )
        class2_high_exposure = float(
            pd.concat(
                [
                    baseline["class2_high_regime_offer_share"],
                    gap_up["class2_high_regime_offer_share"],
                ]
            ).mean()
        )

        exposure_active = bool(
            np.isfinite(class1_low_exposure)
            and np.isfinite(class1_high_exposure)
            and np.isfinite(class2_high_exposure)
            and class1_low_exposure >= EXPOSURE_THRESHOLD
            and class1_high_exposure >= EXPOSURE_THRESHOLD
            and class2_high_exposure >= EXPOSURE_THRESHOLD
        )

        delta_step = (
            step_up["class_1_percent_serviced"]
            - baseline["class_1_percent_serviced"]
        )
        delta_gap = (
            gap_up["class_1_percent_serviced"]
            - baseline["class_1_percent_serviced"]
        )
        difference = delta_gap.abs() - delta_step.abs()

        step_mean, step_low, step_high, n = _paired_ci(delta_step)
        gap_mean, gap_low, gap_high, _ = _paired_ci(delta_gap)
        diff_mean, diff_low, diff_high, _ = _paired_ci(difference)

        if not exposure_active:
            classification = "inactive"
            missing_exposure: list[str] = []
            if not np.isfinite(class1_low_exposure) or class1_low_exposure < EXPOSURE_THRESHOLD:
                missing_exposure.append("class1_pre_threshold")
            if not np.isfinite(class1_high_exposure) or class1_high_exposure < EXPOSURE_THRESHOLD:
                missing_exposure.append("class1_post_threshold")
            if not np.isfinite(class2_high_exposure) or class2_high_exposure < EXPOSURE_THRESHOLD:
                missing_exposure.append("class2_post_threshold")
            failure_component = (
                "insufficient_offer_exposure:" + ",".join(missing_exposure)
            )
        else:
            classification = _classification(diff_mean, diff_low, diff_high)
            if classification == "supported":
                failure_component = ""
            elif classification == "reversed":
                failure_component = "within_class_effect_larger"
            else:
                failure_component = "absolute_effect_difference_inconclusive"

        rows.append(
            {
                **metadata,
                "n_paired_seeds": n,
                "mean_class1_low_regime_offer_share": class1_low_exposure,
                "mean_class1_high_regime_offer_share": class1_high_exposure,
                "mean_class2_high_regime_offer_share": class2_high_exposure,
                "exposure_active": exposure_active,
                "delta_step_effect": step_mean,
                "delta_step_effect_ci_low": step_low,
                "delta_step_effect_ci_high": step_high,
                "absolute_step_effect": float(delta_step.abs().mean()),
                "delta_gap_effect": gap_mean,
                "delta_gap_effect_ci_low": gap_low,
                "delta_gap_effect_ci_high": gap_high,
                "absolute_gap_effect": float(delta_gap.abs().mean()),
                "absolute_effect_difference": diff_mean,
                "absolute_effect_difference_ci_low": diff_low,
                "absolute_effect_difference_ci_high": diff_high,
                "share_seeds_gap_effect_larger": float((difference > 0).mean()),
                "failure_component": failure_component,
                "classification": classification,
            }
        )

    return pd.DataFrame(rows)


def classify_stage1(
    *,
    design_path: Path,
    raw_path: Path,
    output_dir: Path,
) -> Path:
    design = pd.read_csv(design_path)
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    effects = _scenario_effect_rows(design, raw)

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    effects_path = summary_dir / "h8_scenario_effects.csv"
    effects.to_csv(effects_path, index=False)

    counts = (
        effects.groupby("classification", dropna=False)
        .size()
        .rename("n_scenarios")
        .reset_index()
    )
    counts.to_csv(summary_dir / "h8_classification_counts.csv", index=False)

    cell_counts = (
        effects.groupby(
            ["start_within_step", "start_post_gap", "classification"],
            dropna=False,
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    cell_counts.to_csv(
        summary_dir / "h8_start_cell_classification_counts.csv",
        index=False,
    )

    failures = effects[
        effects["classification"].isin(["reversed", "inconclusive"])
    ].copy()
    failures.to_csv(summary_dir / "h8_failure_candidates.csv", index=False)
    failures.to_csv(summary_dir / "h8_stage2_candidates.csv", index=False)

    _write_summary_markdown(
        effects,
        counts,
        cell_counts,
        summary_dir / "h8_stage1_summary.md",
    )
    print(f"Scenario effects: {effects_path}")
    print(f"Stage 2 candidates: {summary_dir / 'h8_stage2_candidates.csv'}")
    return effects_path


def _write_summary_markdown(
    effects: pd.DataFrame,
    counts: pd.DataFrame,
    cell_counts: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# H8 Stage 1 Robustness Summary",
        "",
        f"Background scenarios classified: **{len(effects)}**",
        "",
        "## Scenario classification counts",
        "",
        counts.to_markdown(index=False) if not counts.empty else "No results.",
        "",
        "## Starting-cell classification counts",
        "",
        cell_counts.to_markdown(index=False) if not cell_counts.empty else "No results.",
        "",
        "## Interpretation",
        "",
        (
            "- Every background uses three paired configurations: baseline, "
            "a 0.10 increase in Class 1's within-class balking step, and a "
            "0.10 increase in the between-class post-threshold gap."
        ),
        (
            "- Class 1's post-threshold balking probability is fixed at 0.50, "
            "and Class 2's pre-threshold probability is fixed at 0.00."
        ),
        (
            "- Support requires the between-class gap change to have an "
            "absolute Class 1 served-rate effect at least 0.0025 larger than "
            "the within-class step change, with a paired 95% confidence "
            "interval above zero."
        ),
        (
            "- A scenario is inactive when fewer than 1% of relevant offers "
            "reach the Class 1 pre-threshold, Class 1 post-threshold, or Class 2 "
            "post-threshold region."
        ),
        (
            "- Active reversed and inconclusive backgrounds are exported for "
            "Stage 2 confirmation."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "classify", "all"])
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    design_path = args.output_dir / "design" / "h8_background_scenarios.csv"
    raw_path = args.output_dir / "raw" / "h8_stage1_raw.csv"

    if args.command in {"run", "all"}:
        design_path, raw_path = run_stage1(
            scenarios_path=args.scenarios,
            output_dir=args.output_dir,
            base_config_path=args.base_config,
            workers=args.workers,
            smoke=args.smoke,
            max_scenarios=args.max_scenarios,
            resume=not args.no_resume,
        )

    if args.command in {"classify", "all"}:
        if not design_path.exists():
            raise FileNotFoundError(
                f"H8 design file not found: {design_path}. Run the experiment first."
            )
        classify_stage1(
            design_path=design_path,
            raw_path=raw_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
