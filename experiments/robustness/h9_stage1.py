"""Run and classify Stage 1 robustness tests for H9.

H9: A common increase in both classes' post-threshold no-show probabilities
has a larger effect on aggregate utilization, while a fixed-average increase
in the between-class difference has a larger effect on the served-rate gap.

For an assigned equal baseline probability p, four paired arms are run:

    baseline:       (p, p)
    common_up:      (p + 0.10, p + 0.10)
    gap_c1_higher:  (p + 0.05, p - 0.05)
    gap_c2_higher:  (p - 0.05, p + 0.05)

The two gap orientations are averaged so the result does not depend on which
class is assigned the higher no-show probability.

Run from the repository root:

    py -3 -m experiments.robustness.h9_stage1 all --smoke --workers 1 --no-resume
    py -3 -m experiments.robustness.h9_stage1 all --workers 4
"""

from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "robustness" / "h9"

BASELINE_P_LEVELS = (0.10, 0.30, 0.50, 0.70, 0.80)
COMMON_INCREMENT = 0.10
HALF_GAP = 0.05

EFFECT_THRESHOLD = 0.0025
EXPOSURE_THRESHOLD = 0.01
MIN_CLASS_ARRIVALS = 100.0

FOCAL_COLUMNS = {
    "noshow_high_class1",
    "noshow_high_class2",
}


def feasible_baseline_probabilities(
    noshow_low_class1: float,
    noshow_low_class2: float,
) -> tuple[float, ...]:
    """Return p values for which all four H9 arms remain valid."""
    minimum = max(float(noshow_low_class1), float(noshow_low_class2)) + HALF_GAP
    maximum = 1.0 - COMMON_INCREMENT
    return tuple(
        float(p)
        for p in BASELINE_P_LEVELS
        if p >= minimum - 1e-12 and p <= maximum + 1e-12
    )


def prepare_h9_backgrounds(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Remove focal high no-show rates, deduplicate, and assign baseline p."""
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
        [f"H9B{i:04d}" for i in range(1, len(out) + 1)],
    )

    assigned_p: list[float] = []
    active_values: list[bool] = []
    inactive_reasons: list[str] = []

    for index, row in enumerate(out.to_dict(orient="records")):
        feasible = feasible_baseline_probabilities(
            row["noshow_low_class1"],
            row["noshow_low_class2"],
        )
        reasons: list[str] = []
        if not feasible:
            reasons.append("no_valid_equal_baseline_probability")
            p = math.nan
        else:
            p = feasible[index % len(feasible)]

        assigned_p.append(float(p))
        active_values.append(not reasons)
        inactive_reasons.append(";".join(reasons))

    out["baseline_equal_noshow_high"] = assigned_p
    out["h9_design_active"] = active_values
    out["h9_design_inactive_reason"] = inactive_reasons

    # Placeholders for the shared adapter. Every run overwrites both values.
    out["noshow_high_class1"] = out["baseline_equal_noshow_high"]
    out["noshow_high_class2"] = out["baseline_equal_noshow_high"]
    return out


def _arm_overrides(row: Mapping[str, Any], arm: str) -> dict[str, float]:
    p = float(row["baseline_equal_noshow_high"])
    if arm == "baseline":
        values = (p, p)
    elif arm == "common_up":
        values = (p + COMMON_INCREMENT, p + COMMON_INCREMENT)
    elif arm == "gap_c1_higher":
        values = (p + HALF_GAP, p - HALF_GAP)
    elif arm == "gap_c2_higher":
        values = (p - HALF_GAP, p + HALF_GAP)
    else:
        raise ValueError(f"Unknown H9 arm: {arm}")

    overrides = {
        "noshow_high_class1": round(values[0], 2),
        "noshow_high_class2": round(values[1], 2),
    }

    for key, value in overrides.items():
        if value < -1e-12 or value > 1.0 + 1e-12:
            raise ValueError(f"Invalid probability for {key}: {value}")

    if (
        overrides["noshow_high_class1"]
        < float(row["noshow_low_class1"]) - 1e-12
    ):
        raise ValueError("Class 1 high no-show rate is below its low rate.")
    if (
        overrides["noshow_high_class2"]
        < float(row["noshow_low_class2"]) - 1e-12
    ):
        raise ValueError("Class 2 high no-show rate is below its low rate.")

    return overrides


def _task_payloads(
    backgrounds: pd.DataFrame,
    seeds: Sequence[int],
    completed: set[tuple[str, str, int]],
    base_config_path: str | Path,
) -> Iterable[dict[str, Any]]:
    for row in backgrounds.to_dict(orient="records"):
        if not bool(row["h9_design_active"]):
            continue

        background_id = str(row["background_id"])
        for arm in (
            "baseline",
            "common_up",
            "gap_c1_higher",
            "gap_c2_higher",
        ):
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


def _run_task(task: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.robustness.simulation_adapter import run_scenario

    row = dict(task["row"])
    arm = str(task["arm"])
    seed = int(task["seed"])
    overrides = dict(task["overrides"])

    metrics = run_scenario(
        row,
        seed=seed,
        base_config_path=task["base_config_path"],
        overrides=overrides,
    )

    served_rate_gap = (
        float(metrics["class_1_percent_serviced"])
        - float(metrics["class_2_percent_serviced"])
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
        "baseline_equal_noshow_high": float(
            row["baseline_equal_noshow_high"]
        ),
        "h9_arm": arm,
        "class1_noshow_high_focal": float(
            overrides["noshow_high_class1"]
        ),
        "class2_noshow_high_focal": float(
            overrides["noshow_high_class2"]
        ),
        "served_rate_gap": served_rate_gap,
        "absolute_served_rate_gap": abs(served_rate_gap),
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
    backgrounds = prepare_h9_backgrounds(pd.read_csv(scenarios_path))
    if smoke:
        active = backgrounds[backgrounds["h9_design_active"]].head(2)
        inactive = backgrounds[~backgrounds["h9_design_active"]].head(1)
        backgrounds = pd.concat([active, inactive], ignore_index=True)
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

    design_path = output_dir / "design" / "h9_background_scenarios.csv"
    raw_path = output_dir / "raw" / "h9_stage1_raw.csv"
    design_path.parent.mkdir(parents=True, exist_ok=True)
    backgrounds.to_csv(design_path, index=False)

    completed: set[tuple[str, str, int]] = set()
    if resume and raw_path.exists():
        old = pd.read_csv(
            raw_path,
            usecols=["background_id", "h9_arm", "seed"],
        )
        completed = {
            (str(row.background_id), str(row.h9_arm), int(row.seed))
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
    expected = (
        int(backgrounds["h9_design_active"].sum())
        * 4
        * len(seeds)
    )

    print(f"H9 backgrounds in this run: {len(backgrounds)}")
    print(f"Design-active backgrounds: {int(backgrounds['h9_design_active'].sum())}")
    print("Arms: baseline, common_up, gap_c1_higher, gap_c2_higher")
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


def _positive_status(
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


def _estimate_post_threshold_share(
    baseline: pd.DataFrame,
    common: pd.DataFrame,
    *,
    class_id: int,
) -> pd.Series:
    """Estimate accepted post-threshold exposure from the common-rate arm.

    Raising the post-threshold no-show probability by 0.10 creates additional
    no-shows only among accepted appointments in that region. Dividing the
    paired no-show-count increase by 0.10 times the average booked count gives
    a realized exposure estimate.
    """
    no_show_column = f"class_{class_id}_no_show"
    booked_column = f"class_{class_id}_booked"

    additional_no_shows = (
        common[no_show_column] - baseline[no_show_column]
    )
    average_booked = (
        common[booked_column] + baseline[booked_column]
    ) / 2.0
    denominator = COMMON_INCREMENT * average_booked.replace(0, np.nan)
    return (additional_no_shows / denominator).clip(lower=0.0, upper=1.0)


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
            "baseline_equal_noshow_high": design_row[
                "baseline_equal_noshow_high"
            ],
        }

        if not bool(design_row["h9_design_active"]):
            rows.append(
                {
                    **metadata,
                    "n_paired_seeds": 0,
                    "failure_component": design_row[
                        "h9_design_inactive_reason"
                    ],
                    "classification": "inactive",
                }
            )
            continue

        group = raw[raw["background_id"].astype(str) == background_id]
        baseline = group[group["h9_arm"] == "baseline"].set_index("seed")
        common = group[group["h9_arm"] == "common_up"].set_index("seed")
        gap_1 = group[group["h9_arm"] == "gap_c1_higher"].set_index("seed")
        gap_2 = group[group["h9_arm"] == "gap_c2_higher"].set_index("seed")

        common_seeds = (
            baseline.index.intersection(common.index)
            .intersection(gap_1.index)
            .intersection(gap_2.index)
        )
        if len(common_seeds) == 0:
            rows.append(
                {
                    **metadata,
                    "n_paired_seeds": 0,
                    "failure_component": "no_complete_paired_seeds",
                    "classification": "inconclusive",
                }
            )
            continue

        baseline = baseline.loc[common_seeds]
        common = common.loc[common_seeds]
        gap_1 = gap_1.loc[common_seeds]
        gap_2 = gap_2.loc[common_seeds]

        class1_arrivals = float(
            pd.concat(
                [
                    baseline["class_1_arrivals"],
                    common["class_1_arrivals"],
                    gap_1["class_1_arrivals"],
                    gap_2["class_1_arrivals"],
                ]
            ).mean()
        )
        class2_arrivals = float(
            pd.concat(
                [
                    baseline["class_2_arrivals"],
                    common["class_2_arrivals"],
                    gap_1["class_2_arrivals"],
                    gap_2["class_2_arrivals"],
                ]
            ).mean()
        )

        exposure_1 = _estimate_post_threshold_share(
            baseline,
            common,
            class_id=1,
        )
        exposure_2 = _estimate_post_threshold_share(
            baseline,
            common,
            class_id=2,
        )
        exposure_1_mean = float(exposure_1.mean())
        exposure_2_mean = float(exposure_2.mean())

        exposure_active = bool(
            class1_arrivals >= MIN_CLASS_ARRIVALS
            and class2_arrivals >= MIN_CLASS_ARRIVALS
            and np.isfinite(exposure_1_mean)
            and np.isfinite(exposure_2_mean)
            and exposure_1_mean >= EXPOSURE_THRESHOLD
            and exposure_2_mean >= EXPOSURE_THRESHOLD
        )

        common_utilization_effect = (
            common["average_utilization"]
            - baseline["average_utilization"]
        ).abs()
        gap_utilization_effect = (
            (
                gap_1["average_utilization"]
                - baseline["average_utilization"]
            ).abs()
            + (
                gap_2["average_utilization"]
                - baseline["average_utilization"]
            ).abs()
        ) / 2.0
        utilization_advantage = (
            common_utilization_effect - gap_utilization_effect
        )

        common_gap_effect = (
            common["served_rate_gap"] - baseline["served_rate_gap"]
        ).abs()
        between_gap_effect = (
            (
                gap_1["served_rate_gap"] - baseline["served_rate_gap"]
            ).abs()
            + (
                gap_2["served_rate_gap"] - baseline["served_rate_gap"]
            ).abs()
        ) / 2.0
        served_gap_advantage = between_gap_effect - common_gap_effect

        u_mean, u_low, u_high, n = _paired_ci(utilization_advantage)
        c_mean, c_low, c_high, _ = _paired_ci(served_gap_advantage)
        common_u_mean, _, _, _ = _paired_ci(common_utilization_effect)
        gap_u_mean, _, _, _ = _paired_ci(gap_utilization_effect)
        common_c_mean, _, _, _ = _paired_ci(common_gap_effect)
        gap_c_mean, _, _, _ = _paired_ci(between_gap_effect)

        utilization_status = _positive_status(u_mean, u_low, u_high)
        served_gap_status = _positive_status(c_mean, c_low, c_high)

        if not exposure_active:
            classification = "inactive"
            reasons: list[str] = []
            if class1_arrivals < MIN_CLASS_ARRIVALS:
                reasons.append("class1_too_small")
            if class2_arrivals < MIN_CLASS_ARRIVALS:
                reasons.append("class2_too_small")
            if (
                not np.isfinite(exposure_1_mean)
                or exposure_1_mean < EXPOSURE_THRESHOLD
            ):
                reasons.append("class1_post_threshold_exposure_below_1pct")
            if (
                not np.isfinite(exposure_2_mean)
                or exposure_2_mean < EXPOSURE_THRESHOLD
            ):
                reasons.append("class2_post_threshold_exposure_below_1pct")
            failure_component = ";".join(reasons)
        elif (
            utilization_status == "supported"
            and served_gap_status == "supported"
        ):
            classification = "supported"
            failure_component = ""
        elif (
            utilization_status == "reversed"
            or served_gap_status == "reversed"
        ):
            classification = "reversed"
            failures: list[str] = []
            if utilization_status == "reversed":
                failures.append("utilization_component_reversed")
            if served_gap_status == "reversed":
                failures.append("served_gap_component_reversed")
            failure_component = ";".join(failures)
        else:
            classification = "inconclusive"
            failures = []
            if utilization_status != "supported":
                failures.append("utilization_component_inconclusive")
            if served_gap_status != "supported":
                failures.append("served_gap_component_inconclusive")
            failure_component = ";".join(failures)

        rows.append(
            {
                **metadata,
                "n_paired_seeds": n,
                "mean_class1_arrivals": class1_arrivals,
                "mean_class2_arrivals": class2_arrivals,
                "estimated_class1_post_threshold_accepted_share": exposure_1_mean,
                "estimated_class2_post_threshold_accepted_share": exposure_2_mean,
                "exposure_active": exposure_active,
                "mean_absolute_common_utilization_effect": common_u_mean,
                "mean_absolute_gap_utilization_effect": gap_u_mean,
                "common_minus_gap_utilization_effect": u_mean,
                "common_minus_gap_utilization_effect_ci_low": u_low,
                "common_minus_gap_utilization_effect_ci_high": u_high,
                "mean_absolute_common_served_gap_effect": common_c_mean,
                "mean_absolute_between_gap_effect": gap_c_mean,
                "gap_minus_common_served_gap_effect": c_mean,
                "gap_minus_common_served_gap_effect_ci_low": c_low,
                "gap_minus_common_served_gap_effect_ci_high": c_high,
                "utilization_component": utilization_status,
                "served_gap_component": served_gap_status,
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

    effects_path = summary_dir / "h9_scenario_effects.csv"
    effects.to_csv(effects_path, index=False)

    counts = (
        effects.groupby("classification", dropna=False)
        .size()
        .rename("n_scenarios")
        .reset_index()
    )
    counts.to_csv(summary_dir / "h9_classification_counts.csv", index=False)

    component_counts = (
        effects[effects["classification"] != "inactive"]
        .groupby(
            ["utilization_component", "served_gap_component"],
            dropna=False,
        )
        .size()
        .rename("n_scenarios")
        .reset_index()
    )
    component_counts.to_csv(
        summary_dir / "h9_component_classification_counts.csv",
        index=False,
    )

    p_counts = (
        effects.groupby(
            ["baseline_equal_noshow_high", "classification"],
            dropna=False,
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    p_counts.to_csv(
        summary_dir / "h9_baseline_probability_counts.csv",
        index=False,
    )

    failures = effects[
        effects["classification"].isin(["reversed", "inconclusive"])
    ].copy()
    failures.to_csv(summary_dir / "h9_failure_candidates.csv", index=False)
    failures.to_csv(summary_dir / "h9_stage2_candidates.csv", index=False)

    _write_summary_markdown(
        effects,
        counts,
        component_counts,
        p_counts,
        summary_dir / "h9_stage1_summary.md",
    )
    print(f"Scenario effects: {effects_path}")
    print(f"Stage 2 candidates: {summary_dir / 'h9_stage2_candidates.csv'}")
    return effects_path


def _write_summary_markdown(
    effects: pd.DataFrame,
    counts: pd.DataFrame,
    component_counts: pd.DataFrame,
    p_counts: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# H9 Stage 1 Robustness Summary",
        "",
        f"Background scenarios classified: **{len(effects)}**",
        "",
        "## Scenario classification counts",
        "",
        counts.to_markdown(index=False) if not counts.empty else "No results.",
        "",
        "## Active component combinations",
        "",
        (
            component_counts.to_markdown(index=False)
            if not component_counts.empty
            else "No active results."
        ),
        "",
        "## Classification by equal baseline probability",
        "",
        p_counts.to_markdown(index=False) if not p_counts.empty else "No results.",
        "",
        "## Interpretation",
        "",
        (
            "- The common arm raises both post-threshold no-show probabilities "
            "by 0.10."
        ),
        (
            "- The two gap arms increase the between-class difference by 0.10 "
            "while preserving the average probability; both orientations are "
            "averaged."
        ),
        (
            "- Support requires the common change to have an aggregate "
            "utilization effect at least 0.0025 larger than the gap change."
        ),
        (
            "- Support also requires the gap change to have a served-rate-gap "
            "effect at least 0.0025 larger than the common change."
        ),
        (
            "- Both paired 95% confidence intervals must be above zero."
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
    design_path = args.output_dir / "design" / "h9_background_scenarios.csv"
    raw_path = args.output_dir / "raw" / "h9_stage1_raw.csv"

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
                f"H9 design file not found: {design_path}. Run the experiment first."
            )
        classify_stage1(
            design_path=design_path,
            raw_path=raw_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
