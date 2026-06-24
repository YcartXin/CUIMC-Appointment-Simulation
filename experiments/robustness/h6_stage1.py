"""Run and classify Stage 1 robustness tests for H6.

H6: Balking-threshold changes can produce nonlinear served-rate effects because
moving the threshold by one day reclassifies an entire offered-delay bucket.

For each background, Class 1's balking threshold is swept densely from 0 to
H1 - 2. For the transition tau -> tau + 1, the reclassified bucket is delay
tau + 1. The experiment relates that bucket's offered mass to the absolute
adjacent change in Class 1 served rate.

Run from the repository root:

    py -3 -m experiments.robustness.h6_stage1 all --smoke --workers 1 --no-resume
    py -3 -m experiments.robustness.h6_stage1 all --workers 4
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
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "robustness" / "h6"

SERVED_RATE_THRESHOLD = 0.005
BUCKET_MASS_THRESHOLD = 0.01
SPEARMAN_THRESHOLD = 0.50
MIN_TRANSITIONS = 3

FOCAL_COLUMNS = {"balk_threshold_class1"}


def valid_h6_thresholds(horizon: int) -> tuple[int, ...]:
    """Return the dense valid threshold sweep 0, ..., H - 2."""
    return tuple(range(0, max(0, int(horizon) - 1)))


def prepare_h6_backgrounds(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Remove the Class 1 balking threshold and deduplicate backgrounds."""
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
    signature_columns = [c for c in PARAMETER_COLUMNS if c not in FOCAL_COLUMNS]

    grouped_rows: list[dict[str, Any]] = []
    for _, group in df.groupby(signature_columns, dropna=False, sort=False):
        first = group.iloc[0].to_dict()
        first["source_scenario_ids"] = ";".join(
            group["source_scenario_id"].astype(str)
        )
        first["source_scenario_count"] = int(len(group))
        grouped_rows.append(first)

    out = pd.DataFrame(grouped_rows).reset_index(drop=True)
    out.insert(0, "background_id", [f"H6B{i:04d}" for i in range(1, len(out) + 1)])

    threshold_strings: list[str] = []
    active_values: list[bool] = []
    reasons_values: list[str] = []

    for row in out.to_dict(orient="records"):
        thresholds = valid_h6_thresholds(int(row["horizon_class1"]))
        step = float(row["balk_high_class1"]) - float(row["balk_low_class1"])
        reasons: list[str] = []
        if len(thresholds) - 1 < MIN_TRANSITIONS:
            reasons.append("fewer_than_three_adjacent_transitions")
        if abs(step) < 1e-12:
            reasons.append("no_within_class_balking_step")
        threshold_strings.append(";".join(str(x) for x in thresholds))
        active_values.append(not reasons)
        reasons_values.append(";".join(reasons))

    out["valid_h6_thresholds"] = threshold_strings
    out["h6_design_active"] = active_values
    out["h6_design_inactive_reason"] = reasons_values
    out["balk_threshold_class1"] = 0
    return out


def _parse_thresholds(value: Any) -> tuple[int, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    return tuple(int(float(x)) for x in text.split(";") if x != "")


def _task_payloads(
    backgrounds: pd.DataFrame,
    seeds: Sequence[int],
    completed: set[tuple[str, int, int]],
    base_config_path: str | Path,
) -> Iterable[dict[str, Any]]:
    for row in backgrounds.to_dict(orient="records"):
        if not bool(row["h6_design_active"]):
            continue
        background_id = str(row["background_id"])
        for threshold in _parse_thresholds(row["valid_h6_thresholds"]):
            for seed in seeds:
                key = (background_id, int(threshold), int(seed))
                if key in completed:
                    continue
                yield {
                    "row": row,
                    "background_id": background_id,
                    "threshold": int(threshold),
                    "seed": int(seed),
                    "base_config_path": str(base_config_path),
                }


def _run_task(task: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.robustness.simulation_adapter import (
        run_scenario_with_offered_delay_counts,
    )

    row = dict(task["row"])
    threshold = int(task["threshold"])
    seed = int(task["seed"])

    metrics = run_scenario_with_offered_delay_counts(
        row,
        seed=seed,
        base_config_path=task["base_config_path"],
        overrides={"balk_threshold_class1": threshold},
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
        "balk_threshold_class1_focal": threshold,
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
    backgrounds = prepare_h6_backgrounds(pd.read_csv(scenarios_path))
    if smoke:
        active = backgrounds[backgrounds["h6_design_active"]].head(2)
        inactive = backgrounds[~backgrounds["h6_design_active"]].head(1)
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

    design_path = output_dir / "design" / "h6_background_scenarios.csv"
    raw_path = output_dir / "raw" / "h6_stage1_raw.csv"
    design_path.parent.mkdir(parents=True, exist_ok=True)
    backgrounds.to_csv(design_path, index=False)

    completed: set[tuple[str, int, int]] = set()
    if resume and raw_path.exists():
        old = pd.read_csv(
            raw_path,
            usecols=["background_id", "balk_threshold_class1_focal", "seed"],
        )
        completed = {
            (
                str(row.background_id),
                int(row.balk_threshold_class1_focal),
                int(row.seed),
            )
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
    expected = sum(
        len(_parse_thresholds(row.valid_h6_thresholds)) * len(seeds)
        for row in backgrounds.itertuples(index=False)
        if bool(row.h6_design_active)
    )

    print(f"H6 backgrounds in this run: {len(backgrounds)}")
    print(f"Design-active backgrounds: {int(backgrounds['h6_design_active'].sum())}")
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


def _delay_count(value: Any, delay: int) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    if isinstance(value, dict):
        mapping = value
    else:
        mapping = json.loads(str(value))
    return int(mapping.get(str(delay), mapping.get(delay, 0)))


def _transition_effect_rows(
    design: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for design_row in design.to_dict(orient="records"):
        if not bool(design_row["h6_design_active"]):
            continue

        background_id = str(design_row["background_id"])
        group = raw[raw["background_id"].astype(str) == background_id]
        thresholds = _parse_thresholds(design_row["valid_h6_thresholds"])

        for lower_threshold, higher_threshold in zip(
            thresholds[:-1], thresholds[1:]
        ):
            bucket_delay = higher_threshold
            lower = group[
                group["balk_threshold_class1_focal"] == lower_threshold
            ].set_index("seed")
            higher = group[
                group["balk_threshold_class1_focal"] == higher_threshold
            ].set_index("seed")
            common = lower.index.intersection(higher.index)

            metadata = {
                "background_id": background_id,
                "source_scenario_ids": design_row.get("source_scenario_ids", ""),
                "scenario_type": design_row.get("scenario_type", ""),
                **{column: design_row[column] for column in PARAMETER_COLUMNS},
                "lower_threshold": int(lower_threshold),
                "higher_threshold": int(higher_threshold),
                "reclassified_delay_bucket": int(bucket_delay),
            }

            if len(common) == 0:
                rows.append(
                    {
                        **metadata,
                        "n_paired_seeds": 0,
                        "material_jump": False,
                    }
                )
                continue

            lower = lower.loc[common]
            higher = higher.loc[common]
            signed_delta = (
                higher["class_1_percent_serviced"]
                - lower["class_1_percent_serviced"]
            )

            lower_bucket_mass = pd.Series(
                [
                    _delay_count(value, bucket_delay) / offered
                    if offered > 0
                    else math.nan
                    for value, offered in zip(
                        lower["class_1_offered_delay_counts_json"],
                        lower["class_1_offered"],
                    )
                ],
                index=common,
                dtype=float,
            )
            higher_bucket_mass = pd.Series(
                [
                    _delay_count(value, bucket_delay) / offered
                    if offered > 0
                    else math.nan
                    for value, offered in zip(
                        higher["class_1_offered_delay_counts_json"],
                        higher["class_1_offered"],
                    )
                ],
                index=common,
                dtype=float,
            )
            bucket_mass = (lower_bucket_mass + higher_bucket_mass) / 2.0

            d_mean, d_low, d_high, n = _paired_ci(signed_delta)
            m_mean, m_low, m_high, _ = _paired_ci(bucket_mass)
            material_jump = bool(
                np.isfinite(d_mean)
                and abs(d_mean) >= SERVED_RATE_THRESHOLD
                and (
                    (d_low > 0 and d_high > 0)
                    or (d_low < 0 and d_high < 0)
                )
            )

            rows.append(
                {
                    **metadata,
                    "n_paired_seeds": n,
                    "mean_reclassified_bucket_mass": m_mean,
                    "mean_reclassified_bucket_mass_ci_low": m_low,
                    "mean_reclassified_bucket_mass_ci_high": m_high,
                    "signed_served_rate_jump": d_mean,
                    "signed_served_rate_jump_ci_low": d_low,
                    "signed_served_rate_jump_ci_high": d_high,
                    "absolute_served_rate_jump": abs(d_mean),
                    "material_jump": material_jump,
                }
            )

    return pd.DataFrame(rows)


def _scenario_effect_rows(
    design: pd.DataFrame,
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for design_row in design.to_dict(orient="records"):
        background_id = str(design_row["background_id"])
        metadata = {
            "background_id": background_id,
            "source_scenario_ids": design_row.get("source_scenario_ids", ""),
            "scenario_type": design_row.get("scenario_type", ""),
            **{column: design_row[column] for column in PARAMETER_COLUMNS},
            "valid_h6_thresholds": design_row["valid_h6_thresholds"],
        }

        if not bool(design_row["h6_design_active"]):
            rows.append(
                {
                    **metadata,
                    "n_transitions": 0,
                    "n_exposed_transitions": 0,
                    "failure_component": design_row[
                        "h6_design_inactive_reason"
                    ],
                    "classification": "inactive",
                }
            )
            continue

        group = transitions[
            transitions["background_id"].astype(str) == background_id
        ].dropna(
            subset=[
                "mean_reclassified_bucket_mass",
                "absolute_served_rate_jump",
            ]
        )

        if len(group) < MIN_TRANSITIONS:
            rows.append(
                {
                    **metadata,
                    "n_transitions": int(len(group)),
                    "n_exposed_transitions": int(
                        (
                            group["mean_reclassified_bucket_mass"]
                            >= BUCKET_MASS_THRESHOLD
                        ).sum()
                    ),
                    "failure_component": "fewer_than_three_complete_transitions",
                    "classification": "inactive",
                }
            )
            continue

        n_exposed = int(
            (
                group["mean_reclassified_bucket_mass"]
                >= BUCKET_MASS_THRESHOLD
            ).sum()
        )
        if n_exposed == 0:
            rows.append(
                {
                    **metadata,
                    "n_transitions": int(len(group)),
                    "n_exposed_transitions": 0,
                    "failure_component": "all_reclassified_bucket_masses_below_1pct",
                    "classification": "inactive",
                }
            )
            continue

        correlation = stats.spearmanr(
            group["mean_reclassified_bucket_mass"].astype(float),
            group["absolute_served_rate_jump"].astype(float),
        ).statistic
        spearman = float(correlation) if np.isfinite(correlation) else math.nan

        largest = group.loc[group["absolute_served_rate_jump"].idxmax()]
        median_mass = float(group["mean_reclassified_bucket_mass"].median())
        largest_mass = float(largest["mean_reclassified_bucket_mass"])
        largest_in_upper_half = bool(largest_mass >= median_mass)
        largest_material = bool(largest["material_jump"])

        if (
            np.isfinite(spearman)
            and spearman >= SPEARMAN_THRESHOLD
            and largest_in_upper_half
            and largest_material
        ):
            classification = "supported"
            failure_component = ""
        elif (
            np.isfinite(spearman)
            and spearman <= -SPEARMAN_THRESHOLD
            and not largest_in_upper_half
            and largest_material
        ):
            classification = "reversed"
            failure_component = "bucket_mass_jump_relationship_reversed"
        else:
            classification = "inconclusive"
            failures: list[str] = []
            if not np.isfinite(spearman) or spearman < SPEARMAN_THRESHOLD:
                failures.append("spearman_below_support_threshold")
            if not largest_in_upper_half:
                failures.append("largest_jump_not_in_upper_mass_half")
            if not largest_material:
                failures.append("largest_jump_not_material_and_precise")
            failure_component = ";".join(failures)

        rows.append(
            {
                **metadata,
                "n_transitions": int(len(group)),
                "n_exposed_transitions": n_exposed,
                "spearman_bucket_mass_vs_jump": spearman,
                "median_bucket_mass": median_mass,
                "largest_jump_lower_threshold": int(
                    largest["lower_threshold"]
                ),
                "largest_jump_higher_threshold": int(
                    largest["higher_threshold"]
                ),
                "largest_jump_bucket_delay": int(
                    largest["reclassified_delay_bucket"]
                ),
                "largest_jump_bucket_mass": largest_mass,
                "largest_absolute_served_rate_jump": float(
                    largest["absolute_served_rate_jump"]
                ),
                "largest_jump_material": largest_material,
                "largest_jump_in_upper_mass_half": largest_in_upper_half,
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

    transitions = _transition_effect_rows(design, raw)
    scenarios = _scenario_effect_rows(design, transitions)

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    transition_path = summary_dir / "h6_transition_effects.csv"
    scenario_path = summary_dir / "h6_scenario_effects.csv"
    transitions.to_csv(transition_path, index=False)
    scenarios.to_csv(scenario_path, index=False)

    counts = (
        scenarios.groupby("classification", dropna=False)
        .size()
        .rename("n_scenarios")
        .reset_index()
    )
    counts.to_csv(summary_dir / "h6_classification_counts.csv", index=False)

    failures = scenarios[
        scenarios["classification"].isin(["reversed", "inconclusive"])
    ].copy()
    failures.to_csv(summary_dir / "h6_failure_candidates.csv", index=False)
    failures.to_csv(summary_dir / "h6_stage2_candidates.csv", index=False)

    _write_summary_markdown(
        scenarios,
        transitions,
        counts,
        summary_dir / "h6_stage1_summary.md",
    )
    print(f"Scenario effects: {scenario_path}")
    print(f"Stage 2 candidates: {summary_dir / 'h6_stage2_candidates.csv'}")
    return scenario_path


def _write_summary_markdown(
    scenarios: pd.DataFrame,
    transitions: pd.DataFrame,
    counts: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# H6 Stage 1 Robustness Summary",
        "",
        f"Background scenarios classified: **{len(scenarios)}**",
        f"Adjacent threshold transitions evaluated: **{len(transitions)}**",
        "",
        "## Scenario classification counts",
        "",
        counts.to_markdown(index=False) if not counts.empty else "No results.",
        "",
        "## Interpretation",
        "",
        (
            "- Each transition from threshold tau to tau + 1 reclassifies the "
            "offered-delay bucket at tau + 1."
        ),
        (
            "- Support requires Spearman correlation of at least 0.50 between "
            "reclassified bucket mass and absolute served-rate jump."
        ),
        (
            "- The largest served-rate jump must occur in the upper half of "
            "the bucket-mass distribution and exceed 0.005 with a paired "
            "confidence interval excluding zero."
        ),
        (
            "- A scenario is inactive when there is no within-class balking "
            "step, fewer than three usable transitions, or all reclassified "
            "bucket masses are below 1%."
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
    design_path = args.output_dir / "design" / "h6_background_scenarios.csv"
    raw_path = args.output_dir / "raw" / "h6_stage1_raw.csv"

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
                f"H6 design file not found: {design_path}. Run the experiment first."
            )
        classify_stage1(
            design_path=design_path,
            raw_path=raw_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
