"""Run and classify Stage 1 robustness tests for H5.

H5: A higher Class 1 balking step lowers accepted booking delay mainly through
selection at low-to-moderate step sizes, rather than through congestion relief.

For each background, Class 1's pre-threshold balking probability remains fixed.
The focal step S1 = b1_1 - b0_1 is set to 0.0, 0.1, 0.3, or 0.5, provided the
resulting post-threshold probability does not exceed 0.70. The 0.1 and 0.3
steps are the primary low-to-moderate comparisons; 0.5 is diagnostic.

Run from the repository root:

    py -3 -m experiments.robustness.h5_stage1 all --smoke --workers 1 --no-resume
    py -3 -m experiments.robustness.h5_stage1 all --workers 4
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
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "robustness" / "h5"

STEP_LEVELS = (0.0, 0.1, 0.3, 0.5)
PRIMARY_STEPS = (0.1, 0.3)
MAX_POST_THRESHOLD_BALK = 0.70

DELAY_THRESHOLD = 0.25
SERVED_RATE_THRESHOLD = 0.005
POST_THRESHOLD_EXPOSURE_THRESHOLD = 0.01

FOCAL_COLUMNS = {"balk_high_class1"}


def _step_key(step: float) -> str:
    return f"{float(step):.1f}"


def valid_h5_steps(pre_threshold_rate: float) -> tuple[float, ...]:
    """Return step sizes that keep Class 1's high rate at or below 0.70."""
    return tuple(
        float(step)
        for step in STEP_LEVELS
        if float(pre_threshold_rate) + float(step)
        <= MAX_POST_THRESHOLD_BALK + 1e-12
    )


def prepare_h5_backgrounds(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Remove the H5 focal variable and deduplicate background scenarios."""
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
    out.insert(0, "background_id", [f"H5B{i:04d}" for i in range(1, len(out) + 1)])

    valid_strings: list[str] = []
    active: list[bool] = []
    inactive_reasons: list[str] = []

    for row in out.to_dict(orient="records"):
        pre = float(row["balk_low_class1"])
        steps = valid_h5_steps(pre)
        primary = [step for step in steps if step in PRIMARY_STEPS]
        reasons: list[str] = []
        if not primary:
            reasons.append("no_valid_low_to_moderate_step_increase")
        valid_strings.append(";".join(_step_key(step) for step in steps))
        active.append(not reasons)
        inactive_reasons.append(";".join(reasons))

    out["valid_h5_steps"] = valid_strings
    out["h5_design_active"] = active
    out["h5_design_inactive_reason"] = inactive_reasons

    # The runner overwrites this for each focal step.
    out["balk_high_class1"] = out["balk_low_class1"].astype(float)
    return out


def _parse_steps(value: Any) -> tuple[float, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    return tuple(float(x) for x in text.split(";") if x != "")


def _task_payloads(
    backgrounds: pd.DataFrame,
    seeds: Sequence[int],
    completed: set[tuple[str, float, int]],
    base_config_path: str | Path,
) -> Iterable[dict[str, Any]]:
    for row in backgrounds.to_dict(orient="records"):
        if not bool(row["h5_design_active"]):
            continue
        background_id = str(row["background_id"])
        pre = float(row["balk_low_class1"])
        for step in _parse_steps(row["valid_h5_steps"]):
            post = pre + float(step)
            for seed in seeds:
                key = (background_id, float(step), int(seed))
                if key in completed:
                    continue
                yield {
                    "row": row,
                    "background_id": background_id,
                    "step": float(step),
                    "post": float(post),
                    "seed": int(seed),
                    "base_config_path": str(base_config_path),
                }


def _run_task(task: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.robustness.simulation_adapter import run_scenario

    row = dict(task["row"])
    step = float(task["step"])
    post = float(task["post"])
    seed = int(task["seed"])

    metrics = run_scenario(
        row,
        seed=seed,
        base_config_path=task["base_config_path"],
        overrides={"balk_high_class1": post},
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
        "balk_low_class1_background": float(row["balk_low_class1"]),
        "balk_step_class1_focal": step,
        "balk_high_class1_focal": post,
        "is_primary_step": bool(step in PRIMARY_STEPS),
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
    backgrounds = prepare_h5_backgrounds(pd.read_csv(scenarios_path))
    if smoke:
        active = backgrounds[backgrounds["h5_design_active"]].head(2)
        inactive = backgrounds[~backgrounds["h5_design_active"]].head(1)
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

    design_path = output_dir / "design" / "h5_background_scenarios.csv"
    raw_path = output_dir / "raw" / "h5_stage1_raw.csv"
    design_path.parent.mkdir(parents=True, exist_ok=True)
    backgrounds.to_csv(design_path, index=False)

    completed: set[tuple[str, float, int]] = set()
    if resume and raw_path.exists():
        old = pd.read_csv(
            raw_path,
            usecols=["background_id", "balk_step_class1_focal", "seed"],
        )
        completed = {
            (
                str(row.background_id),
                float(row.balk_step_class1_focal),
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
        len(_parse_steps(row.valid_h5_steps)) * len(seeds)
        for row in backgrounds.itertuples(index=False)
        if bool(row.h5_design_active)
    )

    print(f"H5 backgrounds in this run: {len(backgrounds)}")
    print(f"Design-active backgrounds: {int(backgrounds['h5_design_active'].sum())}")
    print(f"Primary steps: {PRIMARY_STEPS}; diagnostic step: 0.5")
    print(f"Stage 1 seeds: {len(seeds)}")
    print(f"Expected rows after completion: {expected}")
    print(f"Rows already completed: {len(completed)}")
    print(f"Rows to run now: {len(tasks)}")

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


def _negative_status(
    mean: float,
    low: float,
    high: float,
    threshold: float,
) -> str:
    if any(math.isnan(x) for x in (mean, low, high)):
        return "inconclusive"
    if mean <= -threshold and high < 0:
        return "supported"
    if mean >= threshold and low > 0:
        return "reversed"
    return "inconclusive"


def _positive_status(
    mean: float,
    low: float,
    high: float,
    threshold: float,
) -> str:
    if any(math.isnan(x) for x in (mean, low, high)):
        return "inconclusive"
    if mean >= threshold and low > 0:
        return "supported"
    if mean <= -threshold and high < 0:
        return "reversed"
    return "inconclusive"


def _target_effect_rows(
    design: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for design_row in design.to_dict(orient="records"):
        if not bool(design_row["h5_design_active"]):
            continue

        background_id = str(design_row["background_id"])
        group = raw[raw["background_id"].astype(str) == background_id]
        baseline = group[group["balk_step_class1_focal"] == 0.0].set_index("seed")

        for step in _parse_steps(design_row["valid_h5_steps"]):
            if step == 0.0:
                continue
            treatment = group[
                np.isclose(group["balk_step_class1_focal"].astype(float), step)
            ].set_index("seed")
            common = baseline.index.intersection(treatment.index)

            metadata = {
                "background_id": background_id,
                "source_scenario_ids": design_row.get("source_scenario_ids", ""),
                "scenario_type": design_row.get("scenario_type", ""),
                **{column: design_row[column] for column in PARAMETER_COLUMNS},
                "balk_step_class1_focal": float(step),
                "balk_high_class1_focal": (
                    float(design_row["balk_low_class1"]) + float(step)
                ),
                "is_primary_step": bool(step in PRIMARY_STEPS),
            }

            if len(common) == 0:
                rows.append(
                    {
                        **metadata,
                        "n_paired_seeds": 0,
                        "classification": "inconclusive",
                        "failure_component": "no_paired_seeds",
                    }
                )
                continue

            base = baseline.loc[common]
            high = treatment.loc[common]

            delta_accepted = (
                high["class_1_mean_accepted_booking_delay"]
                - base["class_1_mean_accepted_booking_delay"]
            )
            delta_offered = (
                high["class_1_mean_offered_booking_delay"]
                - base["class_1_mean_offered_booking_delay"]
            )
            selection_gap = delta_offered - delta_accepted
            delta_served = high["class_1_percent_serviced"] - base[
                "class_1_percent_serviced"
            ]

            mean_offered_count = (
                high["class_1_offered"] + base["class_1_offered"]
            ) / 2.0
            additional_balked = (
                high["class_1_balked"] - base["class_1_balked"]
            )
            estimated_exposure = (
                additional_balked
                / (float(step) * mean_offered_count.replace(0, np.nan))
            ).clip(lower=0.0, upper=1.0)

            a_mean, a_low, a_high, n = _paired_ci(delta_accepted)
            o_mean, o_low, o_high, _ = _paired_ci(delta_offered)
            s_mean, s_low, s_high, _ = _paired_ci(selection_gap)
            r_mean, r_low, r_high, _ = _paired_ci(delta_served)
            e_mean, e_low, e_high, _ = _paired_ci(estimated_exposure)

            accepted_status = _negative_status(
                a_mean, a_low, a_high, DELAY_THRESHOLD
            )
            selection_status = _positive_status(
                s_mean, s_low, s_high, DELAY_THRESHOLD
            )
            served_status = _negative_status(
                r_mean, r_low, r_high, SERVED_RATE_THRESHOLD
            )

            exposure_active = bool(
                np.isfinite(e_mean)
                and e_mean >= POST_THRESHOLD_EXPOSURE_THRESHOLD
            )

            if not exposure_active:
                classification = "inactive"
                failure_component = "insufficient_post_threshold_offer_exposure"
            else:
                statuses = {
                    "accepted_delay": accepted_status,
                    "selection_gap": selection_status,
                    "served_rate": served_status,
                }
                if "reversed" in statuses.values():
                    classification = "reversed"
                elif all(value == "supported" for value in statuses.values()):
                    classification = "supported"
                else:
                    classification = "inconclusive"
                failure_component = ";".join(
                    key for key, value in statuses.items() if value != "supported"
                )

            strong_selection_evidence = bool(
                classification == "supported"
                and np.isfinite(o_mean)
                and o_mean >= -DELAY_THRESHOLD
            )

            rows.append(
                {
                    **metadata,
                    "n_paired_seeds": n,
                    "estimated_post_threshold_offer_share": e_mean,
                    "estimated_post_threshold_offer_share_ci_low": e_low,
                    "estimated_post_threshold_offer_share_ci_high": e_high,
                    "delta_class1_accepted_delay": a_mean,
                    "delta_class1_accepted_delay_ci_low": a_low,
                    "delta_class1_accepted_delay_ci_high": a_high,
                    "delta_class1_offered_delay": o_mean,
                    "delta_class1_offered_delay_ci_low": o_low,
                    "delta_class1_offered_delay_ci_high": o_high,
                    "selection_gap": s_mean,
                    "selection_gap_ci_low": s_low,
                    "selection_gap_ci_high": s_high,
                    "delta_class1_served_rate": r_mean,
                    "delta_class1_served_rate_ci_low": r_low,
                    "delta_class1_served_rate_ci_high": r_high,
                    "accepted_delay_component": accepted_status,
                    "selection_gap_component": selection_status,
                    "served_rate_component": served_status,
                    "strong_selection_evidence": strong_selection_evidence,
                    "failure_component": failure_component,
                    "classification": classification,
                }
            )

    return pd.DataFrame(rows)


def _scenario_effect_rows(
    design: pd.DataFrame,
    target_effects: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for design_row in design.to_dict(orient="records"):
        background_id = str(design_row["background_id"])
        metadata = {
            "background_id": background_id,
            "source_scenario_ids": design_row.get("source_scenario_ids", ""),
            "scenario_type": design_row.get("scenario_type", ""),
            **{column: design_row[column] for column in PARAMETER_COLUMNS},
            "valid_h5_steps": design_row["valid_h5_steps"],
        }

        if not bool(design_row["h5_design_active"]):
            rows.append(
                {
                    **metadata,
                    "n_active_primary_comparisons": 0,
                    "n_supported_primary_comparisons": 0,
                    "n_reversed_primary_comparisons": 0,
                    "failure_component": design_row[
                        "h5_design_inactive_reason"
                    ],
                    "classification": "inactive",
                }
            )
            continue

        primary = target_effects[
            (target_effects["background_id"].astype(str) == background_id)
            & (target_effects["is_primary_step"].fillna(False))
        ].copy()
        active = primary[primary["classification"] != "inactive"]
        n_supported = int((active["classification"] == "supported").sum())
        n_reversed = int((active["classification"] == "reversed").sum())

        if len(active) == 0:
            classification = "inactive"
            failure_component = "no_active_low_to_moderate_comparison"
        elif n_supported > 0 and n_reversed == 0:
            classification = "supported"
            failure_component = ""
        elif n_reversed > 0 and n_supported == 0:
            classification = "reversed"
            failure_component = "one_or_more_primary_comparisons_reversed"
        else:
            classification = "inconclusive"
            if n_supported > 0 and n_reversed > 0:
                failure_component = "mixed_supported_and_reversed_primary_results"
            else:
                failure_component = "no_primary_comparison_met_all_support_criteria"

        rows.append(
            {
                **metadata,
                "n_active_primary_comparisons": int(len(active)),
                "n_supported_primary_comparisons": n_supported,
                "n_reversed_primary_comparisons": n_reversed,
                "n_strong_selection_primary_comparisons": int(
                    active.get(
                        "strong_selection_evidence",
                        pd.Series(dtype=bool),
                    ).fillna(False).sum()
                ),
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

    target_effects = _target_effect_rows(design, raw)
    scenario_effects = _scenario_effect_rows(design, target_effects)

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    target_path = summary_dir / "h5_step_effects.csv"
    scenario_path = summary_dir / "h5_scenario_effects.csv"
    target_effects.to_csv(target_path, index=False)
    scenario_effects.to_csv(scenario_path, index=False)

    counts = (
        scenario_effects.groupby("classification", dropna=False)
        .size()
        .rename("n_scenarios")
        .reset_index()
    )
    counts.to_csv(summary_dir / "h5_classification_counts.csv", index=False)

    step_counts = (
        target_effects.groupby(
            ["balk_step_class1_focal", "classification"],
            dropna=False,
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    step_counts.to_csv(
        summary_dir / "h5_step_classification_counts.csv",
        index=False,
    )

    failures = scenario_effects[
        scenario_effects["classification"].isin(["reversed", "inconclusive"])
    ].copy()
    failures.to_csv(summary_dir / "h5_failure_candidates.csv", index=False)
    failures.to_csv(summary_dir / "h5_stage2_candidates.csv", index=False)

    _write_summary_markdown(
        scenario_effects,
        target_effects,
        counts,
        step_counts,
        summary_dir / "h5_stage1_summary.md",
    )
    print(f"Scenario effects: {scenario_path}")
    print(f"Stage 2 candidates: {summary_dir / 'h5_stage2_candidates.csv'}")
    return scenario_path


def _write_summary_markdown(
    scenario_effects: pd.DataFrame,
    target_effects: pd.DataFrame,
    counts: pd.DataFrame,
    step_counts: pd.DataFrame,
    path: Path,
) -> None:
    primary = target_effects[target_effects["is_primary_step"].fillna(False)]
    lines = [
        "# H5 Stage 1 Robustness Summary",
        "",
        f"Background scenarios classified: **{len(scenario_effects)}**",
        f"Low-to-moderate step comparisons classified: **{len(primary)}**",
        "",
        "## Scenario classification counts",
        "",
        counts.to_markdown(index=False) if not counts.empty else "No results.",
        "",
        "## Step-level classification counts",
        "",
        step_counts.to_markdown(index=False) if not step_counts.empty else "No results.",
        "",
        "## Interpretation",
        "",
        (
            "- Primary inference uses Class 1 balking-step increases of 0.10 "
            "and 0.30; the 0.50 step is retained as a diagnostic."
        ),
        (
            "- Support requires accepted delay to fall by at least 0.25 days, "
            "the offered-minus-accepted delay contrast to exceed 0.25 days, "
            "and Class 1 served rate to fall by at least 0.005."
        ),
        (
            "- A comparison is inactive when the estimated share of Class 1 "
            "offers in the post-threshold region is below 1%."
        ),
        (
            "- Supported comparisons where offered delay does not materially "
            "fall are flagged as especially strong evidence of selection rather "
            "than congestion relief."
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
    design_path = args.output_dir / "design" / "h5_background_scenarios.csv"
    raw_path = args.output_dir / "raw" / "h5_stage1_raw.csv"

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
                f"H5 design file not found: {design_path}. Run the experiment first."
            )
        classify_stage1(
            design_path=design_path,
            raw_path=raw_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
