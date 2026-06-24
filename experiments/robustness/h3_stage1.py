"""Run and classify Stage 1 robustness tests for H3.

H3: Increasing the post-threshold no-show probability reduces utilization more
when the no-show threshold is lower.

For each background, Class 1's pre-threshold no-show probability remains fixed.
The post-threshold probability is increased from the pre-threshold value to
0.70, and the contrast is evaluated across every valid threshold in
{4, 6, 9, 12}, subject to threshold < horizon_class1 - 1.

Run from the repository root with module syntax:

    py -3 -m experiments.robustness.h3_stage1 all --smoke --workers 1 --no-resume
    py -3 -m experiments.robustness.h3_stage1 all --workers 4
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
    NOSHOW_HIGH_VALUES,
    NOSHOW_THRESHOLD_VALUES,
    PARAMETER_COLUMNS,
    STAGE1_SEEDS,
)

DEFAULT_BASE_CONFIG = REPO_DIR / "configs" / "baseline.yaml"
DEFAULT_SCENARIOS = (
    REPO_DIR / "outputs" / "robustness" / "scenarios" / "all_stage1_scenarios.csv"
)
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "robustness" / "h3"

POST_HIGH_MAX = float(max(NOSHOW_HIGH_VALUES))
UTILIZATION_THRESHOLD = 0.005
THRESHOLD_GRADIENT_THRESHOLD = 0.005
REALIZED_EXPOSURE_THRESHOLD = 0.005
SPEARMAN_THRESHOLD = 0.50

FOCAL_COLUMNS = {
    "noshow_threshold_class1",
    "noshow_high_class1",
}


def _clean_string(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def valid_h3_thresholds(horizon: int) -> tuple[int, ...]:
    """Return proposed thresholds satisfying the project's indexing rule."""
    return tuple(
        int(tau)
        for tau in NOSHOW_THRESHOLD_VALUES
        if int(tau) < int(horizon) - 1
    )


def prepare_h3_backgrounds(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Remove H3 focal variables from the background and deduplicate."""
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
    out.insert(0, "background_id", [f"H3B{i:04d}" for i in range(1, len(out) + 1)])

    valid_values: list[str] = []
    active_values: list[bool] = []
    inactive_reasons: list[str] = []
    focal_low_values: list[float] = []
    focal_high_values: list[float] = []

    for row in out.to_dict(orient="records"):
        thresholds = valid_h3_thresholds(int(row["horizon_class1"]))
        focal_low = float(row["noshow_low_class1"])
        focal_high = POST_HIGH_MAX

        reasons: list[str] = []
        if len(thresholds) < 2:
            reasons.append("fewer_than_two_valid_thresholds")
        if focal_high <= focal_low:
            reasons.append("no_valid_post_threshold_increase")

        valid_values.append(";".join(str(x) for x in thresholds))
        active_values.append(not reasons)
        inactive_reasons.append(";".join(reasons))
        focal_low_values.append(focal_low)
        focal_high_values.append(focal_high)

    out["valid_h3_thresholds"] = valid_values
    out["h3_design_active"] = active_values
    out["h3_design_inactive_reason"] = inactive_reasons
    out["noshow_high_low_focal"] = focal_low_values
    out["noshow_high_high_focal"] = focal_high_values

    # Concrete placeholders keep the adapter valid. The runner overwrites both
    # focal values for every simulation.
    out["noshow_threshold_class1"] = 0
    out["noshow_high_class1"] = out["noshow_low_class1"].astype(float)
    return out


def _parse_thresholds(value: Any) -> tuple[int, ...]:
    text = _clean_string(value).strip()
    if not text:
        return ()
    return tuple(int(float(x)) for x in text.split(";") if x != "")


def _task_payloads(
    backgrounds: pd.DataFrame,
    seeds: Sequence[int],
    completed: set[tuple[str, int, str, int]],
    base_config_path: str | Path,
) -> Iterable[dict[str, Any]]:
    for row in backgrounds.to_dict(orient="records"):
        if not bool(row["h3_design_active"]):
            continue
        background_id = str(row["background_id"])
        thresholds = _parse_thresholds(row["valid_h3_thresholds"])
        focal_low = float(row["noshow_high_low_focal"])
        focal_high = float(row["noshow_high_high_focal"])

        for threshold in thresholds:
            for arm, post_probability in (
                ("low", focal_low),
                ("high", focal_high),
            ):
                for seed in seeds:
                    key = (background_id, int(threshold), arm, int(seed))
                    if key in completed:
                        continue
                    yield {
                        "row": row,
                        "background_id": background_id,
                        "threshold": int(threshold),
                        "arm": arm,
                        "post_probability": float(post_probability),
                        "seed": int(seed),
                        "base_config_path": str(base_config_path),
                    }


def _run_task(task: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.robustness.simulation_adapter import run_scenario

    row = dict(task["row"])
    threshold = int(task["threshold"])
    arm = str(task["arm"])
    post_probability = float(task["post_probability"])
    seed = int(task["seed"])

    metrics = run_scenario(
        row,
        seed=seed,
        base_config_path=task["base_config_path"],
        overrides={
            "noshow_threshold_class1": threshold,
            "noshow_high_class1": post_probability,
        },
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
        "noshow_low_class1_background": float(row["noshow_low_class1"]),
        "noshow_threshold_class1_focal": threshold,
        "noshow_high_class1_focal": post_probability,
        "arm": arm,
        "seed": seed,
        **metrics,
    }


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _select_backgrounds(
    scenarios_path: Path,
    *,
    smoke: bool,
    max_scenarios: int | None,
) -> pd.DataFrame:
    backgrounds = prepare_h3_backgrounds(pd.read_csv(scenarios_path))
    if smoke:
        active = backgrounds[backgrounds["h3_design_active"]].head(2)
        inactive = backgrounds[~backgrounds["h3_design_active"]].head(1)
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

    design_path = output_dir / "design" / "h3_background_scenarios.csv"
    raw_path = output_dir / "raw" / "h3_stage1_raw.csv"
    design_path.parent.mkdir(parents=True, exist_ok=True)
    backgrounds.to_csv(design_path, index=False)

    completed: set[tuple[str, int, str, int]] = set()
    if resume and raw_path.exists():
        old = pd.read_csv(
            raw_path,
            usecols=[
                "background_id",
                "noshow_threshold_class1_focal",
                "arm",
                "seed",
            ],
        )
        completed = {
            (
                str(r.background_id),
                int(r.noshow_threshold_class1_focal),
                str(r.arm),
                int(r.seed),
            )
            for r in old.itertuples(index=False)
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
        len(_parse_thresholds(row.valid_h3_thresholds)) * 2 * len(seeds)
        for row in backgrounds.itertuples(index=False)
        if bool(row.h3_design_active)
    )

    print(f"H3 backgrounds in this run: {len(backgrounds)}")
    print(f"Design-active backgrounds: {int(backgrounds['h3_design_active'].sum())}")
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


def _paired_ci(
    values: pd.Series,
    confidence: float = 0.95,
) -> tuple[float, float, float, int]:
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
        raise ValueError(expected)
    return "inconclusive"


def _threshold_effect_rows(
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
            "noshow_high_low_focal": float(
                design_row["noshow_high_low_focal"]
            ),
            "noshow_high_high_focal": float(
                design_row["noshow_high_high_focal"]
            ),
        }

        if not bool(design_row["h3_design_active"]):
            continue

        group = raw[raw["background_id"].astype(str) == background_id]
        for threshold in _parse_thresholds(
            design_row["valid_h3_thresholds"]
        ):
            low = group[
                (group["noshow_threshold_class1_focal"] == threshold)
                & (group["arm"] == "low")
            ].set_index("seed")
            high = group[
                (group["noshow_threshold_class1_focal"] == threshold)
                & (group["arm"] == "high")
            ].set_index("seed")
            common = low.index.intersection(high.index)

            if len(common) == 0:
                rows.append(
                    {
                        **metadata,
                        "noshow_threshold_class1_focal": threshold,
                        "n_paired_seeds": 0,
                        "direction_component": "inconclusive",
                    }
                )
                continue

            low = low.loc[common]
            high = high.loc[common]
            delta_u = high["average_utilization"] - low["average_utilization"]
            delta_noshow = (
                high["class_1_no_show_rate_per_arrival"]
                - low["class_1_no_show_rate_per_arrival"]
            )

            u_mean, u_low, u_high, n = _paired_ci(delta_u)
            ns_mean, ns_low, ns_high, _ = _paired_ci(delta_noshow)
            direction = _component_status(
                u_mean,
                u_low,
                u_high,
                expected="negative",
                practical_threshold=UTILIZATION_THRESHOLD,
            )

            rows.append(
                {
                    **metadata,
                    "noshow_threshold_class1_focal": threshold,
                    "n_paired_seeds": n,
                    "delta_average_utilization": u_mean,
                    "delta_average_utilization_ci_low": u_low,
                    "delta_average_utilization_ci_high": u_high,
                    "utilization_loss_magnitude": -u_mean,
                    "delta_realized_class1_noshow_rate": ns_mean,
                    "delta_realized_class1_noshow_rate_ci_low": ns_low,
                    "delta_realized_class1_noshow_rate_ci_high": ns_high,
                    "realized_exposure_active": bool(
                        np.isfinite(ns_mean)
                        and ns_mean >= REALIZED_EXPOSURE_THRESHOLD
                    ),
                    "direction_component": direction,
                }
            )

    return pd.DataFrame(rows)


def _scenario_effect_rows(
    design: pd.DataFrame,
    raw: pd.DataFrame,
    threshold_effects: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for design_row in design.to_dict(orient="records"):
        background_id = str(design_row["background_id"])
        metadata = {
            "background_id": background_id,
            "source_scenario_ids": design_row.get("source_scenario_ids", ""),
            "scenario_type": design_row.get("scenario_type", ""),
            **{column: design_row[column] for column in PARAMETER_COLUMNS},
            "valid_h3_thresholds": design_row["valid_h3_thresholds"],
            "noshow_high_low_focal": float(
                design_row["noshow_high_low_focal"]
            ),
            "noshow_high_high_focal": float(
                design_row["noshow_high_high_focal"]
            ),
        }

        if not bool(design_row["h3_design_active"]):
            rows.append(
                {
                    **metadata,
                    "n_exposed_thresholds": 0,
                    "failure_component": design_row[
                        "h3_design_inactive_reason"
                    ],
                    "classification": "inactive",
                }
            )
            continue

        effects = threshold_effects[
            threshold_effects["background_id"].astype(str) == background_id
        ].copy()
        exposed = effects[
            effects["realized_exposure_active"].fillna(False)
        ].sort_values("noshow_threshold_class1_focal")

        if len(exposed) < 2:
            rows.append(
                {
                    **metadata,
                    "n_exposed_thresholds": int(len(exposed)),
                    "failure_component": "insufficient_realized_post_threshold_exposure",
                    "classification": "inactive",
                }
            )
            continue

        low_threshold = int(exposed["noshow_threshold_class1_focal"].min())
        high_threshold = int(exposed["noshow_threshold_class1_focal"].max())
        low_effect = exposed[
            exposed["noshow_threshold_class1_focal"] == low_threshold
        ].iloc[0]
        high_effect = exposed[
            exposed["noshow_threshold_class1_focal"] == high_threshold
        ].iloc[0]

        group = raw[raw["background_id"].astype(str) == background_id]
        delta_by_seed: dict[int, pd.Series] = {}
        for threshold in exposed["noshow_threshold_class1_focal"].astype(int):
            low = group[
                (group["noshow_threshold_class1_focal"] == threshold)
                & (group["arm"] == "low")
            ].set_index("seed")
            high = group[
                (group["noshow_threshold_class1_focal"] == threshold)
                & (group["arm"] == "high")
            ].set_index("seed")
            common = low.index.intersection(high.index)
            delta_by_seed[threshold] = (
                high.loc[common, "average_utilization"]
                - low.loc[common, "average_utilization"]
            )

        common_gradient = delta_by_seed[low_threshold].index.intersection(
            delta_by_seed[high_threshold].index
        )
        # Positive values mean the utilization loss is greater at the lower
        # threshold: delta_U(high threshold) - delta_U(low threshold) > 0.
        gradient = (
            delta_by_seed[high_threshold].loc[common_gradient]
            - delta_by_seed[low_threshold].loc[common_gradient]
        )
        g_mean, g_low, g_high, n_gradient = _paired_ci(gradient)
        gradient_status = _component_status(
            g_mean,
            g_low,
            g_high,
            expected="positive",
            practical_threshold=THRESHOLD_GRADIENT_THRESHOLD,
        )

        direction_status = str(low_effect["direction_component"])
        if len(exposed) >= 2:
            correlation = stats.spearmanr(
                exposed["noshow_threshold_class1_focal"].astype(float),
                exposed["utilization_loss_magnitude"].astype(float),
            ).statistic
            spearman = float(correlation) if np.isfinite(correlation) else math.nan
        else:
            spearman = math.nan

        components = {
            "utilization_direction": direction_status,
            "threshold_gradient": gradient_status,
        }
        if "reversed" in components.values():
            classification = "reversed"
        elif (
            direction_status == "supported"
            and gradient_status == "supported"
            and np.isfinite(spearman)
            and spearman <= -SPEARMAN_THRESHOLD
        ):
            classification = "supported"
        else:
            classification = "inconclusive"

        failure_components = [
            name for name, status in components.items() if status != "supported"
        ]
        if not (
            np.isfinite(spearman)
            and spearman <= -SPEARMAN_THRESHOLD
        ):
            failure_components.append("spearman_pattern")

        rows.append(
            {
                **metadata,
                "n_exposed_thresholds": int(len(exposed)),
                "lowest_exposed_threshold": low_threshold,
                "highest_exposed_threshold": high_threshold,
                "n_gradient_paired_seeds": n_gradient,
                "lowest_threshold_delta_utilization": float(
                    low_effect["delta_average_utilization"]
                ),
                "highest_threshold_delta_utilization": float(
                    high_effect["delta_average_utilization"]
                ),
                "low_minus_high_utilization_loss": g_mean,
                "low_minus_high_utilization_loss_ci_low": g_low,
                "low_minus_high_utilization_loss_ci_high": g_high,
                "spearman_threshold_vs_loss_magnitude": spearman,
                "utilization_direction_component": direction_status,
                "threshold_gradient_component": gradient_status,
                "failure_component": ";".join(sorted(set(failure_components))),
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
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
    else:
        raw = pd.DataFrame()

    threshold_effects = _threshold_effect_rows(design, raw)
    scenario_effects = _scenario_effect_rows(
        design,
        raw,
        threshold_effects,
    )

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = summary_dir / "h3_threshold_effects.csv"
    scenario_path = summary_dir / "h3_scenario_effects.csv"
    threshold_effects.to_csv(threshold_path, index=False)
    scenario_effects.to_csv(scenario_path, index=False)

    counts = (
        scenario_effects.groupby("classification", dropna=False)
        .size()
        .rename("n_scenarios")
        .reset_index()
    )
    counts.to_csv(summary_dir / "h3_classification_counts.csv", index=False)

    failures = scenario_effects[
        scenario_effects["classification"].isin(
            ["reversed", "inconclusive"]
        )
    ].copy()
    failures.to_csv(summary_dir / "h3_failure_candidates.csv", index=False)
    failures.to_csv(summary_dir / "h3_stage2_candidates.csv", index=False)

    _write_summary_markdown(
        scenario_effects,
        threshold_effects,
        summary_dir / "h3_stage1_summary.md",
    )
    print(f"Scenario effects: {scenario_path}")
    print(f"Stage 2 candidates: {summary_dir / 'h3_stage2_candidates.csv'}")
    return scenario_path


def _write_summary_markdown(
    scenario_effects: pd.DataFrame,
    threshold_effects: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# H3 Stage 1 Robustness Summary",
        "",
        f"Background scenarios classified: **{len(scenario_effects)}**",
        "",
        "## Scenario classification counts",
        "",
    ]

    if scenario_effects.empty:
        lines.append("No complete scenarios were available for classification.")
    else:
        counts = (
            scenario_effects.groupby("classification")
            .size()
            .rename("n_scenarios")
            .reset_index()
        )
        lines.append(counts.to_markdown(index=False))
        lines.extend(
            [
                "",
                "## Exposure diagnostics",
                "",
                (
                    "- Design-active threshold comparisons with realized "
                    f"Class 1 no-show exposure of at least "
                    f"{REALIZED_EXPOSURE_THRESHOLD:.3f}: "
                    f"**{int(threshold_effects.get('realized_exposure_active', pd.Series(dtype=bool)).fillna(False).sum())}"
                    f"/{len(threshold_effects)}**."
                ),
                "",
                "## Interpretation",
                "",
                (
                    "- Support requires a material utilization reduction when "
                    "the post-threshold no-show probability increases."
                ),
                (
                    "- The utilization loss must be at least 0.005 greater at "
                    "the lowest exposed threshold than at the highest exposed "
                    "threshold, with a paired confidence interval above zero."
                ),
                (
                    "- The mean threshold pattern must also be decreasing, with "
                    "Spearman correlation no greater than -0.50."
                ),
                (
                    "- Reversed and inconclusive active scenarios are exported "
                    "for Stage 2 confirmation."
                ),
            ]
        )

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
    design_path = args.output_dir / "design" / "h3_background_scenarios.csv"
    raw_path = args.output_dir / "raw" / "h3_stage1_raw.csv"

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
                f"H3 design file not found: {design_path}. Run the experiment first."
            )
        classify_stage1(
            design_path=design_path,
            raw_path=raw_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
