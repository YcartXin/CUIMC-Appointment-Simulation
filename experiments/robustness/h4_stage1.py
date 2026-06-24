"""Run and classify Stage 1 robustness tests for H4.

H4: Under heavy oversubscription, the effect of a common balking step on mean
offered delay is non-monotone: moderate post-threshold balking increases
offered delay before sufficiently high balking reduces it.

The focal experiment applies the same post-threshold balking probability to
both classes while setting both pre-threshold probabilities to zero. It tests
the five levels 0.0, 0.1, 0.3, 0.5, and 0.7 with paired Stage 1 seeds.

Run from the repository root:

    py -3 -m experiments.robustness.h4_stage1 all --smoke --workers 1 --no-resume
    py -3 -m experiments.robustness.h4_stage1 all --workers 4
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
    BALK_HIGH_VALUES,
    PARAMETER_COLUMNS,
    STAGE1_SEEDS,
)

DEFAULT_BASE_CONFIG = REPO_DIR / "configs" / "baseline.yaml"
DEFAULT_SCENARIOS = (
    REPO_DIR / "outputs" / "robustness" / "scenarios" / "all_stage1_scenarios.csv"
)
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "robustness" / "h4"

FOCAL_LEVELS = tuple(float(x) for x in BALK_HIGH_VALUES)
COMMON_PRE_THRESHOLD_BALK = 0.0

DELAY_THRESHOLD = 0.25
EXPOSURE_THRESHOLD = 0.005
MONOTONE_SPEARMAN_THRESHOLD = 0.80

FOCAL_COLUMNS = {
    "balk_low_class1",
    "balk_low_class2",
    "balk_high_class1",
    "balk_high_class2",
}


def demand_regime(rho: float) -> str:
    if rho >= 3.1:
        return "high"
    if rho >= 2.0:
        return "boundary"
    return "low"


def prepare_h4_backgrounds(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Remove common balk-rate focal variables and deduplicate backgrounds."""
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
    out.insert(0, "background_id", [f"H4B{i:04d}" for i in range(1, len(out) + 1)])
    out["demand_regime"] = out["rho"].astype(float).map(demand_regime)
    out["h4_scope_active"] = out["demand_regime"].eq("high")
    out["h4_scope_inactive_reason"] = np.where(
        out["h4_scope_active"],
        "",
        "outside_heavy_oversubscription_scope",
    )

    # Placeholders used by the adapter. The runner overwrites the post-threshold
    # probability at every focal level.
    out["balk_low_class1"] = COMMON_PRE_THRESHOLD_BALK
    out["balk_low_class2"] = COMMON_PRE_THRESHOLD_BALK
    out["balk_high_class1"] = COMMON_PRE_THRESHOLD_BALK
    out["balk_high_class2"] = COMMON_PRE_THRESHOLD_BALK
    return out


def _task_payloads(
    backgrounds: pd.DataFrame,
    seeds: Sequence[int],
    completed: set[tuple[str, float, int]],
    base_config_path: str | Path,
) -> Iterable[dict[str, Any]]:
    for row in backgrounds.to_dict(orient="records"):
        background_id = str(row["background_id"])
        for level in FOCAL_LEVELS:
            for seed in seeds:
                key = (background_id, float(level), int(seed))
                if key in completed:
                    continue
                yield {
                    "row": row,
                    "background_id": background_id,
                    "level": float(level),
                    "seed": int(seed),
                    "base_config_path": str(base_config_path),
                }


def _run_task(task: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.robustness.simulation_adapter import run_scenario

    row = dict(task["row"])
    level = float(task["level"])
    seed = int(task["seed"])

    metrics = run_scenario(
        row,
        seed=seed,
        base_config_path=task["base_config_path"],
        overrides={
            "balk_low_class1": COMMON_PRE_THRESHOLD_BALK,
            "balk_low_class2": COMMON_PRE_THRESHOLD_BALK,
            "balk_high_class1": level,
            "balk_high_class2": level,
        },
    )

    total_balked = (
        float(metrics["class_1_balked"]) + float(metrics["class_2_balked"])
    )
    total_arrivals = float(metrics["total_arrivals"])
    overall_balk_rate = (
        total_balked / total_arrivals if total_arrivals > 0 else math.nan
    )

    return {
        "background_id": task["background_id"],
        "source_scenario_ids": str(row.get("source_scenario_ids", "")),
        "scenario_type": row["scenario_type"],
        "demand_regime": row["demand_regime"],
        "rho": float(row["rho"]),
        "class1_share": float(row["class1_share"]),
        "slots_per_day": int(row["slots_per_day"]),
        "horizon_class1": int(row["horizon_class1"]),
        "horizon_class2": int(row["horizon_class2"]),
        "common_balk_low_focal": COMMON_PRE_THRESHOLD_BALK,
        "common_balk_high_focal": level,
        "seed": seed,
        "overall_balk_rate_per_arrival": overall_balk_rate,
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
    backgrounds = prepare_h4_backgrounds(pd.read_csv(scenarios_path))
    if smoke:
        high = backgrounds[backgrounds["h4_scope_active"]].head(2)
        diagnostic = backgrounds[~backgrounds["h4_scope_active"]].head(1)
        backgrounds = pd.concat([high, diagnostic], ignore_index=True)
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

    design_path = output_dir / "design" / "h4_background_scenarios.csv"
    raw_path = output_dir / "raw" / "h4_stage1_raw.csv"
    design_path.parent.mkdir(parents=True, exist_ok=True)
    backgrounds.to_csv(design_path, index=False)

    completed: set[tuple[str, float, int]] = set()
    if resume and raw_path.exists():
        old = pd.read_csv(
            raw_path,
            usecols=["background_id", "common_balk_high_focal", "seed"],
        )
        completed = {
            (
                str(r.background_id),
                float(r.common_balk_high_focal),
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
    expected = len(backgrounds) * len(FOCAL_LEVELS) * len(seeds)

    print(f"H4 backgrounds in this run: {len(backgrounds)}")
    print(
        "Heavy-oversubscription backgrounds: "
        f"{int(backgrounds['h4_scope_active'].sum())}"
    )
    print(f"Focal levels: {FOCAL_LEVELS}")
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


def _paired_difference(
    pivot: pd.DataFrame,
    higher_level: float,
    lower_level: float,
) -> tuple[float, float, float, int]:
    common = pivot[higher_level].dropna().index.intersection(
        pivot[lower_level].dropna().index
    )
    return _paired_ci(
        pivot.loc[common, higher_level] - pivot.loc[common, lower_level]
    )


def _classify_curve(group: pd.DataFrame) -> dict[str, Any]:
    delay_pivot = group.pivot_table(
        index="seed",
        columns="common_balk_high_focal",
        values="mean_offered_booking_delay",
        aggfunc="first",
    ).reindex(columns=FOCAL_LEVELS)

    balk_pivot = group.pivot_table(
        index="seed",
        columns="common_balk_high_focal",
        values="overall_balk_rate_per_arrival",
        aggfunc="first",
    ).reindex(columns=FOCAL_LEVELS)

    complete_seeds = delay_pivot.dropna().index
    means = delay_pivot.loc[complete_seeds].mean(axis=0)
    n_complete = int(len(complete_seeds))

    result: dict[str, Any] = {
        "n_complete_seeds": n_complete,
        "curve_shape": "incomplete",
        "peak_level": math.nan,
        "trough_level": math.nan,
        "spearman_level_vs_delay": math.nan,
        "delay_range": math.nan,
        "max_exposure_increase": math.nan,
    }
    for level in FOCAL_LEVELS:
        result[f"mean_delay_level_{str(level).replace('.', '_')}"] = (
            float(means[level]) if level in means and np.isfinite(means[level]) else math.nan
        )

    if n_complete == 0 or means.isna().any():
        return result

    max_level = float(means.idxmax())
    min_level = float(means.idxmin())
    max_value = float(means.max())
    min_value = float(means.min())
    delay_range = max_value - min_value
    spearman = float(
        stats.spearmanr(
            np.asarray(FOCAL_LEVELS, dtype=float),
            means.to_numpy(dtype=float),
        ).statistic
    )
    result.update(
        {
            "peak_level": max_level,
            "trough_level": min_level,
            "spearman_level_vs_delay": spearman,
            "delay_range": delay_range,
        }
    )

    if 0.7 in balk_pivot and 0.0 in balk_pivot:
        common = balk_pivot[0.7].dropna().index.intersection(
            balk_pivot[0.0].dropna().index
        )
        exposure = balk_pivot.loc[common, 0.7] - balk_pivot.loc[common, 0.0]
        e_mean, e_low, e_high, e_n = _paired_ci(exposure)
        result.update(
            {
                "max_exposure_increase": e_mean,
                "max_exposure_increase_ci_low": e_low,
                "max_exposure_increase_ci_high": e_high,
                "n_exposure_paired_seeds": e_n,
            }
        )

    interior_levels = FOCAL_LEVELS[1:-1]

    if max_level in interior_levels:
        rise_mean, rise_low, rise_high, rise_n = _paired_difference(
            delay_pivot, max_level, FOCAL_LEVELS[0]
        )
        fall_mean, fall_low, fall_high, fall_n = _paired_difference(
            delay_pivot, max_level, FOCAL_LEVELS[-1]
        )
        result.update(
            {
                "hump_rise_from_low": rise_mean,
                "hump_rise_from_low_ci_low": rise_low,
                "hump_rise_from_low_ci_high": rise_high,
                "hump_fall_to_high": fall_mean,
                "hump_fall_to_high_ci_low": fall_low,
                "hump_fall_to_high_ci_high": fall_high,
                "hump_n_paired_seeds": min(rise_n, fall_n),
            }
        )
        if (
            rise_mean >= DELAY_THRESHOLD
            and rise_low > 0
            and fall_mean >= DELAY_THRESHOLD
            and fall_low > 0
        ):
            result["curve_shape"] = "hump"
            return result

    if min_level in interior_levels:
        drop_mean, drop_low, drop_high, drop_n = _paired_difference(
            delay_pivot, FOCAL_LEVELS[0], min_level
        )
        rebound_mean, rebound_low, rebound_high, rebound_n = _paired_difference(
            delay_pivot, FOCAL_LEVELS[-1], min_level
        )
        result.update(
            {
                "u_drop_from_low": drop_mean,
                "u_drop_from_low_ci_low": drop_low,
                "u_drop_from_low_ci_high": drop_high,
                "u_rebound_to_high": rebound_mean,
                "u_rebound_to_high_ci_low": rebound_low,
                "u_rebound_to_high_ci_high": rebound_high,
                "u_n_paired_seeds": min(drop_n, rebound_n),
            }
        )
        if (
            drop_mean >= DELAY_THRESHOLD
            and drop_low > 0
            and rebound_mean >= DELAY_THRESHOLD
            and rebound_low > 0
        ):
            result["curve_shape"] = "u_shaped"
            return result

    end_mean, end_low, end_high, end_n = _paired_difference(
        delay_pivot, FOCAL_LEVELS[-1], FOCAL_LEVELS[0]
    )
    result.update(
        {
            "high_minus_low_delay": end_mean,
            "high_minus_low_delay_ci_low": end_low,
            "high_minus_low_delay_ci_high": end_high,
            "endpoint_n_paired_seeds": end_n,
        }
    )

    if delay_range < DELAY_THRESHOLD:
        result["curve_shape"] = "flat"
    elif (
        spearman <= -MONOTONE_SPEARMAN_THRESHOLD
        and end_mean <= -DELAY_THRESHOLD
        and end_high < 0
    ):
        result["curve_shape"] = "decreasing"
    elif (
        spearman >= MONOTONE_SPEARMAN_THRESHOLD
        and end_mean >= DELAY_THRESHOLD
        and end_low > 0
    ):
        result["curve_shape"] = "increasing"
    else:
        result["curve_shape"] = "irregular"

    return result


def _classification_for_shape(
    *,
    regime: str,
    shape: str,
    exposure_active: bool,
) -> tuple[str, str]:
    if not exposure_active:
        return "inactive", "insufficient_realized_balking_exposure"

    # H4's inferential claim is specifically about heavy oversubscription.
    # Lower-demand curves are retained as diagnostics but are outside scope.
    if regime != "high":
        return "inactive", "outside_heavy_oversubscription_scope"

    if shape == "hump":
        return "supported", ""
    if shape == "u_shaped":
        return "reversed", "opposite_nonmonotone_pattern"
    if shape == "incomplete":
        return "inconclusive", "incomplete_focal_curve"
    return "inconclusive", f"active_but_{shape}_curve"


def classify_stage1(
    *,
    design_path: Path,
    raw_path: Path,
    output_dir: Path,
) -> Path:
    design = pd.read_csv(design_path)
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for design_row in design.to_dict(orient="records"):
        background_id = str(design_row["background_id"])
        group = raw[raw["background_id"].astype(str) == background_id].copy()
        curve = _classify_curve(group)
        exposure_active = bool(
            np.isfinite(curve.get("max_exposure_increase", math.nan))
            and curve["max_exposure_increase"] >= EXPOSURE_THRESHOLD
        )
        classification, failure_component = _classification_for_shape(
            regime=str(design_row["demand_regime"]),
            shape=str(curve["curve_shape"]),
            exposure_active=exposure_active,
        )

        rows.append(
            {
                "background_id": background_id,
                "source_scenario_ids": design_row.get("source_scenario_ids", ""),
                "scenario_type": design_row.get("scenario_type", ""),
                "demand_regime": design_row["demand_regime"],
                **{column: design_row[column] for column in PARAMETER_COLUMNS},
                "exposure_active": exposure_active,
                **curve,
                "failure_component": failure_component,
                "classification": classification,
            }
        )

    effects = pd.DataFrame(rows)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    effects_path = summary_dir / "h4_scenario_effects.csv"
    effects.to_csv(effects_path, index=False)

    counts = (
        effects.groupby(["demand_regime", "classification"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    counts.to_csv(summary_dir / "h4_classification_counts.csv", index=False)

    shapes = (
        effects.groupby(["demand_regime", "curve_shape"], dropna=False)
        .size()
        .rename("n_scenarios")
        .reset_index()
    )
    shapes.to_csv(summary_dir / "h4_curve_shape_counts.csv", index=False)

    failures = effects[
        effects["classification"].isin(["reversed", "inconclusive"])
    ].copy()
    failures.to_csv(summary_dir / "h4_failure_candidates.csv", index=False)
    failures.to_csv(summary_dir / "h4_stage2_candidates.csv", index=False)

    _write_summary_markdown(
        effects,
        counts,
        shapes,
        summary_dir / "h4_stage1_summary.md",
    )
    print(f"Scenario effects: {effects_path}")
    print(f"Stage 2 candidates: {summary_dir / 'h4_stage2_candidates.csv'}")
    return effects_path


def _write_summary_markdown(
    effects: pd.DataFrame,
    counts: pd.DataFrame,
    shapes: pd.DataFrame,
    path: Path,
) -> None:
    high = effects[effects["demand_regime"] == "high"]
    lines = [
        "# H4 Stage 1 Robustness Summary",
        "",
        f"Background scenarios classified: **{len(effects)}**",
        f"Heavy-oversubscription backgrounds classified: **{len(high)}**",
        "",
        "## Classification counts by demand regime",
        "",
        counts.to_markdown(index=False) if not counts.empty else "No results.",
        "",
        "## Curve-shape counts by demand regime",
        "",
        shapes.to_markdown(index=False) if not shapes.empty else "No results.",
        "",
        "## Interpretation",
        "",
        (
            "- H4 is inferentially evaluated only in heavy-oversubscription "
            "backgrounds; lower-demand curves are retained as diagnostics."
        ),
        (
            "- Support requires a statistically reliable interior maximum in "
            "mean offered delay, at least 0.25 days above both endpoint levels."
        ),
        (
            "- A statistically reliable interior minimum is classified as a "
            "reversal because it is the opposite non-monotone pattern."
        ),
        (
            "- Active high-demand flat, monotone, and irregular curves are "
            "inconclusive and exported for Stage 2 confirmation."
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
    design_path = args.output_dir / "design" / "h4_background_scenarios.csv"
    raw_path = args.output_dir / "raw" / "h4_stage1_raw.csv"

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
                f"H4 design file not found: {design_path}. Run the experiment first."
            )
        classify_stage1(
            design_path=design_path,
            raw_path=raw_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
