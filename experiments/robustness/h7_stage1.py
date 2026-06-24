"""Run and classify Stage 1 robustness tests for H7.

H7: Holding Class 1's balking rates fixed, an equal between-class balking-rate
difference produces a larger served-rate gap when the difference is placed
below the threshold than when it is placed above the threshold.

For a gap magnitude g and Class 1 rates (b0, b1), the two Class 2 arms are:

    pre-threshold gap:  (b0 + g, b1)
    post-threshold gap: (b0, b1 - g)

The Class 2 within-class step is therefore b1 - b0 - g in both arms. This
isolates the location of the between-class difference.

Run from the repository root:

    py -3 -m experiments.robustness.h7_stage1 all --smoke --workers 1 --no-resume
    py -3 -m experiments.robustness.h7_stage1 all --workers 4
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
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "robustness" / "h7"

GAP_LEVELS = (0.05, 0.10, 0.20, 0.30, 0.50)
EFFECT_THRESHOLD = 0.0025
EXPOSURE_THRESHOLD = 0.01
MIN_ACTIVE_GAPS = 2

FOCAL_COLUMNS = {
    "balk_low_class2",
    "balk_high_class2",
}


def _gap_key(value: float) -> str:
    return f"{float(value):.2f}"


def valid_h7_gaps(balk_low_class1: float, balk_high_class1: float) -> tuple[float, ...]:
    """Return gap magnitudes that preserve valid Class 2 probabilities."""
    step = float(balk_high_class1) - float(balk_low_class1)
    if step <= 0:
        return ()
    return tuple(
        float(gap)
        for gap in GAP_LEVELS
        if float(gap) <= step + 1e-12
    )


def prepare_h7_backgrounds(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Remove the Class 2 focal balking rates and deduplicate backgrounds."""
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
    signature_columns = [column for column in PARAMETER_COLUMNS if column not in FOCAL_COLUMNS]

    rows: list[dict[str, Any]] = []
    for _, group in df.groupby(signature_columns, dropna=False, sort=False):
        first = group.iloc[0].to_dict()
        first["source_scenario_ids"] = ";".join(
            group["source_scenario_id"].astype(str)
        )
        first["source_scenario_count"] = int(len(group))
        rows.append(first)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out.insert(0, "background_id", [f"H7B{i:04d}" for i in range(1, len(out) + 1)])

    valid_strings: list[str] = []
    active_values: list[bool] = []
    inactive_reasons: list[str] = []

    for row in out.to_dict(orient="records"):
        gaps = valid_h7_gaps(
            float(row["balk_low_class1"]),
            float(row["balk_high_class1"]),
        )
        reasons: list[str] = []
        if len(gaps) < MIN_ACTIVE_GAPS:
            reasons.append("fewer_than_two_valid_gap_magnitudes")
        valid_strings.append(";".join(_gap_key(gap) for gap in gaps))
        active_values.append(not reasons)
        inactive_reasons.append(";".join(reasons))

    out["valid_h7_gaps"] = valid_strings
    out["h7_design_active"] = active_values
    out["h7_design_inactive_reason"] = inactive_reasons

    # Neutral placeholders. The runner overwrites these for each arm.
    out["balk_low_class2"] = out["balk_low_class1"].astype(float)
    out["balk_high_class2"] = out["balk_high_class1"].astype(float)
    return out


def _parse_gaps(value: Any) -> tuple[float, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    return tuple(float(item) for item in text.split(";") if item != "")


def _task_payloads(
    backgrounds: pd.DataFrame,
    seeds: Sequence[int],
    completed: set[tuple[str, float, str, int]],
    base_config_path: str | Path,
) -> Iterable[dict[str, Any]]:
    for row in backgrounds.to_dict(orient="records"):
        if not bool(row["h7_design_active"]):
            continue

        background_id = str(row["background_id"])
        b0 = float(row["balk_low_class1"])
        b1 = float(row["balk_high_class1"])

        for gap in _parse_gaps(row["valid_h7_gaps"]):
            arms = {
                "pre": {
                    "balk_low_class2": b0 + gap,
                    "balk_high_class2": b1,
                },
                "post": {
                    "balk_low_class2": b0,
                    "balk_high_class2": b1 - gap,
                },
            }
            for arm, overrides in arms.items():
                for seed in seeds:
                    key = (background_id, float(gap), arm, int(seed))
                    if key in completed:
                        continue
                    yield {
                        "row": row,
                        "background_id": background_id,
                        "gap": float(gap),
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
    gap = float(task["gap"])
    arm = str(task["arm"])
    seed = int(task["seed"])

    metrics = run_scenario_with_offered_delay_counts(
        row,
        seed=seed,
        base_config_path=task["base_config_path"],
        overrides=task["overrides"],
    )

    class2_low_share, class2_high_share = _delay_regime_shares(
        metrics["class_2_offered_delay_counts_json"],
        threshold=int(row["balk_threshold_class2"]),
        total_offered=float(metrics["class_2_offered"]),
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
        "gap_magnitude_focal": gap,
        "gap_location_arm": arm,
        "class2_balk_low_focal": float(task["overrides"]["balk_low_class2"]),
        "class2_balk_high_focal": float(task["overrides"]["balk_high_class2"]),
        "class2_low_regime_offer_share": class2_low_share,
        "class2_high_regime_offer_share": class2_high_share,
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
    backgrounds = prepare_h7_backgrounds(pd.read_csv(scenarios_path))
    if smoke:
        active = backgrounds[backgrounds["h7_design_active"]].head(2)
        inactive = backgrounds[~backgrounds["h7_design_active"]].head(1)
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

    design_path = output_dir / "design" / "h7_background_scenarios.csv"
    raw_path = output_dir / "raw" / "h7_stage1_raw.csv"
    design_path.parent.mkdir(parents=True, exist_ok=True)
    backgrounds.to_csv(design_path, index=False)

    completed: set[tuple[str, float, str, int]] = set()
    if resume and raw_path.exists():
        old = pd.read_csv(
            raw_path,
            usecols=[
                "background_id",
                "gap_magnitude_focal",
                "gap_location_arm",
                "seed",
            ],
        )
        completed = {
            (
                str(row.background_id),
                float(row.gap_magnitude_focal),
                str(row.gap_location_arm),
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
        len(_parse_gaps(row.valid_h7_gaps)) * 2 * len(seeds)
        for row in backgrounds.itertuples(index=False)
        if bool(row.h7_design_active)
    )

    print(f"H7 backgrounds in this run: {len(backgrounds)}")
    print(f"Design-active backgrounds: {int(backgrounds['h7_design_active'].sum())}")
    print(f"Gap levels: {GAP_LEVELS}")
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


def _effect_status(
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


def _gap_effect_rows(
    design: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for design_row in design.to_dict(orient="records"):
        if not bool(design_row["h7_design_active"]):
            continue

        background_id = str(design_row["background_id"])
        group = raw[raw["background_id"].astype(str) == background_id]

        for gap in _parse_gaps(design_row["valid_h7_gaps"]):
            pre = group[
                np.isclose(group["gap_magnitude_focal"].astype(float), gap)
                & group["gap_location_arm"].eq("pre")
            ].set_index("seed")
            post = group[
                np.isclose(group["gap_magnitude_focal"].astype(float), gap)
                & group["gap_location_arm"].eq("post")
            ].set_index("seed")
            common = pre.index.intersection(post.index)

            metadata = {
                "background_id": background_id,
                "source_scenario_ids": design_row.get("source_scenario_ids", ""),
                "scenario_type": design_row.get("scenario_type", ""),
                **{column: design_row[column] for column in PARAMETER_COLUMNS},
                "gap_magnitude_focal": float(gap),
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

            pre = pre.loc[common]
            post = post.loc[common]
            d7 = pre["absolute_served_rate_gap"] - post["absolute_served_rate_gap"]

            d_mean, d_low, d_high, n = _paired_ci(d7)
            low_exposure = pd.concat(
                [
                    pre["class2_low_regime_offer_share"],
                    post["class2_low_regime_offer_share"],
                ]
            )
            high_exposure = pd.concat(
                [
                    pre["class2_high_regime_offer_share"],
                    post["class2_high_regime_offer_share"],
                ]
            )
            low_mean = float(pd.to_numeric(low_exposure, errors="coerce").mean())
            high_mean = float(pd.to_numeric(high_exposure, errors="coerce").mean())
            exposure_active = bool(
                np.isfinite(low_mean)
                and np.isfinite(high_mean)
                and low_mean >= EXPOSURE_THRESHOLD
                and high_mean >= EXPOSURE_THRESHOLD
            )

            if not exposure_active:
                classification = "inactive"
                failure_component = "insufficient_pre_or_post_threshold_offer_exposure"
            else:
                classification = _effect_status(d_mean, d_low, d_high)
                failure_component = (
                    ""
                    if classification == "supported"
                    else "pre_minus_post_gap_effect"
                )

            rows.append(
                {
                    **metadata,
                    "n_paired_seeds": n,
                    "mean_class2_low_regime_offer_share": low_mean,
                    "mean_class2_high_regime_offer_share": high_mean,
                    "exposure_active": exposure_active,
                    "pre_minus_post_absolute_served_gap": d_mean,
                    "pre_minus_post_absolute_served_gap_ci_low": d_low,
                    "pre_minus_post_absolute_served_gap_ci_high": d_high,
                    "failure_component": failure_component,
                    "classification": classification,
                }
            )

    return pd.DataFrame(rows)


def _scenario_effect_rows(
    design: pd.DataFrame,
    raw: pd.DataFrame,
    gap_effects: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for design_row in design.to_dict(orient="records"):
        background_id = str(design_row["background_id"])
        metadata = {
            "background_id": background_id,
            "source_scenario_ids": design_row.get("source_scenario_ids", ""),
            "scenario_type": design_row.get("scenario_type", ""),
            **{column: design_row[column] for column in PARAMETER_COLUMNS},
            "valid_h7_gaps": design_row["valid_h7_gaps"],
        }

        if not bool(design_row["h7_design_active"]):
            rows.append(
                {
                    **metadata,
                    "n_active_gaps": 0,
                    "failure_component": design_row["h7_design_inactive_reason"],
                    "classification": "inactive",
                }
            )
            continue

        effects = gap_effects[
            gap_effects["background_id"].astype(str) == background_id
        ]
        active_effects = effects[effects["classification"] != "inactive"].copy()
        active_gaps = tuple(active_effects["gap_magnitude_focal"].astype(float))

        if len(active_gaps) < MIN_ACTIVE_GAPS:
            rows.append(
                {
                    **metadata,
                    "n_active_gaps": int(len(active_gaps)),
                    "failure_component": "fewer_than_two_exposure_active_gaps",
                    "classification": "inactive",
                }
            )
            continue

        group = raw[raw["background_id"].astype(str) == background_id]
        seed_effects: list[pd.Series] = []

        for gap in active_gaps:
            pre = group[
                np.isclose(group["gap_magnitude_focal"].astype(float), gap)
                & group["gap_location_arm"].eq("pre")
            ].set_index("seed")
            post = group[
                np.isclose(group["gap_magnitude_focal"].astype(float), gap)
                & group["gap_location_arm"].eq("post")
            ].set_index("seed")
            common = pre.index.intersection(post.index)
            effect = (
                pre.loc[common, "absolute_served_rate_gap"]
                - post.loc[common, "absolute_served_rate_gap"]
            )
            effect.name = _gap_key(gap)
            seed_effects.append(effect)

        effect_matrix = pd.concat(seed_effects, axis=1)
        average_effect_by_seed = effect_matrix.mean(axis=1, skipna=False)
        mean, low, high, n = _paired_ci(average_effect_by_seed)
        classification = _effect_status(mean, low, high)

        n_supported = int((active_effects["classification"] == "supported").sum())
        n_reversed = int((active_effects["classification"] == "reversed").sum())
        n_inconclusive = int(
            (active_effects["classification"] == "inconclusive").sum()
        )

        if classification == "supported":
            failure_component = ""
        elif classification == "reversed":
            failure_component = "average_pre_minus_post_effect_reversed"
        else:
            failure_component = "average_pre_minus_post_effect_inconclusive"

        rows.append(
            {
                **metadata,
                "n_active_gaps": int(len(active_gaps)),
                "n_supported_gap_comparisons": n_supported,
                "n_reversed_gap_comparisons": n_reversed,
                "n_inconclusive_gap_comparisons": n_inconclusive,
                "n_complete_paired_seeds": n,
                "average_pre_minus_post_absolute_served_gap": mean,
                "average_pre_minus_post_absolute_served_gap_ci_low": low,
                "average_pre_minus_post_absolute_served_gap_ci_high": high,
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

    gap_effects = _gap_effect_rows(design, raw)
    scenario_effects = _scenario_effect_rows(design, raw, gap_effects)

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    gap_path = summary_dir / "h7_gap_effects.csv"
    scenario_path = summary_dir / "h7_scenario_effects.csv"
    gap_effects.to_csv(gap_path, index=False)
    scenario_effects.to_csv(scenario_path, index=False)

    counts = (
        scenario_effects.groupby("classification", dropna=False)
        .size()
        .rename("n_scenarios")
        .reset_index()
    )
    counts.to_csv(summary_dir / "h7_classification_counts.csv", index=False)

    gap_counts = (
        gap_effects.groupby(
            ["gap_magnitude_focal", "classification"],
            dropna=False,
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    gap_counts.to_csv(
        summary_dir / "h7_gap_classification_counts.csv",
        index=False,
    )

    failures = scenario_effects[
        scenario_effects["classification"].isin(["reversed", "inconclusive"])
    ].copy()
    failures.to_csv(summary_dir / "h7_failure_candidates.csv", index=False)
    failures.to_csv(summary_dir / "h7_stage2_candidates.csv", index=False)

    _write_summary_markdown(
        scenario_effects,
        gap_effects,
        counts,
        gap_counts,
        summary_dir / "h7_stage1_summary.md",
    )
    print(f"Scenario effects: {scenario_path}")
    print(f"Stage 2 candidates: {summary_dir / 'h7_stage2_candidates.csv'}")
    return scenario_path


def _write_summary_markdown(
    scenario_effects: pd.DataFrame,
    gap_effects: pd.DataFrame,
    counts: pd.DataFrame,
    gap_counts: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# H7 Stage 1 Robustness Summary",
        "",
        f"Background scenarios classified: **{len(scenario_effects)}**",
        f"Gap-location comparisons classified: **{len(gap_effects)}**",
        "",
        "## Scenario classification counts",
        "",
        counts.to_markdown(index=False) if not counts.empty else "No results.",
        "",
        "## Gap-level classification counts",
        "",
        gap_counts.to_markdown(index=False) if not gap_counts.empty else "No results.",
        "",
        "## Interpretation",
        "",
        (
            "- For each gap magnitude, the Class 2 within-class balking step is "
            "identical in the pre-gap and post-gap arms; only the location of "
            "the between-class difference changes."
        ),
        (
            "- Support requires the absolute served-rate gap to be at least "
            "0.0025 larger in the pre-threshold-gap arm, with a paired 95% "
            "confidence interval above zero."
        ),
        (
            "- Scenario classification uses the paired average effect across "
            "all exposure-active valid gap magnitudes."
        ),
        (
            "- A gap comparison requires at least 1% of Class 2 offers in both "
            "the pre-threshold and post-threshold regimes."
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
    design_path = args.output_dir / "design" / "h7_background_scenarios.csv"
    raw_path = args.output_dir / "raw" / "h7_stage1_raw.csv"

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
                f"H7 design file not found: {design_path}. Run the experiment first."
            )
        classify_stage1(
            design_path=design_path,
            raw_path=raw_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
