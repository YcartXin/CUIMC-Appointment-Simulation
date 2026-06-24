"""Run and classify Stage 1 robustness tests for H2.

H2: When losses concentrate at balking, utilization is preserved. When losses
concentrate at no-show, utilization drops. The affected class's served rate
decreases in both cases.

The experiment compares Class 1 balking-only and no-show-only interventions at
matched realized loss shares of approximately 5%, 10%, and 20% of Class 1
arrivals. Class 1 threshold probabilities are constant across booking delays in
this experiment, so its focal thresholds and probabilities are removed from
the background signature.

Run from the repository root with module syntax:

    py -3 -m experiments.robustness.h2_stage1 all --smoke --workers 1 --no-resume
    py -3 -m experiments.robustness.h2_stage1 all --workers 4
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

from experiments.robustness.scenario_space import PARAMETER_COLUMNS, STAGE1_SEEDS  # noqa: E402

DEFAULT_BASE_CONFIG = REPO_DIR / "configs" / "baseline.yaml"
DEFAULT_SCENARIOS = (
    REPO_DIR / "outputs" / "robustness" / "scenarios" / "all_stage1_scenarios.csv"
)
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "robustness" / "h2"

TARGET_LOSS_SHARES = (0.05, 0.10, 0.20)
CALIBRATION_SEEDS = (900, 901, 902)
CALIBRATION_ITERATIONS = 3
LOSS_MATCH_TOLERANCE = 0.01
TARGET_TOLERANCE = 0.02
UTILIZATION_THRESHOLD = 0.005
SERVED_RATE_THRESHOLD = 0.005
BASELINE_TARGET_SENTINEL = -1.0

FOCAL_COLUMNS = {
    "balk_threshold_class1",
    "balk_low_class1",
    "balk_high_class1",
    "noshow_threshold_class1",
    "noshow_low_class1",
    "noshow_high_class1",
}


def _clean_string(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def prepare_h2_backgrounds(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Remove H2 focal Class 1 behavior from the background and deduplicate."""
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
        first["source_scenario_ids"] = ";".join(group["source_scenario_id"].astype(str))
        first["source_scenario_count"] = int(len(group))
        grouped_rows.append(first)

    out = pd.DataFrame(grouped_rows).reset_index(drop=True)
    out.insert(0, "background_id", [f"H2B{i:04d}" for i in range(1, len(out) + 1)])

    # The focal probabilities are constant across delay, so threshold values do
    # not affect any H2 arm. Concrete zero values keep the simulation adapter
    # valid while making that irrelevance explicit.
    out["balk_threshold_class1"] = 0
    out["noshow_threshold_class1"] = 0
    out["balk_low_class1"] = 0.0
    out["balk_high_class1"] = 0.0
    out["noshow_low_class1"] = 0.0
    out["noshow_high_class1"] = 0.0
    return out


def _arm_overrides(arm: str, probability: float) -> dict[str, float | int]:
    p = float(np.clip(probability, 0.0, 1.0))
    common = {
        "balk_threshold_class1": 0,
        "noshow_threshold_class1": 0,
    }
    if arm == "baseline":
        return {
            **common,
            "balk_low_class1": 0.0,
            "balk_high_class1": 0.0,
            "noshow_low_class1": 0.0,
            "noshow_high_class1": 0.0,
        }
    if arm == "balk":
        return {
            **common,
            "balk_low_class1": p,
            "balk_high_class1": p,
            "noshow_low_class1": 0.0,
            "noshow_high_class1": 0.0,
        }
    if arm == "noshow":
        return {
            **common,
            "balk_low_class1": 0.0,
            "balk_high_class1": 0.0,
            "noshow_low_class1": p,
            "noshow_high_class1": p,
        }
    raise ValueError(f"Unknown H2 arm: {arm}")


def _loss_share(metrics: Mapping[str, Any], arm: str) -> float:
    if arm == "balk":
        return float(metrics["class_1_balk_rate_per_arrival"])
    if arm == "noshow":
        return float(metrics["class_1_no_show_rate_per_arrival"])
    if arm == "baseline":
        return 0.0
    raise ValueError(arm)


def _evaluate_probability(
    row: Mapping[str, Any],
    *,
    arm: str,
    probability: float,
    seeds: Sequence[int],
    base_config_path: str | Path,
    cache: dict[tuple[str, float], float],
) -> float:
    key = (arm, round(float(probability), 6))
    if key in cache:
        return cache[key]

    from experiments.robustness.simulation_adapter import run_scenario

    values = []
    for seed in seeds:
        metrics = run_scenario(
            row,
            seed=int(seed),
            base_config_path=base_config_path,
            overrides=_arm_overrides(arm, probability),
        )
        values.append(_loss_share(metrics, arm))
    estimate = float(np.mean(values)) if values else math.nan
    cache[key] = estimate
    return estimate


def _calibrate_arm(
    row: Mapping[str, Any],
    *,
    arm: str,
    seeds: Sequence[int],
    base_config_path: str | Path,
) -> dict[float, dict[str, Any]]:
    cache: dict[tuple[str, float], float] = {}
    max_loss = _evaluate_probability(
        row,
        arm=arm,
        probability=1.0,
        seeds=seeds,
        base_config_path=base_config_path,
        cache=cache,
    )

    calibrated: dict[float, dict[str, Any]] = {}
    for target in TARGET_LOSS_SHARES:
        if not np.isfinite(max_loss) or max_loss <= 0.0:
            calibrated[target] = {
                "probability": math.nan,
                "estimated_loss": max_loss,
                "target_error": math.nan,
                "attainable": False,
            }
            continue

        candidates: list[tuple[float, float]] = [(1.0, max_loss)]
        p = float(np.clip(target / max_loss, 0.0, 1.0))
        for _ in range(CALIBRATION_ITERATIONS):
            observed = _evaluate_probability(
                row,
                arm=arm,
                probability=p,
                seeds=seeds,
                base_config_path=base_config_path,
                cache=cache,
            )
            candidates.append((p, observed))
            if np.isfinite(observed) and abs(observed - target) <= 0.0025:
                break
            if not np.isfinite(observed) or observed <= 1e-10:
                p = min(1.0, p + 0.10)
            else:
                updated = float(np.clip(p * target / observed, 0.0, 1.0))
                if abs(updated - p) < 0.0005:
                    break
                p = updated

        valid_candidates = [x for x in candidates if np.isfinite(x[1])]
        if not valid_candidates:
            best_p, best_loss = math.nan, math.nan
        else:
            best_p, best_loss = min(valid_candidates, key=lambda x: abs(x[1] - target))
        error = abs(best_loss - target) if np.isfinite(best_loss) else math.nan
        calibrated[target] = {
            "probability": float(best_p),
            "estimated_loss": float(best_loss),
            "target_error": float(error),
            "attainable": bool(
                max_loss >= target - TARGET_TOLERANCE
                and np.isfinite(error)
                and error <= TARGET_TOLERANCE
            ),
        }
    return calibrated


def _calibrate_task(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    row = dict(task["row"])
    seeds = tuple(int(x) for x in task["seeds"])
    base_config_path = task["base_config_path"]
    balk = _calibrate_arm(
        row,
        arm="balk",
        seeds=seeds,
        base_config_path=base_config_path,
    )
    noshow = _calibrate_arm(
        row,
        arm="noshow",
        seeds=seeds,
        base_config_path=base_config_path,
    )

    rows: list[dict[str, Any]] = []
    for target in TARGET_LOSS_SHARES:
        b = balk[target]
        n = noshow[target]
        estimated_match_gap = (
            abs(float(b["estimated_loss"]) - float(n["estimated_loss"]))
            if np.isfinite(b["estimated_loss"]) and np.isfinite(n["estimated_loss"])
            else math.nan
        )
        rows.append(
            {
                "background_id": row["background_id"],
                "source_scenario_ids": _clean_string(row.get("source_scenario_ids")),
                "scenario_type": row["scenario_type"],
                "target_loss_share": float(target),
                "balk_probability": b["probability"],
                "noshow_probability": n["probability"],
                "estimated_balk_loss_share": b["estimated_loss"],
                "estimated_noshow_loss_share": n["estimated_loss"],
                "estimated_match_gap": estimated_match_gap,
                "balk_target_error": b["target_error"],
                "noshow_target_error": n["target_error"],
                "balk_attainable": b["attainable"],
                "noshow_attainable": n["attainable"],
                "calibration_valid": bool(b["attainable"] and n["attainable"]),
                "n_calibration_seeds": len(seeds),
                **{column: row[column] for column in PARAMETER_COLUMNS},
            }
        )
    return rows


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _selected_backgrounds(
    scenarios_path: Path,
    *,
    smoke: bool,
    max_scenarios: int | None,
) -> pd.DataFrame:
    backgrounds = prepare_h2_backgrounds(pd.read_csv(scenarios_path))
    if smoke:
        backgrounds = backgrounds.head(2).copy()
    if max_scenarios is not None:
        backgrounds = backgrounds.head(max_scenarios).copy()
    return backgrounds


def run_calibration(
    *,
    scenarios_path: Path,
    output_dir: Path,
    base_config_path: Path,
    workers: int,
    smoke: bool,
    max_scenarios: int | None,
    resume: bool,
) -> Path:
    backgrounds = _selected_backgrounds(
        scenarios_path, smoke=smoke, max_scenarios=max_scenarios
    )
    seeds = CALIBRATION_SEEDS[:1] if smoke else CALIBRATION_SEEDS

    design_dir = output_dir / "design"
    calibration_dir = output_dir / "calibration"
    background_path = design_dir / "h2_background_scenarios.csv"
    calibration_path = calibration_dir / "h2_loss_calibration.csv"
    design_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    backgrounds.to_csv(background_path, index=False)

    old = pd.DataFrame()
    completed: set[str] = set()
    if resume and calibration_path.exists():
        old = pd.read_csv(calibration_path)
        complete_counts = (
            old.groupby("background_id")["n_calibration_seeds"]
            .max()
            .to_dict()
        )
        completed = {
            str(background_id)
            for background_id, n in complete_counts.items()
            if int(n) >= len(seeds)
        }
    elif calibration_path.exists():
        calibration_path.unlink()

    tasks = [
        {
            "row": row,
            "seeds": seeds,
            "base_config_path": str(base_config_path),
        }
        for row in backgrounds.to_dict(orient="records")
        if str(row["background_id"]) not in completed
    ]

    print(f"H2 calibration backgrounds: {len(backgrounds)}")
    print(f"Calibration seeds: {len(seeds)}")
    print(f"Backgrounds already calibrated: {len(completed)}")
    print(f"Backgrounds to calibrate now: {len(tasks)}")

    new_rows: list[dict[str, Any]] = []
    if workers <= 1:
        iterator: Iterable[list[dict[str, Any]]] = map(_calibrate_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_calibrate_task, tasks, chunksize=1)

    try:
        for index, rows in enumerate(iterator, start=1):
            new_rows.extend(rows)
            if index % 25 == 0 or index == len(tasks):
                print(f"Calibrated {index:,}/{len(tasks):,} new backgrounds")
    finally:
        if executor is not None:
            executor.shutdown()

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        replaced_ids = set(new_df["background_id"].astype(str))
        if not old.empty:
            old = old[~old["background_id"].astype(str).isin(replaced_ids)]
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.sort_values(["background_id", "target_loss_share"])
        combined.to_csv(calibration_path, index=False)
    elif not calibration_path.exists():
        pd.DataFrame().to_csv(calibration_path, index=False)

    print(f"Calibration results: {calibration_path}")
    return calibration_path


def _main_task_payloads(
    backgrounds: pd.DataFrame,
    calibration: pd.DataFrame,
    seeds: Sequence[int],
    completed: set[tuple[str, float, str, int]],
    base_config_path: str | Path,
) -> Iterable[dict[str, Any]]:
    row_map = {
        str(row["background_id"]): row
        for row in backgrounds.to_dict(orient="records")
    }
    valid = calibration[calibration["calibration_valid"].map(_bool_value)].copy()

    for background_id, group in valid.groupby("background_id", sort=False):
        background_id = str(background_id)
        row = row_map.get(background_id)
        if row is None:
            continue

        for seed in seeds:
            baseline_key = (
                background_id,
                BASELINE_TARGET_SENTINEL,
                "baseline",
                int(seed),
            )
            if baseline_key not in completed:
                yield {
                    "row": row,
                    "background_id": background_id,
                    "target_loss_share": BASELINE_TARGET_SENTINEL,
                    "arm": "baseline",
                    "focal_probability": 0.0,
                    "seed": int(seed),
                    "base_config_path": str(base_config_path),
                }

        for calibration_row in group.to_dict(orient="records"):
            target = float(calibration_row["target_loss_share"])
            for arm, probability_column in (
                ("balk", "balk_probability"),
                ("noshow", "noshow_probability"),
            ):
                probability = float(calibration_row[probability_column])
                for seed in seeds:
                    key = (background_id, target, arm, int(seed))
                    if key in completed:
                        continue
                    yield {
                        "row": row,
                        "background_id": background_id,
                        "target_loss_share": target,
                        "arm": arm,
                        "focal_probability": probability,
                        "seed": int(seed),
                        "base_config_path": str(base_config_path),
                    }


def _run_main_task(task: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.robustness.simulation_adapter import run_scenario

    row = dict(task["row"])
    arm = str(task["arm"])
    probability = float(task["focal_probability"])
    seed = int(task["seed"])
    metrics = run_scenario(
        row,
        seed=seed,
        base_config_path=task["base_config_path"],
        overrides=_arm_overrides(arm, probability),
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
        "cancel_class1_background": float(row["cancel_class1"]),
        "cancel_class2_background": float(row["cancel_class2"]),
        "target_loss_share": float(task["target_loss_share"]),
        "arm": arm,
        "focal_probability": probability,
        "realized_focal_loss_share": _loss_share(metrics, arm),
        **metrics,
    }


def run_stage1(
    *,
    scenarios_path: Path,
    calibration_path: Path,
    output_dir: Path,
    base_config_path: Path,
    workers: int,
    smoke: bool,
    max_scenarios: int | None,
    resume: bool,
) -> Path:
    backgrounds = _selected_backgrounds(
        scenarios_path, smoke=smoke, max_scenarios=max_scenarios
    )
    seeds = STAGE1_SEEDS[:2] if smoke else STAGE1_SEEDS
    calibration = pd.read_csv(calibration_path)
    calibration = calibration[
        calibration["background_id"].astype(str).isin(
            set(backgrounds["background_id"].astype(str))
        )
    ].copy()

    raw_path = output_dir / "raw" / "h2_stage1_raw.csv"
    completed: set[tuple[str, float, str, int]] = set()
    if resume and raw_path.exists():
        old = pd.read_csv(
            raw_path,
            usecols=["background_id", "target_loss_share", "arm", "seed"],
        )
        completed = {
            (
                str(r.background_id),
                float(r.target_loss_share),
                str(r.arm),
                int(r.seed),
            )
            for r in old.itertuples(index=False)
        }
    elif raw_path.exists():
        raw_path.unlink()

    tasks = list(
        _main_task_payloads(
            backgrounds,
            calibration,
            seeds,
            completed,
            base_config_path,
        )
    )
    valid = calibration[calibration["calibration_valid"].map(_bool_value)]
    valid_targets = int(len(valid))
    active_backgrounds = int(valid["background_id"].nunique()) if not valid.empty else 0
    total_expected = (active_backgrounds + 2 * valid_targets) * len(seeds)

    print(f"H2 backgrounds: {len(backgrounds)}")
    print(f"Valid matched targets: {valid_targets}")
    print(f"Stage 1 seeds: {len(seeds)}")
    print(f"Expected rows after completion: {total_expected}")
    print(f"Rows already completed: {len(completed)}")
    print(f"Rows to run now: {len(tasks)}")

    buffer: list[dict[str, Any]] = []
    flush_every = 100
    if workers <= 1:
        iterator: Iterable[dict[str, Any]] = map(_run_main_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_run_main_task, tasks, chunksize=4)

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


def _paired_ci(
    values: pd.Series, confidence: float = 0.95
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


def _target_effect_row(
    calibration_row: Mapping[str, Any],
    raw: pd.DataFrame,
) -> dict[str, Any]:
    background_id = str(calibration_row["background_id"])
    target = float(calibration_row["target_loss_share"])
    base_metadata = {
        "background_id": background_id,
        "source_scenario_ids": calibration_row.get("source_scenario_ids", ""),
        "scenario_type": calibration_row.get("scenario_type", ""),
        "rho": float(calibration_row["rho"]),
        "class1_share": float(calibration_row["class1_share"]),
        "slots_per_day": int(calibration_row["slots_per_day"]),
        "horizon_class1": int(calibration_row["horizon_class1"]),
        "horizon_class2": int(calibration_row["horizon_class2"]),
        "cancel_class1": float(calibration_row["cancel_class1"]),
        "cancel_class2": float(calibration_row["cancel_class2"]),
        "target_loss_share": target,
        "balk_probability": calibration_row.get("balk_probability", math.nan),
        "noshow_probability": calibration_row.get("noshow_probability", math.nan),
    }

    if not _bool_value(calibration_row["calibration_valid"]):
        return {
            **base_metadata,
            "n_paired_seeds": 0,
            "mean_balk_loss_share": calibration_row.get(
                "estimated_balk_loss_share", math.nan
            ),
            "mean_noshow_loss_share": calibration_row.get(
                "estimated_noshow_loss_share", math.nan
            ),
            "loss_match_gap": calibration_row.get("estimated_match_gap", math.nan),
            "loss_match_valid": False,
            "utilization_component": "inactive",
            "balk_served_component": "inactive",
            "noshow_served_component": "inactive",
            "failure_component": "target_unattainable",
            "classification": "inactive",
        }

    background_raw = raw[raw["background_id"].astype(str) == background_id]
    baseline = background_raw[
        (background_raw["arm"] == "baseline")
        & (background_raw["target_loss_share"] == BASELINE_TARGET_SENTINEL)
    ].set_index("seed")
    balk = background_raw[
        (background_raw["arm"] == "balk")
        & np.isclose(background_raw["target_loss_share"], target)
    ].set_index("seed")
    noshow = background_raw[
        (background_raw["arm"] == "noshow")
        & np.isclose(background_raw["target_loss_share"], target)
    ].set_index("seed")

    common = baseline.index.intersection(balk.index).intersection(noshow.index)
    if len(common) == 0:
        return {
            **base_metadata,
            "n_paired_seeds": 0,
            "loss_match_valid": False,
            "utilization_component": "inconclusive",
            "balk_served_component": "inconclusive",
            "noshow_served_component": "inconclusive",
            "failure_component": "missing_paired_runs",
            "classification": "inconclusive",
        }

    baseline = baseline.loc[common]
    balk = balk.loc[common]
    noshow = noshow.loc[common]

    mean_balk_loss = float(balk["realized_focal_loss_share"].mean())
    mean_noshow_loss = float(noshow["realized_focal_loss_share"].mean())
    match_gap = abs(mean_balk_loss - mean_noshow_loss)
    loss_match_valid = bool(
        match_gap <= LOSS_MATCH_TOLERANCE
        and abs(mean_balk_loss - target) <= TARGET_TOLERANCE
        and abs(mean_noshow_loss - target) <= TARGET_TOLERANCE
    )

    util_difference = balk["average_utilization"] - noshow["average_utilization"]
    balk_served_change = (
        balk["class_1_percent_serviced"] - baseline["class_1_percent_serviced"]
    )
    noshow_served_change = (
        noshow["class_1_percent_serviced"] - baseline["class_1_percent_serviced"]
    )
    balk_util_change = balk["average_utilization"] - baseline["average_utilization"]
    noshow_util_change = noshow["average_utilization"] - baseline["average_utilization"]

    u_mean, u_low, u_high, n = _paired_ci(util_difference)
    bs_mean, bs_low, bs_high, _ = _paired_ci(balk_served_change)
    ns_mean, ns_low, ns_high, _ = _paired_ci(noshow_served_change)
    bu_mean, bu_low, bu_high, _ = _paired_ci(balk_util_change)
    nu_mean, nu_low, nu_high, _ = _paired_ci(noshow_util_change)

    util_status = _component_status(
        u_mean,
        u_low,
        u_high,
        expected="positive",
        practical_threshold=UTILIZATION_THRESHOLD,
    )
    balk_served_status = _component_status(
        bs_mean,
        bs_low,
        bs_high,
        expected="negative",
        practical_threshold=SERVED_RATE_THRESHOLD,
    )
    noshow_served_status = _component_status(
        ns_mean,
        ns_low,
        ns_high,
        expected="negative",
        practical_threshold=SERVED_RATE_THRESHOLD,
    )

    components = {
        "utilization": util_status,
        "balk_served": balk_served_status,
        "noshow_served": noshow_served_status,
    }
    if not loss_match_valid:
        classification = "inconclusive"
        failure_component = "loss_matching_failed"
    elif "reversed" in components.values():
        classification = "reversed"
        failure_component = ";".join(
            name for name, status in components.items() if status == "reversed"
        )
    elif all(status == "supported" for status in components.values()):
        classification = "supported"
        failure_component = ""
    else:
        classification = "inconclusive"
        failure_component = ";".join(
            name for name, status in components.items() if status != "supported"
        )

    return {
        **base_metadata,
        "n_paired_seeds": n,
        "mean_balk_loss_share": mean_balk_loss,
        "mean_noshow_loss_share": mean_noshow_loss,
        "balk_target_error": abs(mean_balk_loss - target),
        "noshow_target_error": abs(mean_noshow_loss - target),
        "loss_match_gap": match_gap,
        "loss_match_valid": loss_match_valid,
        "delta_utilization_balk_minus_noshow": u_mean,
        "delta_utilization_balk_minus_noshow_ci_low": u_low,
        "delta_utilization_balk_minus_noshow_ci_high": u_high,
        "utilization_component": util_status,
        "delta_class1_served_balk_vs_baseline": bs_mean,
        "delta_class1_served_balk_vs_baseline_ci_low": bs_low,
        "delta_class1_served_balk_vs_baseline_ci_high": bs_high,
        "balk_served_component": balk_served_status,
        "delta_class1_served_noshow_vs_baseline": ns_mean,
        "delta_class1_served_noshow_vs_baseline_ci_low": ns_low,
        "delta_class1_served_noshow_vs_baseline_ci_high": ns_high,
        "noshow_served_component": noshow_served_status,
        "delta_utilization_balk_vs_baseline": bu_mean,
        "delta_utilization_balk_vs_baseline_ci_low": bu_low,
        "delta_utilization_balk_vs_baseline_ci_high": bu_high,
        "delta_utilization_noshow_vs_baseline": nu_mean,
        "delta_utilization_noshow_vs_baseline_ci_low": nu_low,
        "delta_utilization_noshow_vs_baseline_ci_high": nu_high,
        "failure_component": failure_component,
        "classification": classification,
    }


def _aggregate_scenario_effects(target_effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for background_id, group in target_effects.groupby("background_id", sort=False):
        first = group.iloc[0]
        counts = group["classification"].value_counts().to_dict()
        n_supported = int(counts.get("supported", 0))
        n_reversed = int(counts.get("reversed", 0))
        n_inconclusive = int(counts.get("inconclusive", 0))
        n_inactive = int(counts.get("inactive", 0))
        n_active = n_supported + n_reversed + n_inconclusive

        if n_active == 0:
            classification = "inactive"
        elif n_reversed >= 2 or (n_reversed >= 1 and n_supported == 0):
            classification = "reversed"
        elif n_supported >= 2 and n_reversed == 0:
            classification = "supported"
        else:
            classification = "inconclusive"

        failure_components = sorted(
            {
                component
                for value in group["failure_component"].fillna("").astype(str)
                for component in value.split(";")
                if component
            }
        )
        rows.append(
            {
                "background_id": background_id,
                "source_scenario_ids": first.get("source_scenario_ids", ""),
                "scenario_type": first.get("scenario_type", ""),
                "rho": float(first["rho"]),
                "class1_share": float(first["class1_share"]),
                "slots_per_day": int(first["slots_per_day"]),
                "horizon_class1": int(first["horizon_class1"]),
                "horizon_class2": int(first["horizon_class2"]),
                "cancel_class1": float(first["cancel_class1"]),
                "cancel_class2": float(first["cancel_class2"]),
                "n_supported_targets": n_supported,
                "n_reversed_targets": n_reversed,
                "n_inconclusive_targets": n_inconclusive,
                "n_inactive_targets": n_inactive,
                "n_loss_matched_targets": int(group["loss_match_valid"].fillna(False).sum()),
                "failure_component": ";".join(failure_components),
                "classification": classification,
            }
        )
    return pd.DataFrame(rows)


def classify_stage1(
    *,
    raw_path: Path,
    calibration_path: Path,
    output_dir: Path,
) -> Path:
    raw = pd.read_csv(raw_path)
    calibration = pd.read_csv(calibration_path)
    target_rows = [
        _target_effect_row(row, raw)
        for row in calibration.to_dict(orient="records")
    ]
    target_effects = pd.DataFrame(target_rows)
    scenario_effects = _aggregate_scenario_effects(target_effects)

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    target_path = summary_dir / "h2_target_effects.csv"
    scenario_path = summary_dir / "h2_scenario_effects.csv"
    target_effects.to_csv(target_path, index=False)
    scenario_effects.to_csv(scenario_path, index=False)

    counts = (
        scenario_effects.groupby("classification", dropna=False)
        .size()
        .rename("n_scenarios")
        .reset_index()
    )
    counts.to_csv(summary_dir / "h2_classification_counts.csv", index=False)

    failures = target_effects[
        target_effects["classification"].isin(["reversed", "inconclusive"])
    ].copy()
    failures.to_csv(summary_dir / "h2_failure_candidates.csv", index=False)

    stage2 = target_effects[
        target_effects["loss_match_valid"].fillna(False)
        & target_effects["classification"].isin(["reversed", "inconclusive"])
    ].copy()
    stage2.to_csv(summary_dir / "h2_stage2_candidates.csv", index=False)

    _write_summary_markdown(
        target_effects,
        scenario_effects,
        summary_dir / "h2_stage1_summary.md",
    )
    print(f"Target effects: {target_path}")
    print(f"Scenario effects: {scenario_path}")
    print(f"Stage 2 candidates: {summary_dir / 'h2_stage2_candidates.csv'}")
    return scenario_path


def _write_summary_markdown(
    target_effects: pd.DataFrame,
    scenario_effects: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# H2 Stage 1 Robustness Summary",
        "",
        f"Background scenarios classified: **{len(scenario_effects)}**",
        f"Target-level comparisons classified: **{len(target_effects)}**",
        "",
        "## Scenario classification counts",
        "",
    ]
    if scenario_effects.empty:
        lines.append("No complete scenarios were available for classification.")
    else:
        scenario_table = (
            scenario_effects.groupby("classification")
            .size()
            .rename("n_scenarios")
            .to_frame()
        )
        lines.append(scenario_table.to_markdown())
        lines.extend(["", "## Target-level classification counts", ""])
        target_table = (
            target_effects.groupby(["target_loss_share", "classification"])
            .size()
            .unstack(fill_value=0)
        )
        lines.append(target_table.to_markdown())
        matched = int(target_effects["loss_match_valid"].fillna(False).sum())
        lines.extend(
            [
                "",
                "## Matching diagnostics",
                "",
                f"- Target comparisons meeting the realized-loss matching rule: **{matched}/{len(target_effects)}**.",
                f"- Matching requires a between-arm loss-share gap no greater than {LOSS_MATCH_TOLERANCE:.2f} and each arm to be within {TARGET_TOLERANCE:.2f} of its target.",
                "",
                "## Interpretation",
                "",
                "- Support requires utilization to be materially higher in the balking arm than in the no-show arm.",
                "- Class 1 served rate must also fall materially in both arms relative to the zero-focal-loss baseline.",
                "- Matched reversed and inconclusive target comparisons are exported for Stage 2 confirmation.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["calibrate", "run", "classify", "all"]
    )
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    calibration_path = args.output_dir / "calibration" / "h2_loss_calibration.csv"
    raw_path = args.output_dir / "raw" / "h2_stage1_raw.csv"

    if args.command in {"calibrate", "all"}:
        calibration_path = run_calibration(
            scenarios_path=args.scenarios,
            output_dir=args.output_dir,
            base_config_path=args.base_config,
            workers=args.workers,
            smoke=args.smoke,
            max_scenarios=args.max_scenarios,
            resume=not args.no_resume,
        )

    if args.command in {"run", "all"}:
        if not calibration_path.exists():
            raise FileNotFoundError(
                f"H2 calibration file not found: {calibration_path}. Run calibrate first."
            )
        raw_path = run_stage1(
            scenarios_path=args.scenarios,
            calibration_path=calibration_path,
            output_dir=args.output_dir,
            base_config_path=args.base_config,
            workers=args.workers,
            smoke=args.smoke,
            max_scenarios=args.max_scenarios,
            resume=not args.no_resume,
        )

    if args.command in {"classify", "all"}:
        if not calibration_path.exists():
            raise FileNotFoundError(f"H2 calibration file not found: {calibration_path}")
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw H2 results not found: {raw_path}. Run the experiment first."
            )
        classify_stage1(
            raw_path=raw_path,
            calibration_path=calibration_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
