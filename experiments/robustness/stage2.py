"""Stage 2 confirmation runner for robustness hypotheses H1-H9.

Stage 2 uses 100 new paired seeds (2000-2099). It confirms every Stage 1
reversal and reruns only uncertainty-limited inconclusive cases whose point
estimates already satisfy the hypothesis's practical-effect criteria.

Run from the repository root:

    py -3 -m experiments.robustness.stage2 select
    py -3 -m experiments.robustness.stage2 all --workers 4

Use ``--smoke`` before the full run. The full run is resumable.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from experiments.robustness.scenario_space import STAGE2_SEEDS  # noqa: E402

DEFAULT_STAGE1_ROOT = REPO_DIR / "outputs" / "robustness"
DEFAULT_OUTPUT_DIR = DEFAULT_STAGE1_ROOT / "stage2"
DEFAULT_BASE_CONFIG = REPO_DIR / "configs" / "baseline.yaml"
HYPOTHESES = tuple(f"h{i}" for i in range(1, 10))

SERVED_THRESHOLD = 0.005
UTILIZATION_THRESHOLD = 0.005
DELAY_THRESHOLD = 0.25
COMPARATIVE_THRESHOLD = 0.0025
SPEARMAN_THRESHOLD = 0.50


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _bool_series(series: pd.Series) -> pd.Series:
    return series.map(_as_bool)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required Stage 1 file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _scenario_effects_path(stage1_root: Path, hypothesis: str) -> Path:
    return (
        stage1_root
        / hypothesis
        / "summary"
        / f"{hypothesis}_scenario_effects.csv"
    )


def _design_path(stage1_root: Path, hypothesis: str) -> Path:
    return (
        stage1_root
        / hypothesis
        / "design"
        / f"{hypothesis}_background_scenarios.csv"
    )


def _candidate_rows(
    scenario_effects: pd.DataFrame,
    selected_ids: set[str],
    reasons: Mapping[str, str],
    hypothesis: str,
) -> pd.DataFrame:
    if not selected_ids:
        return pd.DataFrame(
            columns=[
                "hypothesis",
                "background_id",
                "stage1_classification",
                "selection_reason",
            ]
        )

    out = scenario_effects[
        scenario_effects["background_id"].astype(str).isin(selected_ids)
    ].copy()
    out.insert(0, "hypothesis", hypothesis.upper())
    out["background_id"] = out["background_id"].astype(str)
    out = out.rename(columns={"classification": "stage1_classification"})
    out["selection_reason"] = out["background_id"].map(reasons)
    preferred = [
        "hypothesis",
        "background_id",
        "stage1_classification",
        "selection_reason",
    ]
    remaining = [column for column in out.columns if column not in preferred]
    return out[preferred + remaining]


def _select_h1(stage1_root: Path) -> pd.DataFrame:
    hypothesis = "h1"
    effects = _read_csv(_scenario_effects_path(stage1_root, hypothesis))
    classification = effects["classification"].astype(str)
    reversed_mask = classification.eq("reversed")

    r1_mean = _numeric(effects["delta_class1_served_rate"])
    delay_mean = _numeric(effects["delta_mean_offered_delay"])
    r1_ok = effects["class1_served_component"].astype(str).eq("supported") | (
        effects["class1_served_component"].astype(str).eq("inconclusive")
        & r1_mean.le(-SERVED_THRESHOLD)
    )
    delay_ok = effects["offered_delay_component"].astype(str).eq("supported") | (
        effects["offered_delay_component"].astype(str).eq("inconclusive")
        & delay_mean.le(-DELAY_THRESHOLD)
    )
    uncertain_mask = (
        classification.eq("inconclusive")
        & effects["demand_regime"].astype(str).isin(["high", "boundary"])
        & r1_ok
        & delay_ok
    )

    reasons = {
        str(background_id): "stage1_reversed"
        for background_id in effects.loc[reversed_mask, "background_id"]
    }
    reasons.update(
        {
            str(background_id): "uncertainty_limited_inconclusive"
            for background_id in effects.loc[uncertain_mask, "background_id"]
        }
    )
    return _candidate_rows(effects, set(reasons), reasons, hypothesis)


def _select_h2(stage1_root: Path) -> pd.DataFrame:
    hypothesis = "h2"
    scenario = _read_csv(_scenario_effects_path(stage1_root, hypothesis))
    target = _read_csv(
        stage1_root / hypothesis / "summary" / "h2_target_effects.csv"
    )
    classification = target["classification"].astype(str)
    match_valid = _bool_series(target["loss_match_valid"])
    reversed_mask = classification.eq("reversed") & match_valid
    uncertain_mask = (
        classification.eq("inconclusive")
        & match_valid
        & _numeric(target["delta_utilization_balk_minus_noshow"]).ge(
            UTILIZATION_THRESHOLD
        )
        & _numeric(target["delta_class1_served_balk_vs_baseline"]).le(
            -SERVED_THRESHOLD
        )
        & _numeric(target["delta_class1_served_noshow_vs_baseline"]).le(
            -SERVED_THRESHOLD
        )
    )

    reversed_ids = set(
        target.loc[reversed_mask, "background_id"].astype(str)
    )
    uncertain_ids = set(
        target.loc[uncertain_mask, "background_id"].astype(str)
    )
    reasons = {background_id: "stage1_reversed" for background_id in reversed_ids}
    for background_id in uncertain_ids:
        reasons.setdefault(background_id, "uncertainty_limited_inconclusive")
    return _candidate_rows(scenario, set(reasons), reasons, hypothesis)


def _select_h3(stage1_root: Path) -> pd.DataFrame:
    hypothesis = "h3"
    effects = _read_csv(_scenario_effects_path(stage1_root, hypothesis))
    classification = effects["classification"].astype(str)
    reversed_mask = classification.eq("reversed")
    uncertain_mask = (
        classification.eq("inconclusive")
        & _numeric(effects["lowest_threshold_delta_utilization"]).le(
            -UTILIZATION_THRESHOLD
        )
        & _numeric(effects["low_minus_high_utilization_loss"]).ge(
            UTILIZATION_THRESHOLD
        )
        & _numeric(effects["spearman_threshold_vs_loss_magnitude"]).le(
            -SPEARMAN_THRESHOLD
        )
    )
    reasons = {
        str(background_id): "stage1_reversed"
        for background_id in effects.loc[reversed_mask, "background_id"]
    }
    reasons.update(
        {
            str(background_id): "uncertainty_limited_inconclusive"
            for background_id in effects.loc[uncertain_mask, "background_id"]
        }
    )
    return _candidate_rows(effects, set(reasons), reasons, hypothesis)


def _select_h4(stage1_root: Path) -> pd.DataFrame:
    hypothesis = "h4"
    effects = _read_csv(_scenario_effects_path(stage1_root, hypothesis))
    classification = effects["classification"].astype(str)
    reversed_mask = classification.eq("reversed")
    peak = _numeric(effects["peak_level"])
    uncertain_mask = (
        classification.eq("inconclusive")
        & effects["demand_regime"].astype(str).eq("high")
        & peak.isin([0.1, 0.3, 0.5])
        & _numeric(effects["hump_rise_from_low"]).ge(DELAY_THRESHOLD)
        & _numeric(effects["hump_fall_to_high"]).ge(DELAY_THRESHOLD)
    )
    reasons = {
        str(background_id): "stage1_reversed"
        for background_id in effects.loc[reversed_mask, "background_id"]
    }
    reasons.update(
        {
            str(background_id): "uncertain_hump_candidate"
            for background_id in effects.loc[uncertain_mask, "background_id"]
        }
    )
    return _candidate_rows(effects, set(reasons), reasons, hypothesis)


def _select_h5(stage1_root: Path) -> pd.DataFrame:
    hypothesis = "h5"
    scenario = _read_csv(_scenario_effects_path(stage1_root, hypothesis))
    steps = _read_csv(
        stage1_root / hypothesis / "summary" / "h5_step_effects.csv"
    )
    reversed_ids = set(
        scenario.loc[
            scenario["classification"].astype(str).eq("reversed"),
            "background_id",
        ].astype(str)
    )
    primary = _bool_series(steps["is_primary_step"])
    uncertainty_rows = steps[
        primary
        & steps["classification"].astype(str).eq("inconclusive")
        & _numeric(steps["delta_class1_accepted_delay"]).le(-DELAY_THRESHOLD)
        & _numeric(steps["selection_gap"]).ge(DELAY_THRESHOLD)
        & _numeric(steps["delta_class1_served_rate"]).le(-SERVED_THRESHOLD)
    ]
    uncertain_ids = set(uncertainty_rows["background_id"].astype(str))
    uncertain_ids &= set(
        scenario.loc[
            scenario["classification"].astype(str).eq("inconclusive"),
            "background_id",
        ].astype(str)
    )

    reasons = {background_id: "stage1_reversed" for background_id in reversed_ids}
    for background_id in uncertain_ids:
        reasons.setdefault(background_id, "uncertainty_limited_inconclusive")
    return _candidate_rows(scenario, set(reasons), reasons, hypothesis)


def _select_h6(stage1_root: Path) -> pd.DataFrame:
    hypothesis = "h6"
    effects = _read_csv(_scenario_effects_path(stage1_root, hypothesis))
    classification = effects["classification"].astype(str)
    reversed_mask = classification.eq("reversed")
    uncertain_mask = (
        classification.eq("inconclusive")
        & _numeric(effects["spearman_bucket_mass_vs_jump"]).ge(
            SPEARMAN_THRESHOLD
        )
        & _bool_series(effects["largest_jump_in_upper_mass_half"])
        & _numeric(effects["largest_absolute_served_rate_jump"]).ge(
            SERVED_THRESHOLD
        )
    )
    reasons = {
        str(background_id): "stage1_reversed"
        for background_id in effects.loc[reversed_mask, "background_id"]
    }
    reasons.update(
        {
            str(background_id): "uncertainty_limited_inconclusive"
            for background_id in effects.loc[uncertain_mask, "background_id"]
        }
    )
    return _candidate_rows(effects, set(reasons), reasons, hypothesis)


def _select_h7(stage1_root: Path) -> pd.DataFrame:
    hypothesis = "h7"
    effects = _read_csv(_scenario_effects_path(stage1_root, hypothesis))
    classification = effects["classification"].astype(str)
    reversed_mask = classification.eq("reversed")
    uncertain_mask = (
        classification.eq("inconclusive")
        & _numeric(effects["average_pre_minus_post_absolute_served_gap"]).ge(
            COMPARATIVE_THRESHOLD
        )
    )
    reasons = {
        str(background_id): "stage1_reversed"
        for background_id in effects.loc[reversed_mask, "background_id"]
    }
    reasons.update(
        {
            str(background_id): "uncertainty_limited_inconclusive"
            for background_id in effects.loc[uncertain_mask, "background_id"]
        }
    )
    return _candidate_rows(effects, set(reasons), reasons, hypothesis)


def _select_h8(stage1_root: Path) -> pd.DataFrame:
    hypothesis = "h8"
    effects = _read_csv(_scenario_effects_path(stage1_root, hypothesis))
    classification = effects["classification"].astype(str)
    reversed_mask = classification.eq("reversed")
    uncertain_mask = (
        classification.eq("inconclusive")
        & _numeric(effects["absolute_effect_difference"]).ge(
            COMPARATIVE_THRESHOLD
        )
    )
    reasons = {
        str(background_id): "stage1_reversed"
        for background_id in effects.loc[reversed_mask, "background_id"]
    }
    reasons.update(
        {
            str(background_id): "uncertainty_limited_inconclusive"
            for background_id in effects.loc[uncertain_mask, "background_id"]
        }
    )
    return _candidate_rows(effects, set(reasons), reasons, hypothesis)


def _select_h9(stage1_root: Path) -> pd.DataFrame:
    hypothesis = "h9"
    effects = _read_csv(_scenario_effects_path(stage1_root, hypothesis))
    classification = effects["classification"].astype(str)
    reversed_mask = classification.eq("reversed")
    uncertain_mask = (
        classification.eq("inconclusive")
        & _numeric(effects["common_minus_gap_utilization_effect"]).ge(
            COMPARATIVE_THRESHOLD
        )
        & _numeric(effects["gap_minus_common_served_gap_effect"]).ge(
            COMPARATIVE_THRESHOLD
        )
    )
    reasons = {
        str(background_id): "stage1_reversed"
        for background_id in effects.loc[reversed_mask, "background_id"]
    }
    reasons.update(
        {
            str(background_id): "uncertainty_limited_inconclusive"
            for background_id in effects.loc[uncertain_mask, "background_id"]
        }
    )
    return _candidate_rows(effects, set(reasons), reasons, hypothesis)


SELECTORS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "h1": _select_h1,
    "h2": _select_h2,
    "h3": _select_h3,
    "h4": _select_h4,
    "h5": _select_h5,
    "h6": _select_h6,
    "h7": _select_h7,
    "h8": _select_h8,
    "h9": _select_h9,
}


def select_candidates(
    *,
    stage1_root: Path,
    output_dir: Path,
    hypotheses: Sequence[str],
) -> Path:
    design_dir = output_dir / "design"
    design_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for hypothesis in hypotheses:
        candidates = SELECTORS[hypothesis](stage1_root)
        path = design_dir / f"{hypothesis}_stage2_candidates.csv"
        candidates.to_csv(path, index=False)
        frames.append(candidates)
        print(f"{hypothesis.upper()}: selected {len(candidates)} backgrounds")

    combined = (
        pd.concat(frames, ignore_index=True, sort=False)
        if frames
        else pd.DataFrame()
    )
    combined_path = design_dir / "stage2_candidates.csv"
    combined.to_csv(combined_path, index=False)

    counts = (
        combined.groupby(["hypothesis", "selection_reason"], dropna=False)
        .size()
        .rename("n_backgrounds")
        .reset_index()
        if not combined.empty
        else pd.DataFrame(
            columns=["hypothesis", "selection_reason", "n_backgrounds"]
        )
    )
    counts.to_csv(design_dir / "stage2_candidate_counts.csv", index=False)
    return combined_path


def _module(hypothesis: str):
    return importlib.import_module(
        f"experiments.robustness.{hypothesis}_stage1"
    )


def _candidate_ids(
    output_dir: Path,
    hypothesis: str,
    *,
    smoke: bool,
) -> list[str]:
    path = output_dir / "design" / f"{hypothesis}_stage2_candidates.csv"
    candidates = _read_csv(path)
    ids = candidates["background_id"].astype(str).drop_duplicates().tolist()
    return ids[:1] if smoke else ids


def _completed_h1(raw: pd.DataFrame) -> set[tuple[str, float, int]]:
    return {
        (str(row.background_id), float(row.cancel_class1_focal), int(row.seed))
        for row in raw.itertuples(index=False)
    }


def _completed_h2(raw: pd.DataFrame) -> set[tuple[str, float, str, int]]:
    return {
        (
            str(row.background_id),
            float(row.target_loss_share),
            str(row.arm),
            int(row.seed),
        )
        for row in raw.itertuples(index=False)
    }


def _completed_h3(raw: pd.DataFrame) -> set[tuple[str, int, str, int]]:
    return {
        (
            str(row.background_id),
            int(row.noshow_threshold_class1_focal),
            str(row.arm),
            int(row.seed),
        )
        for row in raw.itertuples(index=False)
    }


def _completed_h4(raw: pd.DataFrame) -> set[tuple[str, float, int]]:
    return {
        (
            str(row.background_id),
            float(row.common_balk_high_focal),
            int(row.seed),
        )
        for row in raw.itertuples(index=False)
    }


def _completed_h5(raw: pd.DataFrame) -> set[tuple[str, float, int]]:
    return {
        (
            str(row.background_id),
            float(row.balk_step_class1_focal),
            int(row.seed),
        )
        for row in raw.itertuples(index=False)
    }


def _completed_h6(raw: pd.DataFrame) -> set[tuple[str, int, int]]:
    return {
        (
            str(row.background_id),
            int(row.balk_threshold_class1_focal),
            int(row.seed),
        )
        for row in raw.itertuples(index=False)
    }


def _completed_h7(raw: pd.DataFrame) -> set[tuple[str, float, str, int]]:
    return {
        (
            str(row.background_id),
            float(row.gap_magnitude_focal),
            str(row.gap_location_arm),
            int(row.seed),
        )
        for row in raw.itertuples(index=False)
    }


def _completed_h8(raw: pd.DataFrame) -> set[tuple[str, str, int]]:
    return {
        (str(row.background_id), str(row.h8_arm), int(row.seed))
        for row in raw.itertuples(index=False)
    }


def _completed_h9(raw: pd.DataFrame) -> set[tuple[str, str, int]]:
    return {
        (str(row.background_id), str(row.h9_arm), int(row.seed))
        for row in raw.itertuples(index=False)
    }


COMPLETED_BUILDERS: dict[str, Callable[[pd.DataFrame], set[Any]]] = {
    "h1": _completed_h1,
    "h2": _completed_h2,
    "h3": _completed_h3,
    "h4": _completed_h4,
    "h5": _completed_h5,
    "h6": _completed_h6,
    "h7": _completed_h7,
    "h8": _completed_h8,
    "h9": _completed_h9,
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


def _execute_tasks(
    *,
    tasks: list[dict[str, Any]],
    run_task: Callable[[Mapping[str, Any]], dict[str, Any]],
    raw_path: Path,
    workers: int,
) -> None:
    buffer: list[dict[str, Any]] = []
    flush_every = 100
    if workers <= 1:
        iterator: Iterable[dict[str, Any]] = map(run_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(run_task, tasks, chunksize=4)

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


def _stage2_design_subset(
    *,
    stage1_root: Path,
    output_dir: Path,
    hypothesis: str,
    candidate_ids: Sequence[str],
) -> tuple[pd.DataFrame, Path]:
    source = _read_csv(_design_path(stage1_root, hypothesis))
    subset = source[
        source["background_id"].astype(str).isin(set(candidate_ids))
    ].copy()
    subset["background_id"] = subset["background_id"].astype(str)
    target = (
        output_dir
        / hypothesis
        / "design"
        / f"{hypothesis}_background_scenarios.csv"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(target, index=False)
    return subset, target


def _build_tasks(
    *,
    hypothesis: str,
    module: Any,
    design: pd.DataFrame,
    stage1_root: Path,
    output_dir: Path,
    seeds: Sequence[int],
    completed: set[Any],
    base_config_path: Path,
) -> tuple[list[dict[str, Any]], Path | None]:
    if hypothesis == "h1":
        return list(
            module._task_payloads(
                design, seeds, completed, base_config_path
            )
        ), None

    if hypothesis == "h2":
        calibration_source = _read_csv(
            stage1_root / "h2" / "calibration" / "h2_loss_calibration.csv"
        )
        calibration = calibration_source[
            calibration_source["background_id"].astype(str).isin(
                set(design["background_id"].astype(str))
            )
        ].copy()
        calibration_path = (
            output_dir / "h2" / "calibration" / "h2_loss_calibration.csv"
        )
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration.to_csv(calibration_path, index=False)
        tasks = list(
            module._main_task_payloads(
                design,
                calibration,
                seeds,
                completed,
                base_config_path,
            )
        )
        return tasks, calibration_path

    tasks = list(
        module._task_payloads(
            design, seeds, completed, base_config_path
        )
    )
    return tasks, None


def run_hypothesis(
    *,
    hypothesis: str,
    stage1_root: Path,
    output_dir: Path,
    base_config_path: Path,
    workers: int,
    smoke: bool,
    resume: bool,
) -> None:
    candidate_ids = _candidate_ids(output_dir, hypothesis, smoke=smoke)
    if not candidate_ids:
        print(f"{hypothesis.upper()}: no Stage 2 candidates; skipping run")
        return

    design, _ = _stage2_design_subset(
        stage1_root=stage1_root,
        output_dir=output_dir,
        hypothesis=hypothesis,
        candidate_ids=candidate_ids,
    )
    seeds = tuple(STAGE2_SEEDS[:2]) if smoke else STAGE2_SEEDS
    raw_path = (
        output_dir
        / hypothesis
        / "raw"
        / f"{hypothesis}_stage2_raw.csv"
    )

    if raw_path.exists() and not resume:
        raw_path.unlink()
    raw = _read_csv(raw_path, required=False)
    completed = (
        COMPLETED_BUILDERS[hypothesis](raw)
        if not raw.empty
        else set()
    )

    module = _module(hypothesis)
    tasks, _ = _build_tasks(
        hypothesis=hypothesis,
        module=module,
        design=design,
        stage1_root=stage1_root,
        output_dir=output_dir,
        seeds=seeds,
        completed=completed,
        base_config_path=base_config_path,
    )

    print(f"\n{hypothesis.upper()} Stage 2")
    print(f"Candidate backgrounds: {len(design)}")
    print(f"Confirmation seeds: {len(seeds)}")
    print(f"Rows already completed: {len(completed):,}")
    print(f"Rows to run now: {len(tasks):,}")
    _execute_tasks(
        tasks=tasks,
        run_task=(
            module._run_main_task if hypothesis == "h2" else module._run_task
        ),
        raw_path=raw_path,
        workers=workers,
    )
    print(f"Raw Stage 2 results: {raw_path}")


def run_stage2(
    *,
    stage1_root: Path,
    output_dir: Path,
    base_config_path: Path,
    hypotheses: Sequence[str],
    workers: int,
    smoke: bool,
    resume: bool,
) -> None:
    for hypothesis in hypotheses:
        run_hypothesis(
            hypothesis=hypothesis,
            stage1_root=stage1_root,
            output_dir=output_dir,
            base_config_path=base_config_path,
            workers=workers,
            smoke=smoke,
            resume=resume,
        )


def _promote_summary_names(hypothesis_dir: Path, hypothesis: str) -> None:
    summary_dir = hypothesis_dir / "summary"
    old_summary = summary_dir / f"{hypothesis}_stage1_summary.md"
    new_summary = summary_dir / f"{hypothesis}_stage2_summary.md"
    if old_summary.exists():
        text = old_summary.read_text(encoding="utf-8")
        text = text.replace(
            "Stage 1 Robustness Summary",
            "Stage 2 Confirmation Summary",
        )
        text = text.replace("Stage 1", "Stage 2")
        text = text.replace(
            "exported for Stage 2 confirmation with 100 new seeds",
            "retained as unresolved after Stage 2",
        )
        text = text.replace(
            "exported for Stage 2 confirmation",
            "retained as unresolved after Stage 2",
        )
        new_summary.write_text(text, encoding="utf-8")
        old_summary.unlink()

    old_candidates = summary_dir / f"{hypothesis}_stage2_candidates.csv"
    unresolved = summary_dir / f"{hypothesis}_unresolved_after_stage2.csv"
    if old_candidates.exists():
        if unresolved.exists():
            unresolved.unlink()
        old_candidates.rename(unresolved)


def classify_hypothesis(
    *,
    hypothesis: str,
    output_dir: Path,
) -> None:
    hypothesis_dir = output_dir / hypothesis
    raw_path = hypothesis_dir / "raw" / f"{hypothesis}_stage2_raw.csv"
    design_path = (
        hypothesis_dir
        / "design"
        / f"{hypothesis}_background_scenarios.csv"
    )
    if not raw_path.exists() or not design_path.exists():
        print(f"{hypothesis.upper()}: no Stage 2 run found; skipping classification")
        return

    module = _module(hypothesis)
    if hypothesis == "h1":
        module.classify_stage1(
            raw_path=raw_path,
            output_dir=hypothesis_dir,
        )
    elif hypothesis == "h2":
        module.classify_stage1(
            raw_path=raw_path,
            calibration_path=(
                hypothesis_dir
                / "calibration"
                / "h2_loss_calibration.csv"
            ),
            output_dir=hypothesis_dir,
        )
    else:
        module.classify_stage1(
            design_path=design_path,
            raw_path=raw_path,
            output_dir=hypothesis_dir,
        )

    effects_path = (
        hypothesis_dir
        / "summary"
        / f"{hypothesis}_scenario_effects.csv"
    )
    if effects_path.exists():
        effects = pd.read_csv(effects_path)
        effects.insert(0, "stage", 2)
        effects.to_csv(effects_path, index=False)

    target_effects = (
        hypothesis_dir / "summary" / f"{hypothesis}_target_effects.csv"
    )
    if target_effects.exists():
        target = pd.read_csv(target_effects)
        target.insert(0, "stage", 2)
        target.to_csv(target_effects, index=False)

    _promote_summary_names(hypothesis_dir, hypothesis)


def classify_stage2(*, output_dir: Path, hypotheses: Sequence[str]) -> None:
    for hypothesis in hypotheses:
        classify_hypothesis(hypothesis=hypothesis, output_dir=output_dir)


def _transition_label(stage1: str, stage2: str) -> str:
    if not stage2:
        return "not_rerun"
    return f"{stage1}_to_{stage2}"


def summarize_stage2(
    *,
    stage1_root: Path,
    output_dir: Path,
    hypotheses: Sequence[str],
) -> Path:
    confirmation_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    for hypothesis in hypotheses:
        stage1 = _read_csv(_scenario_effects_path(stage1_root, hypothesis))
        stage1["background_id"] = stage1["background_id"].astype(str)
        candidates = _read_csv(
            output_dir
            / "design"
            / f"{hypothesis}_stage2_candidates.csv",
            required=False,
        )
        reason_map = (
            candidates.set_index(candidates["background_id"].astype(str))[
                "selection_reason"
            ].to_dict()
            if not candidates.empty
            else {}
        )

        stage2_path = (
            output_dir
            / hypothesis
            / "summary"
            / f"{hypothesis}_scenario_effects.csv"
        )
        stage2 = _read_csv(stage2_path, required=False)
        stage2_map: dict[str, str] = {}
        if not stage2.empty:
            stage2_map = (
                stage2.assign(
                    background_id=stage2["background_id"].astype(str)
                )
                .set_index("background_id")["classification"]
                .astype(str)
                .to_dict()
            )

        for row in stage1.to_dict(orient="records"):
            background_id = str(row["background_id"])
            stage1_classification = str(row["classification"])
            selected = background_id in reason_map
            stage2_classification = stage2_map.get(background_id, "")
            final_classification = (
                stage2_classification
                if stage2_classification
                else stage1_classification
            )
            final_rows.append(
                {
                    "hypothesis": hypothesis.upper(),
                    "background_id": background_id,
                    "selected_for_stage2": selected,
                    "selection_reason": reason_map.get(background_id, ""),
                    "stage1_classification": stage1_classification,
                    "stage2_classification": stage2_classification,
                    "final_classification": final_classification,
                    "stage_used": 2 if stage2_classification else 1,
                    "source_scenario_ids": row.get("source_scenario_ids", ""),
                    "scenario_type": row.get("scenario_type", ""),
                    "rho": row.get("rho", math.nan),
                    "class1_share": row.get("class1_share", math.nan),
                    "slots_per_day": row.get("slots_per_day", math.nan),
                    "horizon_class1": row.get("horizon_class1", math.nan),
                    "horizon_class2": row.get("horizon_class2", math.nan),
                }
            )
            if selected:
                confirmation_rows.append(
                    {
                        "hypothesis": hypothesis.upper(),
                        "background_id": background_id,
                        "selection_reason": reason_map[background_id],
                        "stage1_classification": stage1_classification,
                        "stage2_classification": stage2_classification,
                        "transition": _transition_label(
                            stage1_classification,
                            stage2_classification,
                        ),
                        "confirmed_reversal": bool(
                            stage2_classification == "reversed"
                        ),
                        "resolved_after_stage2": bool(
                            stage2_classification
                            and stage2_classification != "inconclusive"
                        ),
                    }
                )

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    confirmation_columns = [
        "hypothesis",
        "background_id",
        "selection_reason",
        "stage1_classification",
        "stage2_classification",
        "transition",
        "confirmed_reversal",
        "resolved_after_stage2",
    ]
    confirmations = pd.DataFrame(
        confirmation_rows,
        columns=confirmation_columns,
    )
    final_status = pd.DataFrame(final_rows)
    confirmations.to_csv(
        final_dir / "stage2_confirmation_results.csv",
        index=False,
    )
    final_status.to_csv(
        final_dir / "stage2_final_status.csv",
        index=False,
    )

    confirmed_reversals = confirmations[
        confirmations["confirmed_reversal"].fillna(False)
    ].copy()
    confirmed_reversals.to_csv(
        final_dir / "confirmed_reversals.csv",
        index=False,
    )

    unresolved = confirmations[
        confirmations["stage2_classification"].astype(str).isin(
            ["", "inconclusive"]
        )
    ].copy()
    unresolved.to_csv(
        final_dir / "unresolved_after_stage2.csv",
        index=False,
    )

    active = final_status[
        final_status["final_classification"].astype(str).ne("inactive")
    ]
    counts = (
        active.groupby(["hypothesis", "final_classification"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    counts.to_csv(final_dir / "final_active_classification_counts.csv", index=False)

    transitions = (
        confirmations.groupby(["hypothesis", "transition"])
        .size()
        .rename("n_backgrounds")
        .reset_index()
        if not confirmations.empty
        else pd.DataFrame(columns=["hypothesis", "transition", "n_backgrounds"])
    )
    transitions.to_csv(final_dir / "stage2_transition_counts.csv", index=False)

    inactive_count = int(
        (final_status["final_classification"].astype(str) == "inactive").sum()
    )
    lines = [
        "# Stage 2 Robustness Confirmation Summary",
        "",
        (
            "Stage 2 reruns every Stage 1 reversal and only those "
            "inconclusive cases whose point estimates already meet all "
            "practical-effect criteria but whose confidence intervals are "
            "not decisive."
        ),
        "",
        "## Final active classifications",
        "",
        counts.to_markdown(index=False) if not counts.empty else "No active results.",
        "",
        "## Stage 2 transitions",
        "",
        (
            transitions.to_markdown(index=False)
            if not transitions.empty
            else "No Stage 2 confirmations were available."
        ),
        "",
        (
            f"Inactive configurations excluded from the substantive table: "
            f"**{inactive_count}**."
        ),
        "",
        (
            "Individual scenarios are retained in the CSV outputs but are not "
            "listed one by one in this summary."
        ),
    ]
    summary_path = final_dir / "stage2_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _parse_hypotheses(value: str) -> tuple[str, ...]:
    if value.strip().lower() in {"all", "h1-h9"}:
        return HYPOTHESES
    parsed: list[str] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item.isdigit():
            item = f"h{item}"
        if item not in HYPOTHESES:
            raise argparse.ArgumentTypeError(
                f"Unknown hypothesis {item!r}; use h1,...,h9 or all."
            )
        parsed.append(item)
    if not parsed:
        raise argparse.ArgumentTypeError("No hypotheses selected.")
    return tuple(dict.fromkeys(parsed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["select", "run", "classify", "summarize", "all"],
    )
    parser.add_argument(
        "--hypotheses",
        type=_parse_hypotheses,
        default=HYPOTHESES,
        help="Comma-separated list such as h1,h2,h7, or all.",
    )
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hypotheses = tuple(args.hypotheses)

    if args.command in {"select", "all"}:
        select_candidates(
            stage1_root=args.stage1_root,
            output_dir=args.output_dir,
            hypotheses=hypotheses,
        )

    if args.command in {"run", "all"}:
        missing = [
            hypothesis
            for hypothesis in hypotheses
            if not (
                args.output_dir
                / "design"
                / f"{hypothesis}_stage2_candidates.csv"
            ).exists()
        ]
        if missing:
            select_candidates(
                stage1_root=args.stage1_root,
                output_dir=args.output_dir,
                hypotheses=missing,
            )
        run_stage2(
            stage1_root=args.stage1_root,
            output_dir=args.output_dir,
            base_config_path=args.base_config,
            hypotheses=hypotheses,
            workers=args.workers,
            smoke=args.smoke,
            resume=not args.no_resume,
        )

    if args.command in {"classify", "all"}:
        classify_stage2(output_dir=args.output_dir, hypotheses=hypotheses)

    if args.command in {"summarize", "all"}:
        summary = summarize_stage2(
            stage1_root=args.stage1_root,
            output_dir=args.output_dir,
            hypotheses=hypotheses,
        )
        print(f"Combined Stage 2 summary: {summary}")


if __name__ == "__main__":
    main()
