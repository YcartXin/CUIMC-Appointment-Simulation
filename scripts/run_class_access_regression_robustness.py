#!/usr/bin/env python3
"""
Class-access regression robustness experiment for the pooled FCFS model.

Purpose
-------
Test whether between-class differences in delay-sensitive balking are more
strongly associated with the Class 1 - Class 2 served-rate gap than analogous
between-class differences in no-show behavior, while varying demand, class mix,
cancellation, behavior probabilities, and behavior timing.

Key design constraint
---------------------
For each class:
    no-show threshold <= balking threshold

The timing variables used in regression are therefore:
    no-show threshold
    spacing = balking threshold - no-show threshold
rather than treating the two thresholds as independent.

Default experiment
------------------
- 1,000 Latin-hypercube backgrounds.
- 5 independent simulation seeds per background.
- Basic pooled FCFS only: no reservation, no standby/requeue, no policy.
- Simulation span inherited from configs/baseline.yaml.
- Background-level regression outcome:
      Class 1 served rate - Class 2 served rate.
- Main inference:
    * HC3-robust OLS in raw units.
    * HC3-robust standardized OLS.
    * Direct Wald/linear-contrast test of:
          beta(balking probability gap) = beta(no-show probability gap)
    * Demand interaction and demand-tercile robustness analyses.
    * Repeated held-out R^2 drop-one-family robustness check.

The script is sharding- and resume-friendly for cluster execution.

Examples
--------
Create the design:
    python scripts/run_class_access_regression_robustness.py --mode design

Smoke test:
    python scripts/run_class_access_regression_robustness.py --mode all --smoke --workers 2

Run shard 3 of 20:
    python scripts/run_class_access_regression_robustness.py \
        --mode run --shard-count 20 --shard-index 3 --workers 4

After all shards finish:
    python scripts/run_class_access_regression_robustness.py --mode analyze

A more restricted sensitivity design can be generated with the CLI range
arguments, e.g. --balk-prob-max, --no-show-prob-max, --class-share-min, etc.
Do not label such a profile "clinically plausible" until its ranges are agreed
with the research team.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# Repo imports
# -----------------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from analysis.metrics import outcome_rates_from_result, result_metrics_from_result  # noqa: E402
from simulation.config_loader import load_config  # noqa: E402
from simulation.engine import ClinicAppointmentSimulation  # noqa: E402
from simulation.model import ThresholdRule  # noqa: E402

BASE_CONFIG = load_config(REPO_DIR / "configs" / "baseline.yaml")

DEFAULT_OUTPUT_DIR = REPO_DIR / "docs" / "reports" / "class_access_regression_robustness"
DEFAULT_BACKGROUND_COUNT = 1000
DEFAULT_SEEDS_PER_BACKGROUND = 5
DEFAULT_DESIGN_SEED = 20260831
DEFAULT_SIM_SEED_BASE = 8_310_000
DEFAULT_COV_TYPE = "HC3"
ALPHA = 0.05

# Behavior colors are deliberately not the presentation's Class 1 / Class 2
# colors. These figures compare mechanisms, not patient classes.
BALK_COLOR = "#4C78A8"
NOSHOW_COLOR = "#E45756"
NEUTRAL_COLOR = "#555555"

MAIN_FEATURES = [
    "lambda_total",
    "class_1_share",
    "balk_step_mean",
    "balk_step_gap_c1_minus_c2",
    "no_show_step_mean",
    "no_show_step_gap_c1_minus_c2",
    "cancel_prob_mean",
    "cancel_prob_gap_c1_minus_c2",
    "no_show_threshold_mean",
    "no_show_threshold_gap_c1_minus_c2",
    "threshold_spacing_mean",
    "threshold_spacing_gap_c1_minus_c2",
]

TARGET = "access_advantage_class_1"

BALK_FAMILY = [
    "balk_step_mean",
    "balk_step_gap_c1_minus_c2",
    "threshold_spacing_mean",
    "threshold_spacing_gap_c1_minus_c2",
]
NOSHOW_FAMILY = [
    "no_show_step_mean",
    "no_show_step_gap_c1_minus_c2",
    "no_show_threshold_mean",
    "no_show_threshold_gap_c1_minus_c2",
]

FEATURE_LABELS = {
    "lambda_total": "total arrival rate",
    "class_1_share": "Class 1 arrival share",
    "balk_step_mean": "mean balking probability",
    "balk_step_gap_c1_minus_c2": "balking probability gap (C1-C2)",
    "no_show_step_mean": "mean no-show probability",
    "no_show_step_gap_c1_minus_c2": "no-show probability gap (C1-C2)",
    "cancel_prob_mean": "mean cancellation probability",
    "cancel_prob_gap_c1_minus_c2": "cancellation probability gap (C1-C2)",
    "no_show_threshold_mean": "mean no-show threshold",
    "no_show_threshold_gap_c1_minus_c2": "no-show threshold gap (C1-C2)",
    "threshold_spacing_mean": "mean NS-to-balking spacing",
    "threshold_spacing_gap_c1_minus_c2": "NS-to-balking spacing gap (C1-C2)",
}


def dataclass_field_names(obj) -> set[str]:
    return {f.name for f in fields(obj)}


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> dict:
    def run_git(args: list[str]) -> str | None:
        completed = subprocess.run(
            ["git", *args], cwd=REPO_DIR, text=True, capture_output=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
    }


# -----------------------------------------------------------------------------
# Design generation
# -----------------------------------------------------------------------------
def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Simple dependency-free Latin hypercube on [0, 1)^d."""
    base = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        rng.shuffle(base[:, j])
    return base


def scale_unit(u: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + u * (high - low)


def feasible_threshold_pairs(
    horizon_days: int,
    no_show_threshold_min: int,
    no_show_threshold_max: int | None,
    min_spacing: int,
    max_spacing: int | None,
) -> list[tuple[int, int]]:
    """
    Return all feasible (no_show_threshold, balking_threshold) pairs satisfying:
        no_show_threshold <= balking_threshold < horizon_days
    plus any user-specified threshold/spacing restrictions.
    """
    max_ns = horizon_days - 1 if no_show_threshold_max is None else no_show_threshold_max
    max_sp = horizon_days - 1 if max_spacing is None else max_spacing
    pairs: list[tuple[int, int]] = []
    for ns in range(no_show_threshold_min, min(max_ns, horizon_days - 1) + 1):
        for balk in range(ns, horizon_days):
            spacing = balk - ns
            if spacing < min_spacing or spacing > max_sp:
                continue
            pairs.append((ns, balk))
    if not pairs:
        raise ValueError("No feasible threshold pairs under the requested restrictions.")
    return pairs


def map_pair(u: float, pairs: list[tuple[int, int]]) -> tuple[int, int]:
    index = min(int(math.floor(float(u) * len(pairs))), len(pairs) - 1)
    return pairs[index]


def add_mean_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for prefix in ["balk_step", "no_show_step", "cancel_prob", "no_show_threshold", "threshold_spacing"]:
        c1 = df[f"class_1_{prefix}"]
        c2 = df[f"class_2_{prefix}"]
        df[f"{prefix}_mean"] = (c1 + c2) / 2.0
        df[f"{prefix}_gap_c1_minus_c2"] = c1 - c2
    df["rho_nominal"] = df["lambda_total"] / float(BASE_CONFIG.slots_per_day)
    return df


def generate_design(args: argparse.Namespace) -> pd.DataFrame:
    n = args.n_backgrounds
    rng = np.random.default_rng(args.design_seed)
    lhs = latin_hypercube(n, 10, rng)

    lambda_total_base = sum(p.lambda_per_day for p in BASE_CONFIG.classes.values())
    lambda_min = args.lambda_min_mult * lambda_total_base
    lambda_max = args.lambda_max_mult * lambda_total_base

    pairs = feasible_threshold_pairs(
        horizon_days=BASE_CONFIG.horizon_days,
        no_show_threshold_min=args.no_show_threshold_min,
        no_show_threshold_max=args.no_show_threshold_max,
        min_spacing=args.min_threshold_spacing,
        max_spacing=args.max_threshold_spacing,
    )

    rows = []
    for i in range(n):
        c1_ns_t, c1_balk_t = map_pair(lhs[i, 8], pairs)
        c2_ns_t, c2_balk_t = map_pair(lhs[i, 9], pairs)
        rows.append(
            {
                "background_id": i,
                "lambda_total": float(scale_unit(lhs[i, 0], lambda_min, lambda_max)),
                "class_1_share": float(scale_unit(lhs[i, 1], args.class_share_min, args.class_share_max)),
                "class_1_balk_step": float(scale_unit(lhs[i, 2], args.balk_prob_min, args.balk_prob_max)),
                "class_2_balk_step": float(scale_unit(lhs[i, 3], args.balk_prob_min, args.balk_prob_max)),
                "class_1_no_show_step": float(scale_unit(lhs[i, 4], args.no_show_prob_min, args.no_show_prob_max)),
                "class_2_no_show_step": float(scale_unit(lhs[i, 5], args.no_show_prob_min, args.no_show_prob_max)),
                "class_1_cancel_prob": float(scale_unit(lhs[i, 6], args.cancel_prob_min, args.cancel_prob_max)),
                "class_2_cancel_prob": float(scale_unit(lhs[i, 7], args.cancel_prob_min, args.cancel_prob_max)),
                "class_1_no_show_threshold": int(c1_ns_t),
                "class_1_balk_threshold": int(c1_balk_t),
                "class_1_threshold_spacing": int(c1_balk_t - c1_ns_t),
                "class_2_no_show_threshold": int(c2_ns_t),
                "class_2_balk_threshold": int(c2_balk_t),
                "class_2_threshold_spacing": int(c2_balk_t - c2_ns_t),
            }
        )

    design = add_mean_gap_features(pd.DataFrame(rows))
    validate_design(design, args)
    return design


def validate_design(design: pd.DataFrame, args: argparse.Namespace) -> None:
    if len(design) != args.n_backgrounds:
        raise AssertionError("Unexpected number of backgrounds.")
    for class_id in [1, 2]:
        ns = design[f"class_{class_id}_no_show_threshold"]
        balk = design[f"class_{class_id}_balk_threshold"]
        if not (ns <= balk).all():
            raise AssertionError(f"Found Class {class_id} background with no-show threshold after balking threshold.")
        spacing = design[f"class_{class_id}_threshold_spacing"]
        if not (spacing == balk - ns).all():
            raise AssertionError("Threshold spacing is inconsistent.")
    if not design["background_id"].is_unique:
        raise AssertionError("background_id is not unique.")


# -----------------------------------------------------------------------------
# Simulation execution
# -----------------------------------------------------------------------------
def update_class_params(base_params, *, lambda_per_day, cancel_prob, balk_step, balk_threshold, no_show_step, no_show_threshold):
    changes = {
        "lambda_per_day": float(lambda_per_day),
        "cancel_prob": float(cancel_prob),
        "balk_prob": ThresholdRule(threshold=int(balk_threshold), low=0.0, high=float(balk_step)),
        "no_show_prob": ThresholdRule(threshold=int(no_show_threshold), low=0.0, high=float(no_show_step)),
    }
    names = dataclass_field_names(base_params)
    if "standby_prob" in names:
        changes["standby_prob"] = 0.0
    if "max_standby_days" in names:
        changes["max_standby_days"] = None
    if "standby_eligible_after_days" in names:
        changes["standby_eligible_after_days"] = None
    return replace(base_params, **changes)


def config_from_background(row: dict, seed: int):
    c1_lambda = row["lambda_total"] * row["class_1_share"]
    c2_lambda = row["lambda_total"] * (1.0 - row["class_1_share"])

    classes = dict(BASE_CONFIG.classes)
    classes[1] = update_class_params(
        BASE_CONFIG.classes[1],
        lambda_per_day=c1_lambda,
        cancel_prob=row["class_1_cancel_prob"],
        balk_step=row["class_1_balk_step"],
        balk_threshold=row["class_1_balk_threshold"],
        no_show_step=row["class_1_no_show_step"],
        no_show_threshold=row["class_1_no_show_threshold"],
    )
    classes[2] = update_class_params(
        BASE_CONFIG.classes[2],
        lambda_per_day=c2_lambda,
        cancel_prob=row["class_2_cancel_prob"],
        balk_step=row["class_2_balk_step"],
        balk_threshold=row["class_2_balk_threshold"],
        no_show_step=row["class_2_no_show_step"],
        no_show_threshold=row["class_2_no_show_threshold"],
    )

    changes = {"classes": classes, "seed": int(seed)}
    config_fields = dataclass_field_names(BASE_CONFIG)
    # Force the no-policy pooled-FCFS setting even if the baseline config later
    # gains optional policy fields.
    if "reserved_class_id" in config_fields:
        changes["reserved_class_id"] = None
    if "reserved_slots_per_day" in config_fields:
        changes["reserved_slots_per_day"] = 0
    if "reserved_window_days" in config_fields:
        changes["reserved_window_days"] = None
    if "same_day_cancellation_enabled" in config_fields:
        changes["same_day_cancellation_enabled"] = False
    if "release_unused_reservation_same_day" in config_fields:
        changes["release_unused_reservation_same_day"] = False
    return replace(BASE_CONFIG, **changes)


def simulation_seed(background_id: int, replicate: int, seeds_per_background: int, seed_base: int) -> int:
    return int(seed_base + background_id * seeds_per_background + replicate)


def run_one_job(job: tuple[dict, int, int]) -> dict:
    row, replicate, seed = job
    config = config_from_background(row, seed=seed)
    result = ClinicAppointmentSimulation(config).run()
    return {
        "background_id": int(row["background_id"]),
        "replicate": int(replicate),
        "seed": int(seed),
        **result_metrics_from_result(result),
        **outcome_rates_from_result(result),
    }


def run_shard(args: argparse.Namespace, design: pd.DataFrame, raw_dir: Path) -> Path:
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("shard-index must satisfy 0 <= shard-index < shard-count")

    shard_backgrounds = design[design["background_id"] % args.shard_count == args.shard_index]
    out_path = raw_dir / f"seed_outcomes_shard_{args.shard_index:03d}_of_{args.shard_count:03d}.csv"

    existing = pd.DataFrame()
    done: set[tuple[int, int]] = set()
    if out_path.exists() and not args.no_resume:
        existing = pd.read_csv(out_path)
        if not existing.empty:
            done = set(zip(existing["background_id"].astype(int), existing["replicate"].astype(int)))

    jobs: list[tuple[dict, int, int]] = []
    for row in shard_backgrounds.to_dict(orient="records"):
        bid = int(row["background_id"])
        for replicate in range(args.seeds_per_background):
            key = (bid, replicate)
            if key in done:
                continue
            seed = simulation_seed(bid, replicate, args.seeds_per_background, args.sim_seed_base)
            jobs.append((row, replicate, seed))

    print(
        f"Shard {args.shard_index}/{args.shard_count}: "
        f"{len(shard_backgrounds)} backgrounds, {len(jobs)} missing seed-runs."
    )

    rows = existing.to_dict(orient="records") if not existing.empty else []
    completed_since_checkpoint = 0

    def checkpoint() -> None:
        if rows:
            frame = pd.DataFrame(rows).sort_values(["background_id", "replicate"])
            atomic_write_csv(frame, out_path)

    if args.workers <= 1:
        iterator: Iterable[dict] = map(run_one_job, jobs)
        for result in iterator:
            rows.append(result)
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= args.checkpoint_every:
                checkpoint()
                completed_since_checkpoint = 0
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(run_one_job, jobs, chunksize=1):
                rows.append(result)
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= args.checkpoint_every:
                    checkpoint()
                    completed_since_checkpoint = 0

    checkpoint()
    print(f"Wrote {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Regression helpers
# -----------------------------------------------------------------------------
def fit_raw_ols(data: pd.DataFrame, features: list[str], target: str = TARGET):
    x = sm.add_constant(data[features].astype(float), has_constant="add")
    return sm.OLS(data[target].astype(float), x).fit(cov_type=DEFAULT_COV_TYPE)


def standardize_frame(data: pd.DataFrame, features: list[str], target: str = TARGET):
    x = data[features].astype(float)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=0).replace(0.0, 1.0)
    y = data[target].astype(float)
    y_mean = float(y.mean())
    y_std = float(y.std(ddof=0)) or 1.0
    xz = (x - x_mean) / x_std
    yz = (y - y_mean) / y_std
    return xz, yz, x_mean, x_std, y_mean, y_std


def fit_standardized_ols(data: pd.DataFrame, features: list[str], target: str = TARGET):
    xz, yz, *_ = standardize_frame(data, features, target)
    x = sm.add_constant(xz, has_constant="add")
    return sm.OLS(yz, x).fit(cov_type=DEFAULT_COV_TYPE)


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(float(z)) / math.sqrt(2.0))


def linear_contrast(model, weights: dict[str, float], alpha: float = ALPHA) -> dict:
    names = list(model.params.index)
    vector = np.zeros(len(names), dtype=float)
    for name, weight in weights.items():
        if name not in names:
            raise KeyError(f"Feature {name!r} not in model.")
        vector[names.index(name)] = float(weight)
    params = model.params.to_numpy(dtype=float)
    cov = np.asarray(model.cov_params(), dtype=float)
    effect = float(vector @ params)
    variance = float(vector @ cov @ vector)
    se = math.sqrt(max(variance, 0.0))
    z = effect / se if se > 0 else float("nan")
    p_two = normal_two_sided_p(z) if math.isfinite(z) else float("nan")
    critical = 1.959963984540054
    return {
        "estimate": effect,
        "standard_error": se,
        "z": z,
        "p_value_two_sided": p_two,
        "ci_low_95": effect - critical * se,
        "ci_high_95": effect + critical * se,
    }


def directional_p_more_negative(contrast: dict) -> float:
    """One-sided p for H1: beta_balk - beta_noshow < 0."""
    z = contrast["z"]
    if not math.isfinite(z):
        return float("nan")
    # Phi(z) using erfc, because H1 is z < 0.
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def coefficient_table(model, model_name: str) -> pd.DataFrame:
    ci = model.conf_int(alpha=ALPHA)
    if not isinstance(ci, pd.DataFrame):
        ci = pd.DataFrame(ci, index=model.params.index, columns=[0, 1])
    rows = []
    for name in model.params.index:
        rows.append(
            {
                "model": model_name,
                "feature": name,
                "feature_label": "intercept" if name == "const" else FEATURE_LABELS.get(name, name),
                "coefficient": float(model.params.loc[name]),
                "hc3_standard_error": float(model.bse.loc[name]),
                "ci_low_95_hc3": float(ci.loc[name].iloc[0]),
                "ci_high_95_hc3": float(ci.loc[name].iloc[1]),
                "p_value_hc3": float(model.pvalues.loc[name]),
            }
        )
    return pd.DataFrame(rows)


def heldout_r2(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> float:
    x_train = sm.add_constant(train[features].astype(float), has_constant="add")
    model = sm.OLS(train[TARGET].astype(float), x_train).fit()
    x_test = sm.add_constant(test[features].astype(float), has_constant="add")
    pred = np.asarray(model.predict(x_test), dtype=float)
    y = test[TARGET].to_numpy(dtype=float)
    ss_res = float(np.square(y - pred).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def repeated_drop_family_r2(data: pd.DataFrame, repetitions: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    n = len(data)
    for rep in range(repetitions):
        rng = np.random.default_rng(seed + rep)
        idx = rng.permutation(n)
        cut = int(0.8 * n)
        train = data.iloc[idx[:cut]]
        test = data.iloc[idx[cut:]]
        full = heldout_r2(train, test, MAIN_FEATURES)
        no_balk = heldout_r2(train, test, [x for x in MAIN_FEATURES if x not in BALK_FAMILY])
        no_ns = heldout_r2(train, test, [x for x in MAIN_FEATURES if x not in NOSHOW_FAMILY])
        rows.append(
            {
                "split": rep,
                "full_test_r2": full,
                "without_balking_family_test_r2": no_balk,
                "without_no_show_family_test_r2": no_ns,
                "drop_r2_balking_family": full - no_balk,
                "drop_r2_no_show_family": full - no_ns,
                "drop_difference_balk_minus_noshow": (full - no_balk) - (full - no_ns),
            }
        )
    split_df = pd.DataFrame(rows)
    summary_rows = []
    for col in ["drop_r2_balking_family", "drop_r2_no_show_family", "drop_difference_balk_minus_noshow"]:
        values = split_df[col].dropna().to_numpy()
        summary_rows.append(
            {
                "metric": col,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p025": float(np.quantile(values, 0.025)),
                "p975": float(np.quantile(values, 0.975)),
                "share_positive": float(np.mean(values > 0)),
            }
        )
    return split_df, pd.DataFrame(summary_rows)


# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
def collect_seed_outcomes(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("seed_outcomes_shard_*_of_*.csv"))
    if not files:
        raise FileNotFoundError(f"No shard outputs found in {raw_dir}")
    frames = [pd.read_csv(path) for path in files]
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(["background_id", "replicate"], keep="last")
    return data.sort_values(["background_id", "replicate"]).reset_index(drop=True)


def validate_seed_outcomes(seed_df: pd.DataFrame, design: pd.DataFrame, seeds_per_background: int, allow_incomplete: bool) -> None:
    expected = len(design) * seeds_per_background
    actual = len(seed_df)
    counts = seed_df.groupby("background_id").size()
    bad = counts[counts != seeds_per_background]
    missing_backgrounds = sorted(set(design.background_id.astype(int)) - set(counts.index.astype(int)))
    if (actual != expected or len(bad) or missing_backgrounds) and not allow_incomplete:
        raise RuntimeError(
            f"Incomplete simulation outputs: expected {expected} seed-runs, found {actual}; "
            f"backgrounds with wrong replicate count={len(bad)}, missing backgrounds={len(missing_backgrounds)}."
        )


def aggregate_backgrounds(seed_df: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in seed_df.select_dtypes(include=[np.number]).columns
        if c not in {"background_id", "replicate", "seed"}
    ]
    agg = seed_df.groupby("background_id", as_index=False)[metric_cols].mean()
    data = design.merge(agg, on="background_id", how="inner", validate="one_to_one")
    return data


def demand_stratified_tests(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = data.copy()
    work["demand_band"] = pd.qcut(work["lambda_total"], q=3, labels=["low", "moderate", "high"])
    coef_rows = []
    contrast_rows = []
    for band, sub in work.groupby("demand_band", observed=True):
        model = fit_raw_ols(sub, MAIN_FEATURES)
        coefs = coefficient_table(model, f"demand_band_{band}")
        wanted = coefs[coefs["feature"].isin(["balk_step_gap_c1_minus_c2", "no_show_step_gap_c1_minus_c2"])].copy()
        wanted["demand_band"] = str(band)
        wanted["n_backgrounds"] = len(sub)
        wanted["lambda_min"] = sub["lambda_total"].min()
        wanted["lambda_max"] = sub["lambda_total"].max()
        coef_rows.append(wanted)

        contrast = linear_contrast(
            model,
            {
                "balk_step_gap_c1_minus_c2": 1.0,
                "no_show_step_gap_c1_minus_c2": -1.0,
            },
        )
        contrast_rows.append(
            {
                "demand_band": str(band),
                "n_backgrounds": len(sub),
                "lambda_min": float(sub["lambda_total"].min()),
                "lambda_max": float(sub["lambda_total"].max()),
                **contrast,
                "p_value_one_sided_balk_more_negative": directional_p_more_negative(contrast),
            }
        )
    return pd.concat(coef_rows, ignore_index=True), pd.DataFrame(contrast_rows)


def interaction_analysis(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    work = data.copy()
    demand_center = float(work["lambda_total"].mean())
    work["lambda_total_centered"] = work["lambda_total"] - demand_center
    work["balk_gap_x_demand"] = work["balk_step_gap_c1_minus_c2"] * work["lambda_total_centered"]
    work["no_show_gap_x_demand"] = work["no_show_step_gap_c1_minus_c2"] * work["lambda_total_centered"]

    features = [x for x in MAIN_FEATURES if x != "lambda_total"] + [
        "lambda_total_centered",
        "balk_gap_x_demand",
        "no_show_gap_x_demand",
    ]
    model = fit_raw_ols(work, features)
    coef_df = coefficient_table(model, "demand_interaction_raw")

    rows = []
    for label, q in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        demand = float(work["lambda_total"].quantile(q))
        centered = demand - demand_center
        balk = linear_contrast(
            model,
            {
                "balk_step_gap_c1_minus_c2": 1.0,
                "balk_gap_x_demand": centered,
            },
        )
        noshow = linear_contrast(
            model,
            {
                "no_show_step_gap_c1_minus_c2": 1.0,
                "no_show_gap_x_demand": centered,
            },
        )
        diff = linear_contrast(
            model,
            {
                "balk_step_gap_c1_minus_c2": 1.0,
                "no_show_step_gap_c1_minus_c2": -1.0,
                "balk_gap_x_demand": centered,
                "no_show_gap_x_demand": -centered,
            },
        )
        rows.append(
            {
                "demand_point": label,
                "lambda_total": demand,
                "balk_effect": balk["estimate"],
                "balk_ci_low_95": balk["ci_low_95"],
                "balk_ci_high_95": balk["ci_high_95"],
                "no_show_effect": noshow["estimate"],
                "no_show_ci_low_95": noshow["ci_low_95"],
                "no_show_ci_high_95": noshow["ci_high_95"],
                "balk_minus_no_show": diff["estimate"],
                "difference_ci_low_95": diff["ci_low_95"],
                "difference_ci_high_95": diff["ci_high_95"],
                "difference_p_two_sided": diff["p_value_two_sided"],
                "difference_p_one_sided_balk_more_negative": directional_p_more_negative(diff),
            }
        )
    return coef_df, pd.DataFrame(rows), model


def plot_probability_gap_coefficients(raw_model, figure_dir: Path) -> None:
    names = ["balk_step_gap_c1_minus_c2", "no_show_step_gap_c1_minus_c2"]
    labels = ["Balking", "No-show"]
    colors = [BALK_COLOR, NOSHOW_COLOR]
    estimates = []
    lows = []
    highs = []
    for name in names:
        c = linear_contrast(raw_model, {name: 0.10})  # effect per +10 percentage points
        estimates.append(c["estimate"])
        lows.append(c["ci_low_95"])
        highs.append(c["ci_high_95"])

    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    ax.bar(x, estimates, color=colors, width=0.58)
    yerr = np.vstack([np.array(estimates) - np.array(lows), np.array(highs) - np.array(estimates)])
    ax.errorbar(x, estimates, yerr=yerr, fmt="none", ecolor=NEUTRAL_COLOR, capsize=4, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Change in C1-C2 served-rate gap")
    ax.set_title("Effect of a +10 pp between-class behavior-probability gap")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(figure_dir / "probability_gap_effect_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_interaction_marginal_effects(marginal_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    x = marginal_df["lambda_total"].to_numpy()
    for effect, low, high, label, color in [
        ("balk_effect", "balk_ci_low_95", "balk_ci_high_95", "Balking gap", BALK_COLOR),
        ("no_show_effect", "no_show_ci_low_95", "no_show_ci_high_95", "No-show gap", NOSHOW_COLOR),
    ]:
        y = 0.10 * marginal_df[effect].to_numpy()
        lo = 0.10 * marginal_df[low].to_numpy()
        hi = 0.10 * marginal_df[high].to_numpy()
        ax.plot(x, y, marker="o", linewidth=2.2, label=label, color=color)
        ax.fill_between(x, lo, hi, alpha=0.15, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Total arrivals per day")
    ax.set_ylabel("Effect on C1-C2 served-rate gap per +10 pp behavior gap")
    ax.set_title("Does the relative balking effect change with demand?")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(figure_dir / "demand_interaction_marginal_effects.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    data: pd.DataFrame,
    raw_model,
    standardized_model,
    raw_contrast: dict,
    std_contrast: dict,
    band_contrasts: pd.DataFrame,
    marginal_df: pd.DataFrame,
    r2_summary: pd.DataFrame,
) -> None:
    b_raw = float(raw_model.params["balk_step_gap_c1_minus_c2"])
    n_raw = float(raw_model.params["no_show_step_gap_c1_minus_c2"])
    b_std = float(standardized_model.params["balk_step_gap_c1_minus_c2"])
    n_std = float(standardized_model.params["no_show_step_gap_c1_minus_c2"])

    supported = (
        b_raw < n_raw
        and raw_contrast["p_value_two_sided"] < ALPHA
        and b_raw < 0
        and n_raw < 0
    )

    lines = [
        "# Class-access regression robustness summary",
        "",
        f"Backgrounds analyzed: **{len(data):,}**.",
        f"Outcome: **Class 1 served rate - Class 2 served rate**.",
        "",
        "## Primary probability-gap comparison",
        "",
        f"- Raw balking-gap coefficient: **{b_raw:.6f}**.",
        f"- Raw no-show-gap coefficient: **{n_raw:.6f}**.",
        f"- Raw difference (balking - no-show): **{raw_contrast['estimate']:.6f}** "
        f"(95% CI {raw_contrast['ci_low_95']:.6f}, {raw_contrast['ci_high_95']:.6f}; "
        f"two-sided p={raw_contrast['p_value_two_sided']:.4g}; "
        f"one-sided p for balking more negative={directional_p_more_negative(raw_contrast):.4g}).",
        f"- Standardized balking-gap coefficient: **{b_std:.4f}**.",
        f"- Standardized no-show-gap coefficient: **{n_std:.4f}**.",
        f"- Standardized difference: **{std_contrast['estimate']:.4f}** "
        f"(two-sided p={std_contrast['p_value_two_sided']:.4g}).",
        "",
        "### Pre-specified decision rule",
        "",
        (
            "**SUPPORTED:** In this experiment, the balking probability gap is significantly more negative "
            "than the no-show probability gap." if supported else
            "**NOT YET SUPPORTED:** The primary coefficient/contrast criteria do not all support the stronger-balking claim."
        ),
        "",
        "## Demand-tercile contrasts",
        "",
        "```",
        band_contrasts.to_string(index=False),
        "```",
        "",
        "## Demand-interaction marginal effects",
        "",
        "```",
        marginal_df.to_string(index=False),
        "```",
        "",
        "## Drop-one-family held-out R²",
        "",
        "```",
        r2_summary.to_string(index=False),
        "```",
        "",
        "## Interpretation guardrail",
        "",
        "This experiment supports a statement about the simulated FCFS parameter space. "
        "It does not establish that balking is universally more important in every clinic or at every demand level.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace, design: pd.DataFrame, output_dir: Path) -> None:
    data_dir = output_dir / "data"
    figure_dir = output_dir / "figures"
    raw_dir = output_dir / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    seed_df = collect_seed_outcomes(raw_dir)
    validate_seed_outcomes(seed_df, design, args.seeds_per_background, args.allow_incomplete)
    atomic_write_csv(seed_df, data_dir / "seed_outcomes_combined.csv")

    data = aggregate_backgrounds(seed_df, design)
    atomic_write_csv(data, data_dir / "background_outcomes.csv")

    raw_model = fit_raw_ols(data, MAIN_FEATURES)
    std_model = fit_standardized_ols(data, MAIN_FEATURES)
    raw_coef = coefficient_table(raw_model, "main_raw")
    std_coef = coefficient_table(std_model, "main_standardized")
    atomic_write_csv(pd.concat([raw_coef, std_coef], ignore_index=True), data_dir / "regression_coefficients.csv")

    raw_contrast = linear_contrast(
        raw_model,
        {"balk_step_gap_c1_minus_c2": 1.0, "no_show_step_gap_c1_minus_c2": -1.0},
    )
    std_contrast = linear_contrast(
        std_model,
        {"balk_step_gap_c1_minus_c2": 1.0, "no_show_step_gap_c1_minus_c2": -1.0},
    )
    contrast_df = pd.DataFrame(
        [
            {
                "model": "main_raw",
                "contrast": "balk_step_gap - no_show_step_gap",
                **raw_contrast,
                "p_value_one_sided_balk_more_negative": directional_p_more_negative(raw_contrast),
            },
            {
                "model": "main_standardized",
                "contrast": "balk_step_gap - no_show_step_gap",
                **std_contrast,
                "p_value_one_sided_balk_more_negative": directional_p_more_negative(std_contrast),
            },
        ]
    )
    atomic_write_csv(contrast_df, data_dir / "wald_probability_gap_comparison.csv")

    band_coefs, band_contrasts = demand_stratified_tests(data)
    atomic_write_csv(band_coefs, data_dir / "demand_band_gap_coefficients.csv")
    atomic_write_csv(band_contrasts, data_dir / "demand_band_wald_comparisons.csv")

    int_coefs, marginal_df, _ = interaction_analysis(data)
    atomic_write_csv(int_coefs, data_dir / "demand_interaction_coefficients.csv")
    atomic_write_csv(marginal_df, data_dir / "demand_interaction_marginal_effects.csv")

    r2_splits, r2_summary = repeated_drop_family_r2(
        data,
        repetitions=args.r2_repetitions,
        seed=args.design_seed + 10_000,
    )
    atomic_write_csv(r2_splits, data_dir / "drop_family_r2_splits.csv")
    atomic_write_csv(r2_summary, data_dir / "drop_family_r2_summary.csv")

    plot_probability_gap_coefficients(raw_model, figure_dir)
    plot_interaction_marginal_effects(marginal_df, figure_dir)

    write_summary(
        output_dir / "summary.md",
        data,
        raw_model,
        std_model,
        raw_contrast,
        std_contrast,
        band_contrasts,
        marginal_df,
        r2_summary,
    )

    manifest = {
        "command": [Path(sys.executable).name, *sys.argv],
        "git": git_metadata(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "baseline_config": str((REPO_DIR / "configs" / "baseline.yaml").relative_to(REPO_DIR)),
        "baseline_config_sha256": file_sha256(REPO_DIR / "configs" / "baseline.yaml"),
        "simulation_span": {
            "burn_in_days": BASE_CONFIG.burn_in_days,
            "measure_days": BASE_CONFIG.measure_days,
            "cooldown_days": BASE_CONFIG.cooldown_days,
        },
        "design": {
            "n_backgrounds": len(design),
            "seeds_per_background": args.seeds_per_background,
            "design_seed": args.design_seed,
            "sim_seed_base": args.sim_seed_base,
            "constraint": "no_show_threshold <= balking_threshold for each class",
            "sampling": "Latin hypercube for continuous parameters; Latin-hypercube mapping over feasible threshold pairs",
        },
        "regression": {
            "target": TARGET,
            "features": MAIN_FEATURES,
            "cov_type": DEFAULT_COV_TYPE,
            "primary_contrast": "beta(balk_step_gap_c1_minus_c2) - beta(no_show_step_gap_c1_minus_c2)",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Analysis complete: {output_dir}")
    print(f"Primary results: {output_dir / 'summary.md'}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["design", "run", "analyze", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-backgrounds", type=int, default=DEFAULT_BACKGROUND_COUNT)
    parser.add_argument("--seeds-per-background", type=int, default=DEFAULT_SEEDS_PER_BACKGROUND)
    parser.add_argument("--design-seed", type=int, default=DEFAULT_DESIGN_SEED)
    parser.add_argument("--sim-seed-base", type=int, default=DEFAULT_SIM_SEED_BASE)

    # Global-sensitivity ranges. These reproduce the broad spirit of the
    # existing screening regression but enforce coherent threshold ordering.
    parser.add_argument("--lambda-min-mult", type=float, default=0.4)
    parser.add_argument("--lambda-max-mult", type=float, default=1.7)
    parser.add_argument("--class-share-min", type=float, default=0.1)
    parser.add_argument("--class-share-max", type=float, default=0.9)
    parser.add_argument("--balk-prob-min", type=float, default=0.0)
    parser.add_argument("--balk-prob-max", type=float, default=1.0)
    parser.add_argument("--no-show-prob-min", type=float, default=0.0)
    parser.add_argument("--no-show-prob-max", type=float, default=1.0)
    parser.add_argument("--cancel-prob-min", type=float, default=0.0)
    parser.add_argument("--cancel-prob-max", type=float, default=0.30)

    parser.add_argument("--no-show-threshold-min", type=int, default=0)
    parser.add_argument("--no-show-threshold-max", type=int, default=None)
    parser.add_argument("--min-threshold-spacing", type=int, default=0)
    parser.add_argument("--max-threshold-spacing", type=int, default=None)

    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--r2-repetitions", type=int, default=50)
    parser.add_argument("--overwrite-design", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n_backgrounds <= 0 or args.seeds_per_background <= 0:
        raise ValueError("n-backgrounds and seeds-per-background must be positive.")
    for low, high, name in [
        (args.class_share_min, args.class_share_max, "class share"),
        (args.balk_prob_min, args.balk_prob_max, "balking probability"),
        (args.no_show_prob_min, args.no_show_prob_max, "no-show probability"),
        (args.cancel_prob_min, args.cancel_prob_max, "cancellation probability"),
    ]:
        if not (0 <= low < high <= 1):
            raise ValueError(f"Invalid {name} range: [{low}, {high}].")
    if not (0 < args.lambda_min_mult < args.lambda_max_mult):
        raise ValueError("Invalid lambda multiplier range.")
    if args.min_threshold_spacing < 0:
        raise ValueError("min-threshold-spacing must be nonnegative.")


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.n_backgrounds = min(args.n_backgrounds, 24)
        args.seeds_per_background = min(args.seeds_per_background, 2)
        args.r2_repetitions = min(args.r2_repetitions, 5)
    validate_args(args)

    output_dir = args.output_dir
    data_dir = output_dir / "data"
    raw_dir = output_dir / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    design_path = data_dir / "design.csv"

    if args.mode in {"design", "all"}:
        if design_path.exists() and not args.overwrite_design:
            print(f"Design already exists; leaving unchanged: {design_path}")
            design = pd.read_csv(design_path)
        else:
            design = generate_design(args)
            atomic_write_csv(design, design_path)
            print(f"Wrote design: {design_path}")
    else:
        if not design_path.exists():
            raise FileNotFoundError(
                f"Design not found: {design_path}. Run --mode design first."
            )
        design = pd.read_csv(design_path)

    # Guard against accidentally running with CLI settings that disagree with
    # the existing design file.
    if len(design) != args.n_backgrounds:
        raise RuntimeError(
            f"Existing design has {len(design)} backgrounds but CLI requests {args.n_backgrounds}. "
            "Use matching arguments or --overwrite-design."
        )

    if args.mode in {"run", "all"}:
        run_shard(args, design, raw_dir)

    if args.mode in {"analyze", "all"}:
        # In a multi-shard run, analyze should be invoked separately after all
        # shard jobs finish. --mode all is intended for shard-count=1/smoke use.
        if args.mode == "all" and args.shard_count != 1:
            print("Skipping analyze in --mode all because shard-count != 1. Run --mode analyze after all shards finish.")
        else:
            analyze(args, design, output_dir)


if __name__ == "__main__":
    main()
