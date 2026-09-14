#!/usr/bin/env python3
"""Paired random-background behavior perturbations for the FCFS presentation.

Each random background is crossed with eight fixed demand levels and seven
conditions: one baseline plus three no-show and three balking perturbations.
The perturbations are paired by background, demand, and simulation seed.

The six contrasts are, for each behavior:
1. Increase Class 1 post-threshold probability by 0.10.
2. Increase Class 1 pre-threshold probability by 0.10.
3. Increase both classes' post-threshold probabilities by 0.10.

All generated configurations satisfy post > pre and, for each class,
the balking threshold is strictly later than the no-show threshold.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from analysis.metrics import outcome_rates_from_result, result_metrics_from_result
from simulation.config_loader import load_config
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import SimulationConfig, ThresholdRule


CONFIG_PATH = REPO_DIR / "configs" / "baseline.yaml"
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "fcfs_slide9_random_background_perturbations"

DEMAND_VALUES = list(range(30, 171, 20))
CONDITION_ORDER = [
    "baseline",
    "noshow_post_gap",
    "noshow_pre_gap",
    "noshow_common_post",
    "balking_post_gap",
    "balking_pre_gap",
    "balking_common_post",
]
CONDITION_INFO = {
    "baseline": ("baseline", "baseline", "Unchanged baseline"),
    "noshow_post_gap": ("no_show", "post_gap", "Increase C1 post-threshold probability by 10 pp"),
    "noshow_pre_gap": ("no_show", "pre_gap", "Increase C1 pre-threshold probability by 10 pp"),
    "noshow_common_post": ("no_show", "common_post", "Increase both classes' post-threshold probabilities by 10 pp"),
    "balking_post_gap": ("balking", "post_gap", "Increase C1 post-threshold probability by 10 pp"),
    "balking_pre_gap": ("balking", "pre_gap", "Increase C1 pre-threshold probability by 10 pp"),
    "balking_common_post": ("balking", "common_post", "Increase both classes' post-threshold probabilities by 10 pp"),
}

PERTURBATION = 0.10
PRE_MIN = 0.00
PRE_MAX = 0.20
MIN_POST_PRE_GAP = 0.15
POST_MAX = 0.85
CANCEL_MIN = 0.00
CANCEL_MAX = 0.30
CLASS_SHARE_MIN = 0.10
CLASS_SHARE_MAX = 0.90
DESIGN_SEED = 20260914
SIM_SEED_BASE = 9_140_000
CI_Z = 1.96

CLASS_1_COLOR = "#7B5AC7"
CLASS_2_COLOR = "#F58518"

OUTCOME_METRICS = {
    "class_1_percent_serviced": "Class 1",
    "class_2_percent_serviced": "Class 2",
}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def latin_hypercube(n: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    values = (np.arange(n)[:, None] + rng.random((n, dimensions))) / n
    for column in range(dimensions):
        rng.shuffle(values[:, column])
    return values


def scale(value: float, lower: float, upper: float) -> float:
    return float(lower + value * (upper - lower))


def probability_pair(pre_unit: float, post_unit: float) -> tuple[float, float]:
    pre = scale(pre_unit, PRE_MIN, PRE_MAX)
    lower_post = pre + MIN_POST_PRE_GAP
    post = scale(post_unit, lower_post, POST_MAX)
    return pre, post


def feasible_threshold_pairs(horizon_days: int) -> list[tuple[int, int]]:
    return [
        (noshow_threshold, balking_threshold)
        for noshow_threshold in range(horizon_days - 1)
        for balking_threshold in range(noshow_threshold + 1, horizon_days)
    ]


def select_threshold_pair(unit: float, pairs: list[tuple[int, int]]) -> tuple[int, int]:
    index = min(int(math.floor(unit * len(pairs))), len(pairs) - 1)
    return pairs[index]


def generate_backgrounds(n_backgrounds: int, design_seed: int = DESIGN_SEED) -> pd.DataFrame:
    if n_backgrounds < 2:
        raise ValueError("At least two random backgrounds are required.")
    base = load_config(CONFIG_PATH)
    rng = np.random.default_rng(design_seed)
    lhs = latin_hypercube(n_backgrounds, 13, rng)
    threshold_pairs = feasible_threshold_pairs(base.horizon_days)
    rows: list[dict[str, Any]] = []

    for background_id in range(n_backgrounds):
        u = lhs[background_id]
        c1_balk_low, c1_balk_high = probability_pair(u[3], u[4])
        c2_balk_low, c2_balk_high = probability_pair(u[5], u[6])
        c1_noshow_low, c1_noshow_high = probability_pair(u[7], u[8])
        c2_noshow_low, c2_noshow_high = probability_pair(u[9], u[10])
        c1_noshow_threshold, c1_balk_threshold = select_threshold_pair(u[11], threshold_pairs)
        c2_noshow_threshold, c2_balk_threshold = select_threshold_pair(u[12], threshold_pairs)
        rows.append(
            {
                "background_id": background_id,
                "class_1_share": scale(u[0], CLASS_SHARE_MIN, CLASS_SHARE_MAX),
                "class_1_cancel_prob": scale(u[1], CANCEL_MIN, CANCEL_MAX),
                "class_2_cancel_prob": scale(u[2], CANCEL_MIN, CANCEL_MAX),
                "class_1_balk_low": c1_balk_low,
                "class_1_balk_high": c1_balk_high,
                "class_1_balk_threshold": c1_balk_threshold,
                "class_2_balk_low": c2_balk_low,
                "class_2_balk_high": c2_balk_high,
                "class_2_balk_threshold": c2_balk_threshold,
                "class_1_noshow_low": c1_noshow_low,
                "class_1_noshow_high": c1_noshow_high,
                "class_1_noshow_threshold": c1_noshow_threshold,
                "class_2_noshow_low": c2_noshow_low,
                "class_2_noshow_high": c2_noshow_high,
                "class_2_noshow_threshold": c2_noshow_threshold,
            }
        )
    backgrounds = pd.DataFrame(rows)
    validate_backgrounds(backgrounds)
    return backgrounds


def validate_backgrounds(backgrounds: pd.DataFrame) -> None:
    if not backgrounds["background_id"].is_unique:
        raise AssertionError("background_id must be unique.")
    for class_id in (1, 2):
        for behavior in ("balk", "noshow"):
            low = backgrounds[f"class_{class_id}_{behavior}_low"]
            high = backgrounds[f"class_{class_id}_{behavior}_high"]
            if not ((low >= 0) & (high <= POST_MAX) & (high - low >= MIN_POST_PRE_GAP - 1e-12)).all():
                raise AssertionError(f"Invalid {behavior} probability pair for Class {class_id}.")
            if not (high > low + PERTURBATION).all():
                raise AssertionError("A +10 pp pre-threshold perturbation would violate post > pre.")
            if not (high + PERTURBATION <= 1.0).all():
                raise AssertionError("A +10 pp post-threshold perturbation would exceed one.")
        if not (
            backgrounds[f"class_{class_id}_balk_threshold"]
            > backgrounds[f"class_{class_id}_noshow_threshold"]
        ).all():
            raise AssertionError(f"Class {class_id} balking threshold must be later than no-show threshold.")


def apply_condition(background: dict[str, Any], condition: str) -> dict[str, Any]:
    values = dict(background)
    if condition == "noshow_post_gap":
        values["class_1_noshow_high"] += PERTURBATION
    elif condition == "noshow_pre_gap":
        values["class_1_noshow_low"] += PERTURBATION
    elif condition == "noshow_common_post":
        values["class_1_noshow_high"] += PERTURBATION
        values["class_2_noshow_high"] += PERTURBATION
    elif condition == "balking_post_gap":
        values["class_1_balk_high"] += PERTURBATION
    elif condition == "balking_pre_gap":
        values["class_1_balk_low"] += PERTURBATION
    elif condition == "balking_common_post":
        values["class_1_balk_high"] += PERTURBATION
        values["class_2_balk_high"] += PERTURBATION
    elif condition != "baseline":
        raise ValueError(f"Unknown condition: {condition}")
    return values


def simulation_seed(background_id: int, demand_index: int, replicate: int, seeds_per_background: int) -> int:
    unit = (background_id * len(DEMAND_VALUES) + demand_index) * seeds_per_background + replicate
    return SIM_SEED_BASE + unit


def create_design(
    output_dir: Path,
    study: str,
    n_backgrounds: int,
    seeds_per_background: int,
    design_seed: int,
) -> pd.DataFrame:
    if seeds_per_background < 2:
        raise ValueError("At least two seeds per background are required.")
    stale = list((output_dir / "raw").glob("shard_*.csv")) + list((output_dir / "completed").glob("*.done"))
    if stale:
        raise RuntimeError(f"Existing run output found under {output_dir}; use a new output directory.")

    backgrounds = generate_backgrounds(n_backgrounds, design_seed)
    rows: list[dict[str, Any]] = []
    pair_id = 0
    for background in backgrounds.to_dict(orient="records"):
        for demand_index, demand in enumerate(DEMAND_VALUES):
            for replicate in range(seeds_per_background):
                seed = simulation_seed(int(background["background_id"]), demand_index, replicate, seeds_per_background)
                for condition in CONDITION_ORDER:
                    behavior, perturbation, description = CONDITION_INFO[condition]
                    final = apply_condition(background, condition)
                    rows.append(
                        {
                            "task_id": len(rows),
                            "pair_id": pair_id,
                            "study": study,
                            "condition": condition,
                            "behavior": behavior,
                            "perturbation": perturbation,
                            "condition_description": description,
                            "total_arrivals_per_day": float(demand),
                            "demand_index": demand_index,
                            "replicate": replicate,
                            "seed": seed,
                            **final,
                        }
                    )
                pair_id += 1
    design = pd.DataFrame(rows)
    validate_design(design, n_backgrounds, seeds_per_background)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(backgrounds, output_dir / "backgrounds.csv")
    atomic_csv(design, output_dir / "design.csv")
    print(f"Wrote {len(backgrounds):,} random backgrounds to {output_dir / 'backgrounds.csv'}")
    print(f"Wrote {len(design):,} paired simulation tasks to {output_dir / 'design.csv'}")
    print(design.groupby("condition").size().reindex(CONDITION_ORDER).to_string())
    return design


def validate_design(design: pd.DataFrame, n_backgrounds: int, seeds_per_background: int) -> None:
    expected = n_backgrounds * len(DEMAND_VALUES) * seeds_per_background * len(CONDITION_ORDER)
    if len(design) != expected:
        raise AssertionError(f"Expected {expected} tasks; found {len(design)}.")
    counts = design.groupby("pair_id")["condition"].nunique()
    if not (counts == len(CONDITION_ORDER)).all():
        raise AssertionError("Every paired unit must contain all seven conditions.")
    seeds = design.groupby("pair_id")["seed"].nunique()
    if not (seeds == 1).all():
        raise AssertionError("All conditions in a paired unit must share one seed.")
    for class_id in (1, 2):
        for behavior in ("balk", "noshow"):
            low = design[f"class_{class_id}_{behavior}_low"]
            high = design[f"class_{class_id}_{behavior}_high"]
            if not ((low >= 0) & (high <= 1) & (high > low)).all():
                raise AssertionError(f"Invalid final {behavior} probabilities for Class {class_id}.")
        if not (design[f"class_{class_id}_balk_threshold"] > design[f"class_{class_id}_noshow_threshold"]).all():
            raise AssertionError(f"Invalid threshold ordering for Class {class_id}.")


def replace_class(config: SimulationConfig, class_id: int, **changes: Any) -> SimulationConfig:
    classes = dict(config.classes)
    classes[class_id] = replace(classes[class_id], **changes)
    return replace(config, classes=classes)


def config_for_row(base: SimulationConfig, row: pd.Series) -> SimulationConfig:
    total_demand = float(row["total_arrivals_per_day"])
    class_1_share = float(row["class_1_share"])
    config = replace(base, seed=int(row["seed"]))
    for class_id, share in ((1, class_1_share), (2, 1.0 - class_1_share)):
        config = replace_class(
            config,
            class_id,
            lambda_per_day=total_demand * share,
            cancel_prob=float(row[f"class_{class_id}_cancel_prob"]),
            balk_prob=ThresholdRule(
                threshold=int(row[f"class_{class_id}_balk_threshold"]),
                low=float(row[f"class_{class_id}_balk_low"]),
                high=float(row[f"class_{class_id}_balk_high"]),
            ),
            no_show_prob=ThresholdRule(
                threshold=int(row[f"class_{class_id}_noshow_threshold"]),
                low=float(row[f"class_{class_id}_noshow_low"]),
                high=float(row[f"class_{class_id}_noshow_high"]),
            ),
        )
    optional_controls = {
        "reserved_class_id": None,
        "reserved_slots_per_day": 0,
        "reserved_window_days": None,
        "same_day_cancellation_enabled": False,
        "release_unused_reservation_same_day": False,
    }
    applicable = {key: value for key, value in optional_controls.items() if hasattr(config, key)}
    return replace(config, **applicable)


def validate_resume(existing: pd.DataFrame, shard: pd.DataFrame, path: Path) -> None:
    expected = shard.set_index("task_id")
    for _, row in existing.iterrows():
        task_id = int(row["task_id"])
        if task_id not in expected.index:
            raise RuntimeError(f"Stale task_id={task_id} in {path}.")
        design_row = expected.loc[task_id]
        for column in ("pair_id", "condition", "background_id", "total_arrivals_per_day", "replicate", "seed"):
            if row[column] != design_row[column]:
                raise RuntimeError(f"Stale {column} for task_id={task_id} in {path}.")


def run_shard(output_dir: Path, shard_index: int, shard_count: int, checkpoint_every: int) -> None:
    if not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count.")
    design_path = output_dir / "design.csv"
    if not design_path.exists():
        raise FileNotFoundError(f"Missing design: {design_path}")
    design = pd.read_csv(design_path)
    shard = design[design["pair_id"] % shard_count == shard_index].copy()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"shard_{shard_index:04d}_of_{shard_count:04d}.csv"
    existing = pd.DataFrame()
    completed: set[int] = set()
    if path.exists():
        existing = pd.read_csv(path)
        validate_resume(existing, shard, path)
        completed = set(existing["task_id"].astype(int))
    pending = shard[~shard["task_id"].isin(completed)]
    print(f"Shard {shard_index + 1}/{shard_count}: {len(shard):,} assigned, {len(completed):,} complete, {len(pending):,} pending.")
    if pending.empty:
        return

    base = load_config(CONFIG_PATH)
    new_rows: list[dict[str, Any]] = []
    for position, (_, design_row) in enumerate(pending.iterrows(), start=1):
        result = ClinicAppointmentSimulation(config_for_row(base, design_row)).run()
        new_rows.append(
            {
                **design_row.to_dict(),
                **result_metrics_from_result(result),
                **outcome_rates_from_result(result),
            }
        )
        if position % checkpoint_every == 0 or position == len(pending):
            combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            combined = combined.sort_values("task_id").drop_duplicates("task_id", keep="last")
            atomic_csv(combined, path)
            print(f"Checkpoint: {len(combined):,}/{len(shard):,} rows -> {path}")


def load_complete_results(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    design = pd.read_csv(output_dir / "design.csv")
    paths = sorted((output_dir / "raw").glob("shard_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No shard files found under {output_dir / 'raw'}")
    results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    results = results.sort_values("task_id").drop_duplicates("task_id", keep="last")
    expected = set(design["task_id"].astype(int))
    observed = set(results["task_id"].astype(int))
    if expected != observed:
        raise RuntimeError(f"Incomplete results: missing={len(expected-observed)}, unexpected={len(observed-expected)}")
    return design, results


def paired_differences(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["pair_id", "background_id", "total_arrivals_per_day", "replicate", "seed"]
    baseline = results[results["condition"] == "baseline"]
    perturbations = results[results["condition"] != "baseline"]
    pieces = []
    for metric, class_label in OUTCOME_METRICS.items():
        reference = baseline[keys + [metric]].rename(columns={metric: "baseline_served_rate"})
        paired = perturbations[keys + ["condition", "behavior", "perturbation", metric]].merge(
            reference,
            on=keys,
            how="inner",
            validate="many_to_one",
        )
        paired["class"] = class_label
        paired["perturbed_served_rate"] = paired[metric]
        paired["served_rate_change_pp"] = 100.0 * (paired[metric] - paired["baseline_served_rate"])
        pieces.append(
            paired[
                keys
                + [
                    "condition",
                    "behavior",
                    "perturbation",
                    "class",
                    "baseline_served_rate",
                    "perturbed_served_rate",
                    "served_rate_change_pp",
                ]
            ]
        )
    differences = pd.concat(pieces, ignore_index=True)
    expected = len(baseline) * 6 * 2
    if len(differences) != expected:
        raise AssertionError(f"Expected {expected} paired class contrasts; found {len(differences)}.")
    return differences


def summarize_backgrounds(differences: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    background_keys = [
        "background_id",
        "total_arrivals_per_day",
        "condition",
        "behavior",
        "perturbation",
        "class",
    ]
    background_means = (
        differences.groupby(background_keys, as_index=False)["served_rate_change_pp"]
        .mean()
        .rename(columns={"served_rate_change_pp": "background_mean_change_pp"})
    )
    summary_keys = ["condition", "behavior", "perturbation", "class", "total_arrivals_per_day"]
    summary = (
        background_means.groupby(summary_keys)["background_mean_change_pp"]
        .agg(mean_change_pp="mean", std_across_backgrounds="std", n_backgrounds="count")
        .reset_index()
    )
    summary["se_across_backgrounds"] = summary["std_across_backgrounds"] / np.sqrt(summary["n_backgrounds"])
    summary["ci95_half_width"] = CI_Z * summary["se_across_backgrounds"]
    summary["ci_low_95"] = summary["mean_change_pp"] - summary["ci95_half_width"]
    summary["ci_high_95"] = summary["mean_change_pp"] + summary["ci95_half_width"]
    return background_means, summary


def common_y_limits(summary: pd.DataFrame) -> tuple[float, float]:
    lower = min(0.0, float(summary["ci_low_95"].min()))
    upper = max(0.0, float(summary["ci_high_95"].max()))
    span = max(upper - lower, 0.5)
    padding = 0.08 * span
    return lower - padding, upper + padding


def plot_condition(ax: plt.Axes, summary: pd.DataFrame, condition: str, y_limits: tuple[float, float]) -> None:
    subset = summary[summary["condition"] == condition]
    styles = [
        ("Class 1", CLASS_1_COLOR, "s", "--"),
        ("Class 2", CLASS_2_COLOR, "^", ":"),
    ]
    for class_label, color, marker, linestyle in styles:
        data = subset[subset["class"] == class_label].sort_values("total_arrivals_per_day")
        x = data["total_arrivals_per_day"].to_numpy(dtype=float)
        mean = data["mean_change_pp"].to_numpy(dtype=float)
        low = data["ci_low_95"].to_numpy(dtype=float)
        high = data["ci_high_95"].to_numpy(dtype=float)
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
        ax.errorbar(
            x,
            mean,
            yerr=np.vstack([mean - low, high - mean]),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2,
            markersize=5.5,
            elinewidth=1.2,
            capsize=3,
            label=class_label,
        )
    ax.axhline(0.0, color="#333333", linewidth=1.2)
    ax.set_xlim(min(DEMAND_VALUES) - 4, max(DEMAND_VALUES) + 4)
    ax.set_ylim(*y_limits)
    ax.set_xticks(DEMAND_VALUES[::2])
    ax.grid(True, alpha=0.23)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def condition_title(condition: str) -> str:
    titles = {
        "noshow_post_gap": "C1 post-threshold +10 pp",
        "noshow_pre_gap": "C1 pre-threshold +10 pp",
        "noshow_common_post": "C1 and C2 post-threshold +10 pp",
        "balking_post_gap": "C1 post-threshold +10 pp",
        "balking_pre_gap": "C1 pre-threshold +10 pp",
        "balking_common_post": "C1 and C2 post-threshold +10 pp",
    }
    return titles[condition]


def create_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    y_limits = common_y_limits(summary)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )
    panels = {
        "no_show": ["noshow_post_gap", "noshow_pre_gap", "noshow_common_post"],
        "balking": ["balking_post_gap", "balking_pre_gap", "balking_common_post"],
    }
    for behavior, conditions in panels.items():
        prefix = "noshow" if behavior == "no_show" else "balking"
        slide_number = "09" if behavior == "no_show" else "10"
        for condition in conditions:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            plot_condition(ax, summary, condition, y_limits)
            ax.set_title(condition_title(condition))
            ax.set_xlabel("Total arrivals per day")
            ax.set_ylabel("Change in served rate (pp)")
            ax.legend(frameon=False)
            fig.savefig(
                figure_dir / f"slide_{slide_number}_{condition}_served_rate_change_ci.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.1), sharex=True, sharey=True)
        for ax, condition in zip(axes, conditions):
            plot_condition(ax, summary, condition, y_limits)
            ax.set_title(condition_title(condition))
            ax.set_xlabel("Total arrivals per day")
        axes[0].set_ylabel("Change in served rate (pp)")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.015))
        behavior_label = "No-show" if behavior == "no_show" else "Balking"
        fig.suptitle(f"Effect of 10 pp {behavior_label.lower()} probability perturbations", fontsize=17)
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.20, top=0.82, wspace=0.05)
        fig.savefig(
            figure_dir / f"slide_{slide_number}_{prefix}_three_panel_served_rate_change_ci.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def analyze(output_dir: Path) -> None:
    design, results = load_complete_results(output_dir)
    differences = paired_differences(results)
    background_means, summary = summarize_backgrounds(differences)
    summary_dir = output_dir / "summary"
    atomic_csv(results, summary_dir / "combined_seed_results.csv")
    atomic_csv(differences, summary_dir / "paired_seed_differences.csv")
    atomic_csv(background_means, summary_dir / "background_mean_differences.csv")
    atomic_csv(summary, summary_dir / "pointwise_background_95ci.csv")
    create_figures(summary, output_dir)
    print(f"Validated and analyzed {len(design):,} simulations.")
    print(f"Random backgrounds: {design['background_id'].nunique():,}")
    print(f"Paired seeds per background/demand: {design['replicate'].nunique():,}")
    print(f"Pointwise background-level intervals: {summary_dir / 'pointwise_background_95ci.csv'}")
    print(f"Figures: {output_dir / 'figures'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["design", "run", "analyze", "all"], required=True)
    parser.add_argument("--study", choices=["smoke", "full"], required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-backgrounds", type=int, default=None)
    parser.add_argument("--seeds-per-background", type=int, default=None)
    parser.add_argument("--design-seed", type=int, default=DESIGN_SEED)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_backgrounds = 5 if args.study == "smoke" else 100
    default_seeds = 2 if args.study == "smoke" else 5
    n_backgrounds = args.n_backgrounds if args.n_backgrounds is not None else default_backgrounds
    seeds_per_background = args.seeds_per_background if args.seeds_per_background is not None else default_seeds
    if args.mode in {"design", "all"}:
        create_design(args.output_dir, args.study, n_backgrounds, seeds_per_background, args.design_seed)
    if args.mode in {"run", "all"}:
        run_shard(args.output_dir, args.shard_index, args.shard_count, args.checkpoint_every)
    if args.mode in {"analyze", "all"}:
        analyze(args.output_dir)


if __name__ == "__main__":
    main()
