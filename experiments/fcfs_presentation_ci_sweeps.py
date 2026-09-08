#!/usr/bin/env python3
"""Replicated FCFS sweeps for presentation slides 5, 7, 8, and 10.

The script has three stages:

1. ``design`` creates one row per simulation replication.
2. ``run`` executes one resumable shard of the design.
3. ``analyze`` combines all shards, calculates pointwise 95% confidence
   intervals and paired differences from the pre-specified baseline, and
   creates the eight figures used by the presentation.

All sweeps inherit the simulation duration and non-focal parameters from
``configs/baseline.yaml``. The 3/5 thresholds in the patient-behavior
factorial are a separate experimental design and are not substituted into
these original FCFS presentation sweeps.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

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
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "fcfs_presentation_ci"
DEFAULT_SEEDS = 100
CI_Z = 1.96

DEMAND_VALUES = list(range(30, 171, 20))
DEMAND_BALKING_VALUES = [0.00, 0.50, 1.00]
DEMAND_NOSHOW_VALUES = [0.00, 0.15, 0.30, 0.60]
CLASS1_BALKING_VALUES = np.round(np.arange(0.0, 1.0, 0.1), 2).tolist()
CLASS1_NOSHOW_VALUES = np.round(np.arange(0.0, 1.0, 0.1), 2).tolist()
CLASS1_NOSHOW_THRESHOLDS = list(range(0, 14))
CLASS1_CANCELLATION_VALUES = np.round(np.arange(0.0, 0.301, 0.02), 2).tolist()

SWEEP_ORDER = [
    "demand_balking",
    "demand_noshow",
    "class1_balking",
    "class1_noshow_magnitude",
    "class1_noshow_threshold",
    "class1_cancellation",
]

GROUP_COLUMNS = {
    "demand_balking": ["total_arrivals_per_day", "focal_value"],
    "demand_noshow": ["total_arrivals_per_day", "focal_value"],
    "class1_balking": ["focal_value"],
    "class1_noshow_magnitude": ["focal_value"],
    "class1_noshow_threshold": ["focal_value"],
    "class1_cancellation": ["focal_value"],
}

BASELINE_VALUES = {
    "demand_balking": 0.00,
    "demand_noshow": 0.00,
    "class1_balking": 0.50,
    "class1_noshow_magnitude": 0.30,
    "class1_noshow_threshold": 6.00,
    "class1_cancellation": 0.10,
}

METRICS = [
    "average_utilization",
    "overall_percent_serviced",
    "class_1_percent_serviced",
    "class_2_percent_serviced",
    "class_1_slot_utilization",
    "class_2_slot_utilization",
]

OVERALL_COLOR = "#000000"
CLASS_1_COLOR = "#7B5AC7"
CLASS_2_COLOR = "#F58518"
BALKING_COLOR = "#E68613"
NOSHOW_COLOR = "#2474B5"


def _step_rule(rule: ThresholdRule, *, high: float | None = None, threshold: int | None = None) -> ThresholdRule:
    return ThresholdRule(
        threshold=rule.threshold if threshold is None else int(threshold),
        low=rule.low,
        high=rule.high if high is None else float(high),
    )


def _replace_class(config: SimulationConfig, class_id: int, **changes: Any) -> SimulationConfig:
    classes = dict(config.classes)
    classes[class_id] = replace(classes[class_id], **changes)
    return replace(config, classes=classes)


def config_for_design_row(base: SimulationConfig, row: pd.Series) -> SimulationConfig:
    """Construct the simulation configuration represented by one design row."""

    sweep = str(row["sweep"])
    focal = float(row["focal_value"])
    config = replace(base, seed=int(row["seed"]))

    if sweep in {"demand_balking", "demand_noshow"}:
        total_demand = float(row["total_arrivals_per_day"])
        for class_id in (1, 2):
            config = _replace_class(config, class_id, lambda_per_day=total_demand / 2.0)

        if sweep == "demand_balking":
            for class_id in (1, 2):
                rule = config.classes[class_id].balk_prob
                if not isinstance(rule, ThresholdRule):
                    raise TypeError("Demand-balking sweep requires ThresholdRule balking probabilities.")
                config = _replace_class(config, class_id, balk_prob=_step_rule(rule, high=focal))
        else:
            for class_id in (1, 2):
                rule = config.classes[class_id].no_show_prob
                if not isinstance(rule, ThresholdRule):
                    raise TypeError("Demand-no-show sweep requires ThresholdRule no-show probabilities.")
                config = _replace_class(config, class_id, no_show_prob=_step_rule(rule, high=focal))

    elif sweep == "class1_balking":
        rule = config.classes[1].balk_prob
        if not isinstance(rule, ThresholdRule):
            raise TypeError("Class 1 balking sweep requires a ThresholdRule.")
        config = _replace_class(config, 1, balk_prob=_step_rule(rule, high=focal))

    elif sweep == "class1_noshow_magnitude":
        rule = config.classes[1].no_show_prob
        if not isinstance(rule, ThresholdRule):
            raise TypeError("Class 1 no-show sweep requires a ThresholdRule.")
        config = _replace_class(config, 1, no_show_prob=_step_rule(rule, high=focal))

    elif sweep == "class1_noshow_threshold":
        rule = config.classes[1].no_show_prob
        if not isinstance(rule, ThresholdRule):
            raise TypeError("Class 1 no-show threshold sweep requires a ThresholdRule.")
        config = _replace_class(config, 1, no_show_prob=_step_rule(rule, threshold=int(focal)))

    elif sweep == "class1_cancellation":
        config = _replace_class(config, 1, cancel_prob=focal)

    else:
        raise ValueError(f"Unknown sweep: {sweep}")

    return config


def _design_rows(seeds: Iterable[int], smoke: bool) -> list[dict[str, Any]]:
    grids: dict[str, tuple[list[float], list[float | None]]] = {
        "demand_balking": ([float(x) for x in DEMAND_BALKING_VALUES], [float(x) for x in DEMAND_VALUES]),
        "demand_noshow": ([float(x) for x in DEMAND_NOSHOW_VALUES], [float(x) for x in DEMAND_VALUES]),
        "class1_balking": ([float(x) for x in CLASS1_BALKING_VALUES], [None]),
        "class1_noshow_magnitude": ([float(x) for x in CLASS1_NOSHOW_VALUES], [None]),
        "class1_noshow_threshold": ([float(x) for x in CLASS1_NOSHOW_THRESHOLDS], [None]),
        "class1_cancellation": ([float(x) for x in CLASS1_CANCELLATION_VALUES], [None]),
    }

    seed_values = list(seeds)
    if smoke:
        seed_values = seed_values[:2]
        for sweep, (focal_values, demand_values) in grids.items():
            baseline = BASELINE_VALUES[sweep]
            selected_focal = [focal_values[0]]
            if baseline not in selected_focal:
                selected_focal.append(baseline)
            grids[sweep] = (selected_focal, demand_values[:2])

    rows: list[dict[str, Any]] = []
    for sweep in SWEEP_ORDER:
        focal_values, demand_values = grids[sweep]
        for demand in demand_values:
            for focal in focal_values:
                for seed in seed_values:
                    rows.append(
                        {
                            "sweep": sweep,
                            "total_arrivals_per_day": demand,
                            "focal_value": focal,
                            "seed": int(seed),
                        }
                    )
    for task_id, row in enumerate(rows):
        row["task_id"] = task_id
    return rows


def create_design(output_dir: Path, n_seeds: int, smoke: bool) -> pd.DataFrame:
    if n_seeds < 2:
        raise ValueError("At least two seeds are required for confidence intervals.")
    design = pd.DataFrame(_design_rows(range(1, n_seeds + 1), smoke=smoke))
    design = design[["task_id", "sweep", "total_arrivals_per_day", "focal_value", "seed"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    design.to_csv(output_dir / "design.csv", index=False)
    print(f"Wrote {len(design):,} simulation tasks to {output_dir / 'design.csv'}")
    print(design.groupby("sweep").size().to_string())
    return design


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def _same_design_value(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return bool(np.isclose(float(left), float(right)))
    return left == right


def _validate_resumed_rows(existing: pd.DataFrame, shard: pd.DataFrame, shard_path: Path) -> None:
    """Refuse to resume if a shard belongs to a different design."""

    design_by_id = shard.set_index("task_id")
    for _, result_row in existing.iterrows():
        task_id = int(result_row["task_id"])
        if task_id not in design_by_id.index:
            raise RuntimeError(f"Stale task_id={task_id} in {shard_path}; use a new output directory.")
        design_row = design_by_id.loc[task_id]
        for column in ["sweep", "total_arrivals_per_day", "focal_value", "seed"]:
            if not _same_design_value(result_row[column], design_row[column]):
                raise RuntimeError(
                    f"Stale design value in {shard_path}: task_id={task_id}, column={column}. "
                    "Use a new output directory rather than mixing runs."
                )


def run_shard(output_dir: Path, shard_index: int, shard_count: int, resume: bool, checkpoint_every: int) -> None:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require shard_count >= 1 and 0 <= shard_index < shard_count.")

    design_path = output_dir / "design.csv"
    if not design_path.exists():
        raise FileNotFoundError(f"Missing design: {design_path}. Run --mode design first.")

    design = pd.read_csv(design_path)
    shard = design[design["task_id"] % shard_count == shard_index].copy()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shard_path = raw_dir / f"shard_{shard_index:04d}_of_{shard_count:04d}.csv"

    existing = pd.DataFrame()
    completed: set[int] = set()
    if resume and shard_path.exists():
        existing = pd.read_csv(shard_path)
        _validate_resumed_rows(existing, shard, shard_path)
        completed = set(existing["task_id"].astype(int))

    pending = shard[~shard["task_id"].isin(completed)]
    print(
        f"Shard {shard_index + 1}/{shard_count}: "
        f"{len(shard):,} assigned, {len(completed):,} complete, {len(pending):,} pending."
    )
    if pending.empty:
        return

    base = load_config(CONFIG_PATH)
    new_rows: list[dict[str, Any]] = []
    for position, (_, design_row) in enumerate(pending.iterrows(), start=1):
        config = config_for_design_row(base, design_row)
        result = ClinicAppointmentSimulation(config).run()
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
            _atomic_csv(combined, shard_path)
            print(f"Checkpoint: {len(combined):,}/{len(shard):,} rows -> {shard_path}")


def _load_complete_results(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    design = pd.read_csv(output_dir / "design.csv")
    shard_paths = sorted((output_dir / "raw").glob("shard_*.csv"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard CSVs found under {output_dir / 'raw'}")
    results = pd.concat([pd.read_csv(path) for path in shard_paths], ignore_index=True)
    results = results.sort_values("task_id").drop_duplicates("task_id", keep="last")

    expected = set(design["task_id"].astype(int))
    observed = set(results["task_id"].astype(int))
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Incomplete/mismatched results: missing={len(missing)}, unexpected={len(unexpected)}. "
            "Do not analyze until all shards finish."
        )
    return design, results


def pointwise_summary(results: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for sweep in SWEEP_ORDER:
        sub = results[results["sweep"] == sweep]
        group_cols = GROUP_COLUMNS[sweep]
        for metric in METRICS:
            summary = sub.groupby(group_cols, dropna=False)[metric].agg(mean="mean", std="std", n="count").reset_index()
            summary["se"] = summary["std"] / np.sqrt(summary["n"])
            summary["ci95"] = CI_Z * summary["se"]
            summary["ci_low_95"] = summary["mean"] - summary["ci95"]
            summary["ci_high_95"] = summary["mean"] + summary["ci95"]
            summary["metric"] = metric
            summary["sweep"] = sweep
            pieces.append(summary)
    return pd.concat(pieces, ignore_index=True, sort=False)


def paired_difference_summary(results: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for sweep in SWEEP_ORDER:
        sub = results[results["sweep"] == sweep].copy()
        baseline_value = BASELINE_VALUES[sweep]
        baseline = sub[np.isclose(sub["focal_value"], baseline_value)].copy()
        pair_cols = ["seed"]
        summary_groups = ["focal_value"]
        if sweep.startswith("demand_"):
            pair_cols.append("total_arrivals_per_day")
            summary_groups.insert(0, "total_arrivals_per_day")
        if baseline.empty:
            raise RuntimeError(f"No baseline rows found for {sweep} at focal_value={baseline_value}.")

        for metric in METRICS:
            base_metric = baseline[pair_cols + [metric]].rename(columns={metric: "baseline_metric"})
            paired = sub[pair_cols + ["focal_value", metric]].merge(base_metric, on=pair_cols, how="inner")
            paired["difference"] = paired[metric] - paired["baseline_metric"]
            summary = paired.groupby(summary_groups, dropna=False)["difference"].agg(
                mean_difference="mean", std_difference="std", n_paired="count"
            ).reset_index()
            summary["se_difference"] = summary["std_difference"] / np.sqrt(summary["n_paired"])
            summary["ci95"] = CI_Z * summary["se_difference"]
            summary["ci_low_95"] = summary["mean_difference"] - summary["ci95"]
            summary["ci_high_95"] = summary["mean_difference"] + summary["ci95"]
            summary["metric"] = metric
            summary["sweep"] = sweep
            summary["baseline_focal_value"] = baseline_value
            pieces.append(summary)
    return pd.concat(pieces, ignore_index=True, sort=False)


def _metric(summary: pd.DataFrame, sweep: str, metric: str) -> pd.DataFrame:
    return summary[(summary["sweep"] == sweep) & (summary["metric"] == metric)].copy()


def _errorbar(ax: plt.Axes, x: np.ndarray, y: np.ndarray, ci: np.ndarray, **kwargs: Any) -> None:
    ax.errorbar(
        x,
        y,
        yerr=ci,
        capsize=3.5,
        capthick=1.1,
        elinewidth=1.1,
        markersize=5.5,
        linewidth=2.0,
        **kwargs,
    )


def _finish_axis(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_demand(summary: pd.DataFrame, sweep: str, color: str, label_word: str, output_path: Path) -> None:
    data = _metric(summary, sweep, "overall_percent_serviced")
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    markers = ["o", "s", "^", "D"]
    linestyles = ["-", "--", "-.", ":"]
    focal_values = sorted(data["focal_value"].unique())
    alphas = np.linspace(0.45, 1.0, len(focal_values))
    for index, focal in enumerate(focal_values):
        sub = data[np.isclose(data["focal_value"], focal)].sort_values("total_arrivals_per_day")
        _errorbar(
            ax,
            sub["total_arrivals_per_day"].to_numpy(),
            sub["mean"].to_numpy(),
            sub["ci95"].to_numpy(),
            color=color,
            alpha=float(alphas[index]),
            marker=markers[index],
            linestyle=linestyles[index],
            label=f"{label_word} probability = {focal:.2f}",
        )
    _finish_axis(
        ax,
        "Total arrivals per day",
        "Overall served rate",
        f"Overall served rate vs demand at different {label_word.lower()} levels",
    )
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overall_and_classes(
    summary: pd.DataFrame,
    sweep: str,
    overall_metric: str,
    class_metric_suffix: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    series = [
        (overall_metric, "overall", OVERALL_COLOR, "o", "-"),
        (f"class_1_{class_metric_suffix}", "Class 1", CLASS_1_COLOR, "s", "--"),
        (f"class_2_{class_metric_suffix}", "Class 2", CLASS_2_COLOR, "^", ":"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for metric, label, color, marker, linestyle in series:
        sub = _metric(summary, sweep, metric).sort_values("focal_value")
        _errorbar(
            ax,
            sub["focal_value"].to_numpy(),
            sub["mean"].to_numpy(),
            sub["ci95"].to_numpy(),
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    _finish_axis(ax, xlabel, ylabel, title)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_figures(summary: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_demand(
        summary,
        "demand_balking",
        BALKING_COLOR,
        "Balking",
        figure_dir / "slide_05_balking_demand_served_rate_ci.png",
    )
    plot_demand(
        summary,
        "demand_noshow",
        NOSHOW_COLOR,
        "No-show",
        figure_dir / "slide_05_noshow_demand_served_rate_ci.png",
    )
    plot_overall_and_classes(
        summary,
        "class1_balking",
        "average_utilization",
        "slot_utilization",
        "Class 1 high balking probability",
        "Completed visits as share of total slots",
        "Average Utilization and Class Slot Shares",
        figure_dir / "slide_07_balking_utilization_slot_share_ci.png",
    )
    plot_overall_and_classes(
        summary,
        "class1_balking",
        "overall_percent_serviced",
        "percent_serviced",
        "Class 1 high balking probability",
        "Served rate",
        "Served Rate",
        figure_dir / "slide_07_balking_served_rate_ci.png",
    )
    plot_overall_and_classes(
        summary,
        "class1_noshow_magnitude",
        "average_utilization",
        "slot_utilization",
        "Class 1 no-show high-delay probability",
        "Completed visits as share of total slots",
        "Magnitude – post-threshold probability",
        figure_dir / "slide_08_noshow_magnitude_slot_share_ci.png",
    )
    plot_overall_and_classes(
        summary,
        "class1_noshow_threshold",
        "average_utilization",
        "slot_utilization",
        "Class 1 no-show threshold (days)",
        "Completed visits as share of total slots",
        "Delay – threshold",
        figure_dir / "slide_08_noshow_threshold_slot_share_ci.png",
    )
    plot_overall_and_classes(
        summary,
        "class1_cancellation",
        "overall_percent_serviced",
        "percent_serviced",
        "Class 1 cancellation probability",
        "Served rate",
        "Class 1 cancellation changes",
        figure_dir / "slide_10_cancellation_served_rate_ci.png",
    )
    plot_overall_and_classes(
        summary,
        "class1_cancellation",
        "average_utilization",
        "slot_utilization",
        "Class 1 cancellation probability",
        "Completed visits as share of total slots",
        "Class 1 cancellation changes",
        figure_dir / "slide_10_cancellation_slot_share_ci.png",
    )


def analyze(output_dir: Path) -> None:
    _, results = _load_complete_results(output_dir)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(results, summary_dir / "combined_seed_results.csv")

    point_summary = pointwise_summary(results)
    paired_summary = paired_difference_summary(results)
    _atomic_csv(point_summary, summary_dir / "pointwise_95ci.csv")
    _atomic_csv(paired_summary, summary_dir / "paired_baseline_differences_95ci.csv")
    create_figures(point_summary, output_dir / "figures")

    print(f"Analyzed {len(results):,} completed simulations.")
    print(f"Pointwise intervals: {summary_dir / 'pointwise_95ci.csv'}")
    print(f"Paired differences: {summary_dir / 'paired_baseline_differences_95ci.csv'}")
    print(f"Presentation figures: {output_dir / 'figures'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["design", "run", "analyze", "all"], required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.mode in {"design", "all"}:
        create_design(output_dir, n_seeds=args.n_seeds, smoke=args.smoke)
    if args.mode in {"run", "all"}:
        run_shard(
            output_dir,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            resume=not args.no_resume,
            checkpoint_every=args.checkpoint_every,
        )
    if args.mode in {"analyze", "all"}:
        analyze(output_dir)


if __name__ == "__main__":
    main()
