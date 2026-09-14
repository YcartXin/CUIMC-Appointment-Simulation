#!/usr/bin/env python3
"""Controlled no-show versus balking experiment for the FCFS presentation.

The design holds the pre-threshold probability at 0.02 and the threshold at
five days.  Class 2 stays at the low post-threshold probability (0.05).
Class 1 is varied through low, medium, and high post-threshold probabilities
(0.05, 0.15, 0.25) for one behavior at a time, while the other behavior stays
low for both classes.

Stages
------
``design``
    Write the task bank.  ``--study audit`` creates only the low/low profile;
    ``--study full`` creates the five unique profiles needed for six panels.
``run``
    Run one resumable shard.
``analyze``
    Validate and combine shards.  Audit analysis reports five-day exposure.
    Full analysis calculates pointwise and paired 95% CIs and creates figures.
"""

from __future__ import annotations

import argparse
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

from analysis.metrics import outcome_rates_from_result, result_metrics_from_result, safe_divide
from simulation.config_loader import load_config
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import SimulationConfig, SimulationResults, ThresholdRule


CONFIG_PATH = REPO_DIR / "configs" / "baseline.yaml"
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "fcfs_slide9_behavior_comparison"

DEMAND_VALUES = list(range(30, 171, 20))
THRESHOLD_DAYS = 5
PRE_THRESHOLD_PROBABILITY = 0.02
SEVERITY_VALUES = {"low": 0.05, "medium": 0.15, "high": 0.25}
CANCELLATION_PROBABILITY = 0.10
CI_Z = 1.96

CLASS_1_COLOR = "#7B5AC7"
CLASS_2_COLOR = "#F58518"

# The low/low profile is reused as the low panel in both behavior groups.
PROFILE_ORDER = ["low_low", "noshow_medium", "noshow_high", "balking_medium", "balking_high"]
PROFILES = {
    "low_low": {
        "panel_behavior": "baseline",
        "severity": "low",
        "class_1_balk_high": SEVERITY_VALUES["low"],
        "class_2_balk_high": SEVERITY_VALUES["low"],
        "class_1_noshow_high": SEVERITY_VALUES["low"],
        "class_2_noshow_high": SEVERITY_VALUES["low"],
    },
    "noshow_medium": {
        "panel_behavior": "no_show",
        "severity": "medium",
        "class_1_balk_high": SEVERITY_VALUES["low"],
        "class_2_balk_high": SEVERITY_VALUES["low"],
        "class_1_noshow_high": SEVERITY_VALUES["medium"],
        "class_2_noshow_high": SEVERITY_VALUES["low"],
    },
    "noshow_high": {
        "panel_behavior": "no_show",
        "severity": "high",
        "class_1_balk_high": SEVERITY_VALUES["low"],
        "class_2_balk_high": SEVERITY_VALUES["low"],
        "class_1_noshow_high": SEVERITY_VALUES["high"],
        "class_2_noshow_high": SEVERITY_VALUES["low"],
    },
    "balking_medium": {
        "panel_behavior": "balking",
        "severity": "medium",
        "class_1_balk_high": SEVERITY_VALUES["medium"],
        "class_2_balk_high": SEVERITY_VALUES["low"],
        "class_1_noshow_high": SEVERITY_VALUES["low"],
        "class_2_noshow_high": SEVERITY_VALUES["low"],
    },
    "balking_high": {
        "panel_behavior": "balking",
        "severity": "high",
        "class_1_balk_high": SEVERITY_VALUES["high"],
        "class_2_balk_high": SEVERITY_VALUES["low"],
        "class_1_noshow_high": SEVERITY_VALUES["low"],
        "class_2_noshow_high": SEVERITY_VALUES["low"],
    },
}

POINTWISE_METRICS = [
    "overall_percent_serviced",
    "class_1_percent_serviced",
    "class_2_percent_serviced",
    "access_advantage_class_1",
    "average_utilization",
    "pooled_accepted_threshold_exposure_share",
    "class_1_accepted_threshold_exposure_share",
    "class_2_accepted_threshold_exposure_share",
]


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _step_rule(high: float) -> ThresholdRule:
    return ThresholdRule(
        threshold=THRESHOLD_DAYS,
        low=PRE_THRESHOLD_PROBABILITY,
        high=float(high),
    )


def _replace_class(config: SimulationConfig, class_id: int, **changes: Any) -> SimulationConfig:
    classes = dict(config.classes)
    classes[class_id] = replace(classes[class_id], **changes)
    return replace(config, classes=classes)


def config_for_design_row(base: SimulationConfig, row: pd.Series) -> SimulationConfig:
    """Return the fully controlled configuration for one design row."""

    total_demand = float(row["total_arrivals_per_day"])
    config = replace(base, seed=int(row["seed"]))
    for class_id in (1, 2):
        config = _replace_class(
            config,
            class_id,
            lambda_per_day=total_demand / 2.0,
            cancel_prob=CANCELLATION_PROBABILITY,
        )

    config = _replace_class(
        config,
        1,
        balk_prob=_step_rule(float(row["class_1_balk_high"])),
        no_show_prob=_step_rule(float(row["class_1_noshow_high"])),
    )
    config = _replace_class(
        config,
        2,
        balk_prob=_step_rule(float(row["class_2_balk_high"])),
        no_show_prob=_step_rule(float(row["class_2_noshow_high"])),
    )
    return config


def create_design(output_dir: Path, study: str, n_seeds: int) -> pd.DataFrame:
    if n_seeds < 2:
        raise ValueError("At least two seeds are required for confidence intervals.")
    stale_raw = list((output_dir / "raw").glob("shard_*.csv"))
    stale_markers = list((output_dir / "completed").glob("*.done"))
    if stale_raw or stale_markers:
        raise RuntimeError(
            f"Existing run output found under {output_dir}. Use a new output directory "
            "or deliberately archive/remove the old run before creating a new design."
        )
    profile_ids = ["low_low"] if study == "audit" else PROFILE_ORDER
    rows: list[dict[str, Any]] = []
    for profile_id in profile_ids:
        profile = PROFILES[profile_id]
        for demand in DEMAND_VALUES:
            for seed in range(1, n_seeds + 1):
                rows.append(
                    {
                        "study": study,
                        "profile_id": profile_id,
                        **profile,
                        "total_arrivals_per_day": float(demand),
                        "threshold_days": THRESHOLD_DAYS,
                        "pre_threshold_probability": PRE_THRESHOLD_PROBABILITY,
                        "cancellation_probability": CANCELLATION_PROBABILITY,
                        "seed": seed,
                    }
                )
    for task_id, row in enumerate(rows):
        row["task_id"] = task_id
    design = pd.DataFrame(rows)
    columns = [
        "task_id",
        "study",
        "profile_id",
        "panel_behavior",
        "severity",
        "total_arrivals_per_day",
        "class_1_balk_high",
        "class_2_balk_high",
        "class_1_noshow_high",
        "class_2_noshow_high",
        "threshold_days",
        "pre_threshold_probability",
        "cancellation_probability",
        "seed",
    ]
    design = design[columns]
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(design, output_dir / "design.csv")
    print(f"Wrote {len(design):,} {study} tasks to {output_dir / 'design.csv'}")
    print(design.groupby("profile_id").size().to_string())
    return design


def _threshold_exposure_metrics(result: SimulationResults) -> dict[str, float]:
    """Summarize accepted bookings whose original delay reaches the threshold.

    The engine records accepted-delay histograms.  This gives exact exposure
    for the no-show rule and a conservative indicator of exposure to the
    balking rule because patients who balk are not accepted bookings.
    """

    output: dict[str, float] = {}
    pooled_booked = 0
    pooled_exposed = 0
    pooled_max = 0
    for class_id in (1, 2):
        metrics = result.class_metrics[class_id]
        exposed = sum(count for delay, count in metrics.accepted_delay_counts.items() if delay >= THRESHOLD_DAYS)
        maximum = max(metrics.accepted_delay_counts, default=0)
        output[f"class_{class_id}_accepted_threshold_count"] = int(exposed)
        output[f"class_{class_id}_accepted_threshold_exposure_share"] = safe_divide(exposed, metrics.booked)
        output[f"class_{class_id}_max_accepted_delay"] = int(maximum)
        pooled_booked += metrics.booked
        pooled_exposed += exposed
        pooled_max = max(pooled_max, maximum)
    output["pooled_accepted_threshold_count"] = int(pooled_exposed)
    output["pooled_accepted_threshold_exposure_share"] = safe_divide(pooled_exposed, pooled_booked)
    output["pooled_max_accepted_delay"] = int(pooled_max)
    return output


def _same_value(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return bool(np.isclose(float(left), float(right)))
    return left == right


def _validate_resumed_rows(existing: pd.DataFrame, shard: pd.DataFrame, shard_path: Path) -> None:
    design_by_id = shard.set_index("task_id")
    check_columns = [
        "study",
        "profile_id",
        "total_arrivals_per_day",
        "class_1_balk_high",
        "class_1_noshow_high",
        "seed",
    ]
    for _, result_row in existing.iterrows():
        task_id = int(result_row["task_id"])
        if task_id not in design_by_id.index:
            raise RuntimeError(f"Stale task_id={task_id} in {shard_path}; use a new output directory.")
        design_row = design_by_id.loc[task_id]
        for column in check_columns:
            if not _same_value(result_row[column], design_row[column]):
                raise RuntimeError(
                    f"Stale design value in {shard_path}: task_id={task_id}, column={column}. "
                    "Use a new output directory rather than mixing runs."
                )


def run_shard(
    output_dir: Path,
    shard_index: int,
    shard_count: int,
    resume: bool,
    checkpoint_every: int,
) -> None:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require shard_count >= 1 and 0 <= shard_index < shard_count.")
    design_path = output_dir / "design.csv"
    if not design_path.exists():
        raise FileNotFoundError(f"Missing design: {design_path}. Run the design stage first.")

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
        f"Shard {shard_index + 1}/{shard_count}: {len(shard):,} assigned, "
        f"{len(completed):,} complete, {len(pending):,} pending."
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
                **_threshold_exposure_metrics(result),
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
    missing = expected - observed
    unexpected = observed - expected
    if missing or unexpected:
        raise RuntimeError(
            f"Incomplete or mismatched results: missing={len(missing)}, unexpected={len(unexpected)}."
        )
    return design, results


def _summarize_metric(data: pd.DataFrame, group_columns: list[str], metric: str) -> pd.DataFrame:
    summary = data.groupby(group_columns, dropna=False)[metric].agg(mean="mean", std="std", n="count").reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["n"])
    summary["ci95"] = CI_Z * summary["se"]
    summary["ci_low_95"] = summary["mean"] - summary["ci95"]
    summary["ci_high_95"] = summary["mean"] + summary["ci95"]
    summary["metric"] = metric
    return summary


def analyze_audit(results: pd.DataFrame, output_dir: Path) -> None:
    summary_parts = []
    metrics = [
        "pooled_accepted_threshold_exposure_share",
        "class_1_accepted_threshold_exposure_share",
        "class_2_accepted_threshold_exposure_share",
        "pooled_max_accepted_delay",
        "class_1_max_accepted_delay",
        "class_2_max_accepted_delay",
        "mean_offered_booking_delay",
    ]
    for metric in metrics:
        summary_parts.append(_summarize_metric(results, ["total_arrivals_per_day"], metric))
    summary = pd.concat(summary_parts, ignore_index=True)
    _atomic_csv(results, output_dir / "summary" / "combined_seed_results.csv")
    _atomic_csv(summary, output_dir / "summary" / "threshold_exposure_audit_95ci.csv")

    table = summary.pivot(index="total_arrivals_per_day", columns="metric", values="mean").reset_index()
    table = table[
        [
            "total_arrivals_per_day",
            "pooled_accepted_threshold_exposure_share",
            "class_1_accepted_threshold_exposure_share",
            "class_2_accepted_threshold_exposure_share",
            "pooled_max_accepted_delay",
            "mean_offered_booking_delay",
        ]
    ]
    _atomic_csv(table, output_dir / "summary" / "threshold_exposure_audit_readable.csv")
    display = table.copy()
    for column in [
        "pooled_accepted_threshold_exposure_share",
        "class_1_accepted_threshold_exposure_share",
        "class_2_accepted_threshold_exposure_share",
    ]:
        display[column] = 100.0 * display[column]
    print("\nFive-day accepted-booking exposure audit (shares shown as percentages):")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nAudit summary: {output_dir / 'summary' / 'threshold_exposure_audit_readable.csv'}")


def pointwise_summary(results: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    groups = ["profile_id", "panel_behavior", "severity", "total_arrivals_per_day"]
    for metric in POINTWISE_METRICS:
        pieces.append(_summarize_metric(results, groups, metric))
    return pd.concat(pieces, ignore_index=True)


def paired_summary(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results[results["profile_id"] == "low_low"]
    pair_keys = ["seed", "total_arrivals_per_day"]
    pieces = []
    for metric in ["class_1_percent_serviced", "class_2_percent_serviced", "access_advantage_class_1"]:
        base = baseline[pair_keys + [metric]].rename(columns={metric: "baseline_value"})
        paired = results[pair_keys + ["profile_id", "panel_behavior", "severity", metric]].merge(
            base,
            on=pair_keys,
            how="inner",
        )
        paired["difference"] = paired[metric] - paired["baseline_value"]
        groups = ["profile_id", "panel_behavior", "severity", "total_arrivals_per_day"]
        summary = paired.groupby(groups)["difference"].agg(mean_difference="mean", std_difference="std", n_paired="count").reset_index()
        summary["se_difference"] = summary["std_difference"] / np.sqrt(summary["n_paired"])
        summary["ci95"] = CI_Z * summary["se_difference"]
        summary["ci_low_95"] = summary["mean_difference"] - summary["ci95"]
        summary["ci_high_95"] = summary["mean_difference"] + summary["ci95"]
        summary["metric"] = metric
        summary["baseline_profile_id"] = "low_low"
        pieces.append(summary)
    return pd.concat(pieces, ignore_index=True)


def _profile_for_panel(behavior: str, severity: str) -> str:
    if severity == "low":
        return "low_low"
    return f"{'noshow' if behavior == 'no_show' else 'balking'}_{severity}"


def _plot_one_panel(ax: plt.Axes, summary: pd.DataFrame, behavior: str, severity: str) -> None:
    profile_id = _profile_for_panel(behavior, severity)
    data = summary[summary["profile_id"] == profile_id]
    for metric, label, color, marker, linestyle in [
        ("class_1_percent_serviced", "Class 1", CLASS_1_COLOR, "s", "--"),
        ("class_2_percent_serviced", "Class 2", CLASS_2_COLOR, "^", ":"),
    ]:
        sub = data[data["metric"] == metric].sort_values("total_arrivals_per_day")
        x = sub["total_arrivals_per_day"].to_numpy(dtype=float)
        mean = sub["mean"].to_numpy(dtype=float)
        low = sub["ci_low_95"].to_numpy(dtype=float)
        high = sub["ci_high_95"].to_numpy(dtype=float)
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
            label=label,
        )
    probability = int(round(100 * SEVERITY_VALUES[severity]))
    behavior_label = "No-show" if behavior == "no_show" else "Balking"
    ax.set_title(f"{behavior_label}: {severity.capitalize()} ({probability}%)")
    ax.set_xlim(min(DEMAND_VALUES) - 4, max(DEMAND_VALUES) + 4)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(DEMAND_VALUES[::2])
    ax.grid(True, alpha=0.23)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )
    for behavior, prefix in [("no_show", "noshow"), ("balking", "balking")]:
        for severity in ["low", "medium", "high"]:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            _plot_one_panel(ax, summary, behavior, severity)
            ax.set_xlabel("Total arrivals per day")
            ax.set_ylabel("Served rate")
            ax.legend(frameon=False)
            slide_number = "09" if behavior == "no_show" else "10"
            fig.savefig(
                figure_dir / f"slide_{slide_number}_{prefix}_{severity}_served_rate_ci.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.1), sharex=True, sharey=True)
        for ax, severity in zip(axes, ["low", "medium", "high"]):
            _plot_one_panel(ax, summary, behavior, severity)
            ax.set_xlabel("Total arrivals per day")
        axes[0].set_ylabel("Served rate")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.015))
        fig.suptitle(
            f"Class served rates across {('no-show' if behavior == 'no_show' else 'balking')} magnitude and demand",
            fontsize=17,
        )
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.20, top=0.82, wspace=0.05)
        slide_number = "09" if behavior == "no_show" else "10"
        fig.savefig(
            figure_dir / f"slide_{slide_number}_{prefix}_three_panel_served_rate_ci.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def analyze_full(results: pd.DataFrame, output_dir: Path) -> None:
    pointwise = pointwise_summary(results)
    paired = paired_summary(results)
    _atomic_csv(results, output_dir / "summary" / "combined_seed_results.csv")
    _atomic_csv(pointwise, output_dir / "summary" / "pointwise_95ci.csv")
    _atomic_csv(paired, output_dir / "summary" / "paired_low_profile_differences_95ci.csv")
    create_figures(pointwise, output_dir)
    print(f"Analyzed {len(results):,} simulations.")
    print(f"Pointwise intervals: {output_dir / 'summary' / 'pointwise_95ci.csv'}")
    print(f"Paired differences: {output_dir / 'summary' / 'paired_low_profile_differences_95ci.csv'}")
    print(f"Figures: {output_dir / 'figures'}")


def analyze(output_dir: Path, study: str) -> None:
    design, results = _load_complete_results(output_dir)
    observed_studies = set(design["study"])
    if observed_studies != {study}:
        raise RuntimeError(f"Design study mismatch: expected {study}, observed {sorted(observed_studies)}")
    if study == "audit":
        analyze_audit(results, output_dir)
    else:
        analyze_full(results, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["design", "run", "analyze", "all"], required=True)
    parser.add_argument("--study", choices=["audit", "full"], required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_seeds = 10 if args.study == "audit" else 100
    n_seeds = args.n_seeds if args.n_seeds is not None else default_seeds
    if args.mode in {"design", "all"}:
        create_design(args.output_dir, args.study, n_seeds)
    if args.mode in {"run", "all"}:
        run_shard(
            args.output_dir,
            args.shard_index,
            args.shard_count,
            resume=not args.no_resume,
            checkpoint_every=args.checkpoint_every,
        )
    if args.mode in {"analyze", "all"}:
        analyze(args.output_dir, args.study)


if __name__ == "__main__":
    main()
