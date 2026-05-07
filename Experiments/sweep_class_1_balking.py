from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Path setup
# ============================================================

# This file should live in:
# CUIMC-Appointment-Simulation/experiments/sweep_class1_balking.py

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from engine_files.config_loader import load_config
from engine_files.engine import ClinicAppointmentSimulation
from engine_files.model import SimulationConfig, ThresholdRule


# ============================================================
# Experiment settings
# ============================================================

CONFIG_PATH = REPO_DIR / "configs" / "baseline.yaml"

OUTPUT_DIR = REPO_DIR / "outputs" / "class1_balking"
RAW_DIR = OUTPUT_DIR / "raw"
SUMMARY_DIR = OUTPUT_DIR / "summary"
FIGURE_DIR = OUTPUT_DIR / "figures"

CLASS1_HIGH_BALK_VALUES = np.round(np.arange(0.0, 1.0, 0.1), 2)

# 100 replications per parameter value.
SEEDS = range(1, 101)


# ============================================================
# Config modification
# ============================================================

def make_class1_balking_config(
    base_config: SimulationConfig,
    class1_high_balk: float,
    seed: int,
) -> SimulationConfig:
    """
    Return a new config where only Class 1's high balking probability
    and the random seed are changed.

    All other parameters, including Class 2's parameters, remain unchanged.
    """

    class1_params = base_config.classes[1]

    if not isinstance(class1_params.balk_prob, ThresholdRule):
        raise TypeError(
            "This sweep expects Class 1 balk_prob to be a ThresholdRule."
        )

    old_rule = class1_params.balk_prob

    new_class1_params = replace(
        class1_params,
        balk_prob=ThresholdRule(
            threshold=old_rule.threshold,
            low=old_rule.low,
            high=float(class1_high_balk),
        ),
    )

    new_classes = dict(base_config.classes)
    new_classes[1] = new_class1_params

    return replace(
        base_config,
        classes=new_classes,
        seed=int(seed),
    )


# ============================================================
# Run sweep
# ============================================================

def run_sweep(
    class1_high_balk_values: Iterable[float],
    seeds: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the Class 1 high-balking sweep.

    Returns:
        class_results:
            one row per class per simulation run

        aggregate_results:
            one row per simulation run
    """

    base_config = load_config(CONFIG_PATH)

    class_rows = []
    aggregate_rows = []

    for class1_high_balk in class1_high_balk_values:
        print(f"Running Class 1 high balking = {class1_high_balk:.2f}")

        for seed in seeds:
            config = make_class1_balking_config(
                base_config=base_config,
                class1_high_balk=class1_high_balk,
                seed=seed,
            )

            sim = ClinicAppointmentSimulation(config)
            results = sim.run()

            aggregate_rows.append(
                {
                    "class1_high_balk": class1_high_balk,
                    "seed": seed,
                    "average_utilization": results.average_utilization,
                    "overall_percent_serviced": results.overall_percent_serviced,
                    "total_served": results.total_served,
                }
            )

            for class_id, metrics in results.class_metrics.items():
                class_rows.append(
                    {
                        "class1_high_balk": class1_high_balk,
                        "seed": seed,
                        "class_id": class_id,
                        "arrivals": metrics.arrivals,
                        "booked": metrics.booked,
                        "balked": metrics.balked,
                        "offered": metrics.offered,
                        "no_offer": metrics.no_offer,
                        "canceled": metrics.canceled,
                        "no_show": metrics.no_show,
                        "served": metrics.served,
                        "mean_accepted_booking_delay": (
                            metrics.mean_accepted_booking_delay
                        ),
                        "mean_offered_booking_delay": (
                            metrics.mean_offered_booking_delay
                        ),
                        "percent_serviced": metrics.percent_serviced,
                    }
                )

    class_results = pd.DataFrame(class_rows)
    aggregate_results = pd.DataFrame(aggregate_rows)

    return class_results, aggregate_results


# ============================================================
# Aggregation helpers
# ============================================================

def summarize_metric(
    df: pd.DataFrame,
    group_cols: list[str],
    metric: str,
) -> pd.DataFrame:
    """
    Compute mean, standard deviation, standard error, and 95% CI
    for one metric.
    """

    summary = (
        df.groupby(group_cols)[metric]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    summary["std"] = summary["std"].fillna(0.0)
    summary["se"] = summary["std"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["se"]
    summary["metric"] = metric

    return summary


def create_class_summary(class_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate class-level metrics by Class 1 high balking probability
    and patient class.
    """

    metrics = [
        "mean_accepted_booking_delay",
        "mean_offered_booking_delay",
        "percent_serviced",
        "arrivals",
        "booked",
        "balked",
        "offered",
        "no_offer",
        "canceled",
        "no_show",
        "served",
    ]

    summaries = [
        summarize_metric(
            df=class_results,
            group_cols=["class1_high_balk", "class_id"],
            metric=metric,
        )
        for metric in metrics
    ]

    return pd.concat(summaries, ignore_index=True)


def create_aggregate_summary(aggregate_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate system-level metrics by Class 1 high balking probability.
    """

    metrics = [
        "average_utilization",
        "overall_percent_serviced",
        "total_served",
    ]

    summaries = [
        summarize_metric(
            df=aggregate_results,
            group_cols=["class1_high_balk"],
            metric=metric,
        )
        for metric in metrics
    ]

    return pd.concat(summaries, ignore_index=True)


# ============================================================
# Plotting helpers
# ============================================================

def plot_class_metric(
    class_summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot one class-level metric with separate lines for each class.
    """

    plot_df = class_summary[class_summary["metric"] == metric].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    for class_id, sub in plot_df.groupby("class_id"):
        sub = sub.sort_values("class1_high_balk")

        ax.errorbar(
            sub["class1_high_balk"],
            sub["mean"],
            yerr=sub["ci95"],
            marker="o",
            capsize=3,
            label=f"Class {class_id}",
        )

    ax.set_title(title)
    ax.set_xlabel("Class 1 high balking probability")
    ax.set_ylabel(ylabel)
    ax.set_xticks(CLASS1_HIGH_BALK_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Patient class")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_aggregate_metric(
    aggregate_summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot one aggregate metric.
    """

    plot_df = aggregate_summary[aggregate_summary["metric"] == metric].copy()
    plot_df = plot_df.sort_values("class1_high_balk")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        plot_df["class1_high_balk"],
        plot_df["mean"],
        yerr=plot_df["ci95"],
        marker="o",
        capsize=3,
    )

    ax.set_title(title)
    ax.set_xlabel("Class 1 high balking probability")
    ax.set_ylabel(ylabel)
    ax.set_xticks(CLASS1_HIGH_BALK_VALUES)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def create_figures(
    class_summary: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
) -> None:
    """
    Create the four requested figures:

    1. Mean accepted booking delay by class
    2. Mean offered booking delay by class
    3. Aggregate average utilization
    4. Percent serviced by class
    """

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_class_metric(
        class_summary=class_summary,
        metric="mean_accepted_booking_delay",
        title="Mean Accepted Booking Delay by Class",
        ylabel="Mean accepted booking delay",
        output_path=FIGURE_DIR / "mean_accepted_delay_by_class.png",
    )

    plot_class_metric(
        class_summary=class_summary,
        metric="mean_offered_booking_delay",
        title="Mean Offered Booking Delay by Class",
        ylabel="Mean offered booking delay",
        output_path=FIGURE_DIR / "mean_offered_delay_by_class.png",
    )

    plot_aggregate_metric(
        aggregate_summary=aggregate_summary,
        metric="average_utilization",
        title="Aggregate Average Utilization",
        ylabel="Average utilization",
        output_path=FIGURE_DIR / "average_utilization_aggregate.png",
    )

    plot_class_metric(
        class_summary=class_summary,
        metric="percent_serviced",
        title="Percent Serviced by Class",
        ylabel="Percent serviced",
        output_path=FIGURE_DIR / "percent_serviced_by_class.png",
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    class_results, aggregate_results = run_sweep(
        class1_high_balk_values=CLASS1_HIGH_BALK_VALUES,
        seeds=SEEDS,
    )

    class_summary = create_class_summary(class_results)
    aggregate_summary = create_aggregate_summary(aggregate_results)

    class_results.to_csv(RAW_DIR / "class_results.csv", index=False)
    aggregate_results.to_csv(RAW_DIR / "aggregate_results.csv", index=False)

    class_summary.to_csv(SUMMARY_DIR / "class_summary.csv", index=False)
    aggregate_summary.to_csv(SUMMARY_DIR / "aggregate_summary.csv", index=False)

    create_figures(
        class_summary=class_summary,
        aggregate_summary=aggregate_summary,
    )

    print("\nDone.")
    print(f"Raw outputs saved to:     {RAW_DIR}")
    print(f"Summary outputs saved to: {SUMMARY_DIR}")
    print(f"Figures saved to:         {FIGURE_DIR}")


if __name__ == "__main__":
    main()