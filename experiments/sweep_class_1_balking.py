from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
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

BALKING_COLOR = "#9467bd"


def blend_color(color: str, target: str = "#ffffff", amount: float = 0.5) -> tuple[float, float, float]:
    base_rgb = np.array(to_rgb(color))
    target_rgb = np.array(to_rgb(target))
    return tuple((1.0 - amount) * base_rgb + amount * target_rgb)


def series_style(label: str, index: int) -> dict[str, object]:
    role = str(label).lower()
    if "class 1" in role:
        return {
            "color": blend_color(BALKING_COLOR, "#000000", 0.10),
            "linestyle": "--",
            "marker": "s",
            "linewidth": 2.2,
        }
    if "class 2" in role:
        return {
            "color": blend_color(BALKING_COLOR, amount=0.24),
            "linestyle": ":",
            "marker": "^",
            "linewidth": 2.2,
        }
    if index == 1:
        return {
            "color": blend_color(BALKING_COLOR, "#000000", 0.10),
            "linestyle": "--",
            "marker": "s",
            "linewidth": 2.2,
        }
    if index == 2:
        return {
            "color": blend_color(BALKING_COLOR, amount=0.24),
            "linestyle": ":",
            "marker": "^",
            "linewidth": 2.2,
        }
    return {
        "color": BALKING_COLOR,
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.5,
    }

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

            class_metrics = results.class_metrics
            total_arrivals = sum(metrics.arrivals for metrics in class_metrics.values())
            total_booked = sum(metrics.booked for metrics in class_metrics.values())
            total_offered = sum(metrics.offered for metrics in class_metrics.values())
            total_balked = sum(metrics.balked for metrics in class_metrics.values())
            total_accepted_delay = sum(
                metrics.total_booking_delay for metrics in class_metrics.values()
            )
            total_offered_delay = sum(
                metrics.total_offered_booking_delay
                for metrics in class_metrics.values()
            )

            aggregate_rows.append(
                {
                    "class1_high_balk": class1_high_balk,
                    "seed": seed,
                    "average_utilization": results.average_utilization,
                    "overall_percent_serviced": results.overall_percent_serviced,
                    "total_served": results.total_served,
                    "mean_accepted_booking_delay": (
                        total_accepted_delay / total_booked if total_booked else 0.0
                    ),
                    "mean_offered_booking_delay": (
                        total_offered_delay / total_offered if total_offered else 0.0
                    ),
                    "overall_balking_rate": (
                        total_balked / total_offered if total_offered else 0.0
                    ),
                    "total_arrivals": total_arrivals,
                    "total_booked": total_booked,
                    "total_offered": total_offered,
                    "total_balked": total_balked,
                }
            )

            for class_id, metrics in class_metrics.items():
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
                        "slot_utilization": (
                            metrics.served / results.total_slots
                            if results.total_slots
                            else 0.0
                        ),
                        "balking_rate": (
                            metrics.balked / metrics.offered
                            if metrics.offered
                            else 0.0
                        ),
                        "total_booking_delay": metrics.total_booking_delay,
                        "total_offered_booking_delay": (
                            metrics.total_offered_booking_delay
                        ),
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
        "slot_utilization",
        "balking_rate",
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
        "mean_accepted_booking_delay",
        "mean_offered_booking_delay",
        "overall_balking_rate",
        "total_served",
        "total_arrivals",
        "total_booked",
        "total_offered",
        "total_balked",
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

def plot_overall_and_class_metric(
    class_summary: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
    class_metric: str,
    aggregate_metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    y_limits: tuple[float, float] | None = None,
) -> None:
    """
    Plot one metric with overall, Class 1, and Class 2 values.
    """

    aggregate_df = aggregate_summary[
        aggregate_summary["metric"] == aggregate_metric
    ].copy()
    aggregate_df = aggregate_df.sort_values("class1_high_balk")

    class_df = class_summary[class_summary["metric"] == class_metric].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        aggregate_df["class1_high_balk"],
        aggregate_df["mean"],
        yerr=aggregate_df["ci95"],
        capsize=3,
        label="overall",
        **series_style("overall", 0),
    )

    for index, (class_id, sub) in enumerate(class_df.groupby("class_id"), start=1):
        sub = sub.sort_values("class1_high_balk")

        ax.errorbar(
            sub["class1_high_balk"],
            sub["mean"],
            yerr=sub["ci95"],
            capsize=3,
            label=f"Class {class_id}",
            **series_style(f"Class {class_id}", index),
        )

    ax.set_title(title)
    ax.set_xlabel("Class 1 high balking probability")
    ax.set_ylabel(ylabel)
    ax.set_xticks(CLASS1_HIGH_BALK_VALUES)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="Series",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_figures(
    class_summary: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
) -> None:
    """
    Create one figure per reported metric:

    1. Mean accepted booking delay
    2. Mean offered booking delay
    3. Average utilization
    4. Percent serviced
    5. Balking rate
    """

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_overall_and_class_metric(
        class_summary=class_summary,
        aggregate_summary=aggregate_summary,
        class_metric="mean_accepted_booking_delay",
        aggregate_metric="mean_accepted_booking_delay",
        title="Mean Accepted Booking Delay",
        ylabel="Mean accepted booking delay",
        output_path=FIGURE_DIR / "mean_accepted_delay_by_class.png",
    )

    plot_overall_and_class_metric(
        class_summary=class_summary,
        aggregate_summary=aggregate_summary,
        class_metric="mean_offered_booking_delay",
        aggregate_metric="mean_offered_booking_delay",
        title="Mean Offered Booking Delay",
        ylabel="Mean offered booking delay",
        output_path=FIGURE_DIR / "mean_offered_delay_by_class.png",
    )

    plot_overall_and_class_metric(
        class_summary=class_summary,
        aggregate_summary=aggregate_summary,
        class_metric="slot_utilization",
        aggregate_metric="average_utilization",
        title="Average Utilization and Class Slot Shares",
        ylabel="Share of available slots",
        output_path=FIGURE_DIR / "average_utilization_aggregate.png",
    )

    plot_overall_and_class_metric(
        class_summary=class_summary,
        aggregate_summary=aggregate_summary,
        class_metric="percent_serviced",
        aggregate_metric="overall_percent_serviced",
        title="Served Rate",
        ylabel="Served rate",
        output_path=FIGURE_DIR / "percent_serviced_by_class.png",
    )

    plot_overall_and_class_metric(
        class_summary=class_summary,
        aggregate_summary=aggregate_summary,
        class_metric="balking_rate",
        aggregate_metric="overall_balking_rate",
        title="Balking Rate Among Offered Patients",
        ylabel="Balked / offered",
        output_path=FIGURE_DIR / "balking_rate_by_class.png",
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
