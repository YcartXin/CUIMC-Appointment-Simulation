"""
Class 1 cancellation probability sweep.
Class 2 held at baseline (cancel_prob = 0.10). 100 seeds per parameter value.
 
Hypothesis H1:
    A class with higher cancellation probability shortens offered delay
    for everyone but lowers its own served rate.
 
Test:
    - H1 stands if offered delay drops for both classes, Class 1 served
      rate drops, and utilization stays flat or rises.
    - H1 fails if offered delay does not decrease for both classes.
 
Outputs
-------
outputs/class1_cancellation/raw/class_results.csv
outputs/class1_cancellation/raw/aggregate_results.csv
outputs/class1_cancellation/summary/class_summary.csv
outputs/class1_cancellation/summary/aggregate_summary.csv
outputs/class1_cancellation/figures/mean_offered_delay_by_class.png
outputs/class1_cancellation/figures/percent_serviced_by_class.png
outputs/class1_cancellation/figures/average_utilization_aggregate.png
"""
from __future__ import annotations
 
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
 
# ============================================================
# Path setup
# ============================================================
 
REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
 
from simulation.config_loader import load_config
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import SimulationConfig
from analysis.metrics import aggregate_result_row, class_result_rows
from analysis.plot_style import driver_line_style
 
 
# ============================================================
# Experiment settings
# ============================================================
 
CONFIG_PATH = REPO_DIR / "configs" / "baseline.yaml"
 
OUTPUT_DIR = REPO_DIR / "outputs" / "class1_cancellation"
RAW_DIR = OUTPUT_DIR / "raw"
SUMMARY_DIR = OUTPUT_DIR / "summary"
FIGURE_DIR = OUTPUT_DIR / "figures"
 
CLASS1_CANCEL_VALUES = np.round(np.arange(0.0, 0.35, 0.05), 2)
 
# 100 replications per parameter value.
SEEDS = range(1, 101)
 
 
# ============================================================
# Config modification
# ============================================================
 
def make_class1_cancellation_config(
    base_config: SimulationConfig,
    class1_cancel_prob: float,
    seed: int,
) -> SimulationConfig:
    """
    Return a new config where only Class 1's cancellation probability
    and the random seed are changed.
 
    All other parameters, including Class 2's parameters, remain unchanged.
    """
 
    class1_params = base_config.classes[1]
 
    new_class1_params = replace(
        class1_params,
        cancel_prob=float(class1_cancel_prob),
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
    class1_cancel_values: Iterable[float],
    seeds: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the Class 1 cancellation probability sweep.
 
    Returns:
        class_results:
            one row per class per simulation run
 
        aggregate_results:
            one row per simulation run
    """
 
    base_config = load_config(CONFIG_PATH)
 
    class_rows = []
    aggregate_rows = []
 
    for class1_cancel_prob in class1_cancel_values:
        print(f"Running Class 1 cancel_prob = {class1_cancel_prob:.2f}")
 
        for seed in seeds:
            config = make_class1_cancellation_config(
                base_config=base_config,
                class1_cancel_prob=class1_cancel_prob,
                seed=seed,
            )
 
            sim = ClinicAppointmentSimulation(config)
            results = sim.run()
 
            aggregate_rows.append(
                aggregate_result_row(
                    results,
                    {
                        "class1_cancel_prob": class1_cancel_prob,
                        "seed": seed,
                    },
                )
            )
 
            class_rows.extend(
                class_result_rows(
                    results,
                    {
                        "class1_cancel_prob": class1_cancel_prob,
                        "seed": seed,
                    },
                )
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
    Aggregate class-level metrics by Class 1 cancellation probability
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
            group_cols=["class1_cancel_prob", "class_id"],
            metric=metric,
        )
        for metric in metrics
    ]
 
    return pd.concat(summaries, ignore_index=True)
 
 
def create_aggregate_summary(aggregate_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate system-level metrics by Class 1 cancellation probability.
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
            group_cols=["class1_cancel_prob"],
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
    aggregate_df = aggregate_df.sort_values("class1_cancel_prob")
 
    class_df = class_summary[class_summary["metric"] == class_metric].copy()
 
    fig, ax = plt.subplots(figsize=(8, 5))
 
    ax.errorbar(
        aggregate_df["class1_cancel_prob"],
        aggregate_df["mean"],
        yerr=aggregate_df["ci95"],
        capsize=3,
        label="overall",
        **driver_line_style("cancellation", "overall", 0),
    )
 
    for index, (class_id, sub) in enumerate(class_df.groupby("class_id"), start=1):
        sub = sub.sort_values("class1_cancel_prob")
 
        ax.errorbar(
            sub["class1_cancel_prob"],
            sub["mean"],
            yerr=sub["ci95"],
            capsize=3,
            label=f"Class {class_id}",
            **driver_line_style("cancellation", f"Class {class_id}", index),
        )
 
    ax.set_title(title)
    ax.set_xlabel("Class 1 cancellation probability")
    ax.set_ylabel(ylabel)
    ax.set_xticks(CLASS1_CANCEL_VALUES)
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
 
    1. Mean offered booking delay (by class) — tests delay-shortening claim
    2. Percent serviced (by class) — tests served-rate-harm claim
    3. Average utilization (aggregate) — tiebreaker for system-level effect
    """
 
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
 
    plot_overall_and_class_metric(
        class_summary=class_summary,
        aggregate_summary=aggregate_summary,
        class_metric="mean_offered_booking_delay",
        aggregate_metric="mean_offered_booking_delay",
        title="Mean Offered Booking Delay",
        ylabel="Mean offered booking delay (days)",
        output_path=FIGURE_DIR / "mean_offered_delay_by_class.png",
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
        class_metric="slot_utilization",
        aggregate_metric="average_utilization",
        title="Average Utilization and Class Slot Shares",
        ylabel="Completed visits as share of total slots",
        output_path=FIGURE_DIR / "average_utilization_aggregate.png",
    )
 
 
# ============================================================
# Main
# ============================================================
 
def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
 
    class_results, aggregate_results = run_sweep(
        class1_cancel_values=CLASS1_CANCEL_VALUES,
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