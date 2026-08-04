"""
H9 no-show delay-exposure sweep.

Hypothesis:
    No-shows are unrebookable service-day losses, but their utilization
    impact depends on delay exposure.

Test:
    Vary Class 1's no-show threshold and high no-show probability.
    A lower threshold exposes more accepted bookings to high no-show risk.
    If H9 is correct, utilization should fall more steeply when exposure
    is higher.

Outputs
-------
outputs/h9_no_show_exposure/raw/results.csv
outputs/h9_no_show_exposure/summary/summary.csv
outputs/h9_no_show_exposure/figures/no_show_exposure_utilization.png
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
from simulation.model import SimulationConfig, ThresholdRule
from analysis.metrics import aggregate_result_row
from analysis.plot_style import driver_line_style


# ============================================================
# Experiment settings
# ============================================================

CONFIG_PATH = REPO_DIR / "configs" / "baseline.yaml"

OUTPUT_DIR = REPO_DIR / "outputs" / "h9_no_show_exposure"
RAW_DIR = OUTPUT_DIR / "raw"
SUMMARY_DIR = OUTPUT_DIR / "summary"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Lower threshold = more accepted bookings exposed to high no-show risk.
NO_SHOW_THRESHOLDS = [3, 6, 9, 12]

CLASS1_HIGH_NOSHOW_VALUES = np.round(np.arange(0.0, 1.0, 0.1), 2)

# Use 50 for a good runtime/precision compromise.
SEEDS = range(1, 51)


# ============================================================
# Config modification
# ============================================================

def make_config(
    base_config: SimulationConfig,
    class1_no_show_threshold: int,
    class1_high_no_show: float,
    seed: int,
) -> SimulationConfig:
    """
    Return a config where only Class 1's no-show threshold,
    Class 1's high no-show probability, and the seed are changed.
    """

    old_rule = base_config.classes[1].no_show_prob

    if not isinstance(old_rule, ThresholdRule):
        raise TypeError("Expected Class 1 no_show_prob to be a ThresholdRule.")

    new_rule = ThresholdRule(
        threshold=int(class1_no_show_threshold),
        low=old_rule.low,
        high=float(class1_high_no_show),
    )

    new_class1 = replace(
        base_config.classes[1],
        no_show_prob=new_rule,
    )

    new_classes = dict(base_config.classes)
    new_classes[1] = new_class1

    return replace(
        base_config,
        classes=new_classes,
        seed=int(seed),
    )


# ============================================================
# Run sweep
# ============================================================

def run_sweep(
    thresholds: Iterable[int],
    high_values: Iterable[float],
    seeds: Iterable[int],
) -> pd.DataFrame:
    """
    Run the no-show delay-exposure sweep.

    Returns one row per threshold, high no-show value, and seed.
    """

    base_config = load_config(CONFIG_PATH)
    rows = []

    total_slots = base_config.slots_per_day * base_config.measure_days

    for threshold in thresholds:
        for high_value in high_values:
            print(
                f"Running Class 1 no-show threshold = {threshold}, "
                f"high no-show = {high_value:.2f}"
            )

            for seed in seeds:
                config = make_config(
                    base_config=base_config,
                    class1_no_show_threshold=threshold,
                    class1_high_no_show=high_value,
                    seed=seed,
                )

                result = ClinicAppointmentSimulation(config).run()

                aggregate = aggregate_result_row(
                    result,
                    {
                        "class1_no_show_threshold": threshold,
                        "class1_high_no_show": high_value,
                        "seed": seed,
                    },
                )

                class1_metrics = result.class_metrics[1]

                accepted_delay_counts = class1_metrics.accepted_delay_counts
                total_accepted = sum(accepted_delay_counts.values())
                exposed_accepted = sum(
                    count
                    for tau, count in accepted_delay_counts.items()
                    if tau > threshold
                )

                high_delay_exposure_share = (
                    exposed_accepted / total_accepted
                    if total_accepted > 0
                    else 0.0
                )

                class1_no_show_slot_loss = class1_metrics.no_show / total_slots

                class1_served_rate = (
                    class1_metrics.served / class1_metrics.arrivals
                    if class1_metrics.arrivals > 0
                    else 0.0
                )

                rows.append(
                    {
                        **aggregate,
                        "class1_no_show_threshold": threshold,
                        "class1_high_no_show": high_value,
                        "seed": seed,
                        "class1_high_delay_exposure_share": high_delay_exposure_share,
                        "class1_no_show_slot_loss": class1_no_show_slot_loss,
                        "class1_served_rate": class1_served_rate,
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# Summarize
# ============================================================

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize each threshold × high-no-show setting across seeds.
    """

    metrics = [
        "average_utilization",
        "overall_percent_serviced",
        "class1_high_delay_exposure_share",
        "class1_no_show_slot_loss",
        "class1_served_rate",
    ]

    rows = []

    for metric in metrics:
        summary = (
            df.groupby(["class1_no_show_threshold", "class1_high_no_show"])[metric]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
        )

        summary["std"] = summary["std"].fillna(0.0)
        summary["se"] = summary["std"] / np.sqrt(summary["n"])
        summary["ci95"] = 1.96 * summary["se"]
        summary["metric"] = metric

        rows.append(summary)

    return pd.concat(rows, ignore_index=True)


# ============================================================
# Plot
# ============================================================

def create_figure(summary: pd.DataFrame) -> None:
    """
    Create a two-panel figure for H9.

    Left:
        Average utilization response to high no-show probability.

    Right:
        Share of Class 1 accepted bookings exposed to high no-show risk.
    """

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    util = summary[summary["metric"] == "average_utilization"].copy()
    exposure = summary[
        summary["metric"] == "class1_high_delay_exposure_share"
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # --------------------------------------------------------
    # Panel 1: utilization response
    # --------------------------------------------------------

    ax = axes[0]

    for idx, (threshold, sub) in enumerate(
        util.groupby("class1_no_show_threshold")
    ):
        sub = sub.sort_values("class1_high_no_show")

        ax.errorbar(
            sub["class1_high_no_show"],
            sub["mean"],
            yerr=sub["ci95"],
            capsize=3,
            label=f"threshold = {threshold}",
            **driver_line_style("no_show", f"threshold {threshold}", idx),
        )

    ax.set_title("Utilization Loss Depends on No-Show Exposure")
    ax.set_xlabel("Class 1 high no-show probability")
    ax.set_ylabel("Average utilization")
    ax.set_xticks(CLASS1_HIGH_NOSHOW_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Class 1 no-show threshold", frameon=False)

    # --------------------------------------------------------
    # Panel 2: exposure by threshold
    # --------------------------------------------------------

    ax = axes[1]

    exposure_by_threshold = (
        exposure.groupby("class1_no_show_threshold")["mean"]
        .mean()
        .reset_index()
        .sort_values("class1_no_show_threshold")
    )

    ax.bar(
        exposure_by_threshold["class1_no_show_threshold"].astype(str),
        exposure_by_threshold["mean"],
        alpha=0.8,
    )

    ax.set_title("Accepted Bookings Exposed to High No-Show Risk")
    ax.set_xlabel("Class 1 no-show threshold")
    ax.set_ylabel("Share of Class 1 accepted bookings with tau > threshold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(
        FIGURE_DIR / "no_show_exposure_utilization.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    results = run_sweep(
        thresholds=NO_SHOW_THRESHOLDS,
        high_values=CLASS1_HIGH_NOSHOW_VALUES,
        seeds=SEEDS,
    )

    results.to_csv(RAW_DIR / "results.csv", index=False)

    summary = summarize(results)
    summary.to_csv(SUMMARY_DIR / "summary.csv", index=False)

    create_figure(summary)

    print("\nDone.")
    print(f"Raw results saved to: {RAW_DIR}")
    print(f"Summary saved to:     {SUMMARY_DIR}")
    print(f"Figure saved to:      {FIGURE_DIR}")


if __name__ == "__main__":
    main()