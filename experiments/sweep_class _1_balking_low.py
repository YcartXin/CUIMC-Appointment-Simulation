"""
Class 1 pre-threshold balking rate sweep.

Varies Class 1's *low* (pre-threshold) balking probability from 0.0 to 0.5
while holding the *high* (post-threshold) probability fixed at baseline 0.5.
Class 2 is held at baseline throughout.

Purpose: disentangle whether the effects of balking step come from the
gap between pre- and post-threshold rates, or from the absolute value
of the post-threshold rate.

    - If varying low produces similar metric shifts as the existing
      high sweep → the gap drives the effect.
    - If varying low has little effect → the absolute post-threshold
      value drives the effect.

Produces the same five figures as sweep_class_1_balking.py for direct
comparison.

Outputs
-------
outputs/class1_balking_low/raw/class_results.csv
outputs/class1_balking_low/raw/aggregate_results.csv
outputs/class1_balking_low/summary/class_summary.csv
outputs/class1_balking_low/summary/aggregate_summary.csv
outputs/class1_balking_low/figures/mean_accepted_delay_by_class.png
outputs/class1_balking_low/figures/mean_offered_delay_by_class.png
outputs/class1_balking_low/figures/average_utilization_aggregate.png
outputs/class1_balking_low/figures/percent_serviced_by_class.png
outputs/class1_balking_low/figures/balking_rate_by_class.png
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
from analysis.metrics import aggregate_result_row, class_result_rows
from analysis.plot_style import driver_line_style

# ============================================================
# Experiment settings
# ============================================================

CONFIG_PATH = REPO_DIR / "configs" / "baseline.yaml"

OUTPUT_DIR = REPO_DIR / "outputs" / "class1_balking_low"
RAW_DIR = OUTPUT_DIR / "raw"
SUMMARY_DIR = OUTPUT_DIR / "summary"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Sweep the pre-threshold (low) rate from 0.0 up to 0.5 (matching high).
# At 0.5, there is no step — balking is flat at 0.5 for all delays.
CLASS1_LOW_BALK_VALUES = np.round(np.arange(0.0, 0.55, 0.05), 2)

SEEDS = range(1, 101)

# ============================================================
# Config modification
# ============================================================

def make_config(
    base_config: SimulationConfig,
    class1_low_balk: float,
    seed: int,
) -> SimulationConfig:
    """
    Return a new config where only Class 1's low (pre-threshold)
    balking probability and the seed are changed. The high
    (post-threshold) rate stays at baseline (0.5).
    """
    class1_params = base_config.classes[1]

    if not isinstance(class1_params.balk_prob, ThresholdRule):
        raise TypeError("Expected ThresholdRule for balk_prob")

    old_rule = class1_params.balk_prob

    new_class1_params = replace(
        class1_params,
        balk_prob=ThresholdRule(
            threshold=old_rule.threshold,
            low=float(class1_low_balk),
            high=old_rule.high,
        ),
    )

    new_classes = dict(base_config.classes)
    new_classes[1] = new_class1_params

    return replace(base_config, classes=new_classes, seed=int(seed))


# ============================================================
# Run sweep
# ============================================================

def run_sweep(
    low_balk_values: Iterable[float],
    seeds: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    base_config = load_config(CONFIG_PATH)

    class_rows = []
    aggregate_rows = []

    for low_balk in low_balk_values:
        print(f"  low_balk = {low_balk:.2f}")

        for seed in seeds:
            config = make_config(base_config, low_balk, seed)
            sim = ClinicAppointmentSimulation(config)
            results = sim.run()

            aggregate_rows.append(
                aggregate_result_row(
                    results,
                    {"class1_low_balk": low_balk, "seed": seed},
                )
            )
            class_rows.extend(
                class_result_rows(
                    results,
                    {"class1_low_balk": low_balk, "seed": seed},
                )
            )

    return pd.DataFrame(class_rows), pd.DataFrame(aggregate_rows)


# ============================================================
# Aggregation helpers
# ============================================================

def summarize_metric(
    df: pd.DataFrame,
    group_cols: list[str],
    metric: str,
) -> pd.DataFrame:
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
    return pd.concat(
        [
            summarize_metric(class_results, ["class1_low_balk", "class_id"], m)
            for m in metrics
        ],
        ignore_index=True,
    )


def create_aggregate_summary(aggregate_results: pd.DataFrame) -> pd.DataFrame:
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
    return pd.concat(
        [
            summarize_metric(aggregate_results, ["class1_low_balk"], m)
            for m in metrics
        ],
        ignore_index=True,
    )


# ============================================================
# Plotting helpers
# ============================================================

X_COL = "class1_low_balk"
X_LABEL = "Class 1 pre-threshold balking probability"


def plot_overall_and_class(
    class_summary: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
    class_metric: str,
    aggregate_metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:

    agg = aggregate_summary[aggregate_summary["metric"] == aggregate_metric].copy()
    agg = agg.sort_values(X_COL)

    cls = class_summary[class_summary["metric"] == class_metric].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        agg[X_COL], agg["mean"], yerr=agg["ci95"], capsize=3,
        label="overall", **driver_line_style("balking", "overall", 0),
    )

    for idx, (cid, sub) in enumerate(cls.groupby("class_id"), start=1):
        sub = sub.sort_values(X_COL)
        ax.errorbar(
            sub[X_COL], sub["mean"], yerr=sub["ci95"], capsize=3,
            label=f"Class {cid}",
            **driver_line_style("balking", f"Class {cid}", idx),
        )

    ax.set_title(title)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_xticks(CLASS1_LOW_BALK_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="Series", frameon=False,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_figures(
    class_summary: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
) -> None:

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_overall_and_class(
        class_summary, aggregate_summary,
        "mean_accepted_booking_delay", "mean_accepted_booking_delay",
        "Mean Accepted Booking Delay", "Mean accepted booking delay",
        FIGURE_DIR / "mean_accepted_delay_by_class.png",
    )
    plot_overall_and_class(
        class_summary, aggregate_summary,
        "mean_offered_booking_delay", "mean_offered_booking_delay",
        "Mean Offered Booking Delay", "Mean offered booking delay (days)",
        FIGURE_DIR / "mean_offered_delay_by_class.png",
    )
    plot_overall_and_class(
        class_summary, aggregate_summary,
        "slot_utilization", "average_utilization",
        "Average Utilization and Class Slot Shares",
        "Completed visits as share of total slots",
        FIGURE_DIR / "average_utilization_aggregate.png",
    )
    plot_overall_and_class(
        class_summary, aggregate_summary,
        "percent_serviced", "overall_percent_serviced",
        "Served Rate", "Served rate",
        FIGURE_DIR / "percent_serviced_by_class.png",
    )
    plot_overall_and_class(
        class_summary, aggregate_summary,
        "balking_rate", "overall_balking_rate",
        "Balking Rate Among Offered Patients", "Balked / offered",
        FIGURE_DIR / "balking_rate_by_class.png",
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Running Class 1 pre-threshold balking sweep …")
    class_results, aggregate_results = run_sweep(CLASS1_LOW_BALK_VALUES, SEEDS)

    class_results.to_csv(RAW_DIR / "class_results.csv", index=False)
    aggregate_results.to_csv(RAW_DIR / "aggregate_results.csv", index=False)

    class_summary = create_class_summary(class_results)
    aggregate_summary = create_aggregate_summary(aggregate_results)

    class_summary.to_csv(SUMMARY_DIR / "class_summary.csv", index=False)
    aggregate_summary.to_csv(SUMMARY_DIR / "aggregate_summary.csv", index=False)

    create_figures(class_summary, aggregate_summary)

    print(f"\nDone. Figures → {FIGURE_DIR}")


if __name__ == "__main__":
    main()