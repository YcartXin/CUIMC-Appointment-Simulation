"""
Class 1 high-delay no-show probability sweep.
Class 2 held at baseline. 100 seeds per parameter value.

Outputs
-------
outputs/class1_no_show/raw/class_results.csv
outputs/class1_no_show/raw/aggregate_results.csv
outputs/class1_no_show/summary/class_summary.csv
outputs/class1_no_show/summary/aggregate_summary.csv
outputs/class1_no_show/figures/percent_serviced_by_class.png
outputs/class1_no_show/figures/average_utilization_aggregate.png
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

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from simulation.config_loader import load_config
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import SimulationConfig, ThresholdRule
from analysis.metrics import aggregate_result_row, class_result_rows
from analysis.plot_style import driver_line_style

CONFIG_PATH = REPO_DIR / "configs" / "baseline.yaml"
OUTPUT_DIR  = REPO_DIR / "outputs" / "class1_no_show"
RAW_DIR     = OUTPUT_DIR / "raw"
SUMMARY_DIR = OUTPUT_DIR / "summary"
FIGURE_DIR  = OUTPUT_DIR / "figures"

CLASS1_HIGH_NOSHOW_VALUES = np.round(np.arange(0.0, 1.0, 0.1), 2)
SEEDS = range(1, 101)


def make_config(base: SimulationConfig, xi_high: float, seed: int) -> SimulationConfig:
    old = base.classes[1].no_show_prob
    if not isinstance(old, ThresholdRule):
        raise TypeError("Expected ThresholdRule for no_show_prob.")
    new_c1 = replace(
        base.classes[1],
        no_show_prob=ThresholdRule(threshold=old.threshold, low=old.low, high=float(xi_high)),
    )
    return replace(base, classes={**base.classes, 1: new_c1}, seed=int(seed))


def run_sweep(xi_values: Iterable[float], seeds: Iterable[int]):
    base = load_config(CONFIG_PATH)
    class_rows, agg_rows = [], []
    for xi in xi_values:
        print(f"  xi_high = {xi:.2f}")
        for seed in seeds:
            result = ClinicAppointmentSimulation(make_config(base, xi, seed)).run()
            agg_rows.append(aggregate_result_row(result, {"class1_xi_high": xi, "seed": seed}))
            class_rows.extend(class_result_rows(result, {"class1_xi_high": xi, "seed": seed}))
    return pd.DataFrame(class_rows), pd.DataFrame(agg_rows)


def summarize(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    s = df.groupby(group_cols)[metric].agg(mean="mean", std="std", n="count").reset_index()
    s["std"] = s["std"].fillna(0.0)
    s["se"]  = s["std"] / np.sqrt(s["n"])
    s["ci95"] = 1.96 * s["se"]
    s["metric"] = metric
    return s


def create_class_summary(class_results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["percent_serviced", "slot_utilization", "no_show", "served",
               "mean_accepted_booking_delay", "mean_offered_booking_delay"]
    return pd.concat(
        [summarize(class_results, ["class1_xi_high", "class_id"], m) for m in metrics],
        ignore_index=True,
    )


def create_aggregate_summary(agg_results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["average_utilization", "overall_percent_serviced",
               "mean_accepted_booking_delay", "mean_offered_booking_delay"]
    return pd.concat(
        [summarize(agg_results, ["class1_xi_high"], m) for m in metrics],
        ignore_index=True,
    )


def plot_overall_and_class(class_summary, agg_summary, class_metric, agg_metric,
                           title, ylabel, output_path, y_limits=None):
    agg_df = agg_summary[agg_summary.metric == agg_metric].sort_values("class1_xi_high")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(agg_df.class1_xi_high, agg_df["mean"], yerr=agg_df.ci95, capsize=3,
                label="overall", **driver_line_style("no_show", "overall", 0))

    cls_df = class_summary[class_summary.metric == class_metric]
    for idx, (cid, sub) in enumerate(cls_df.groupby("class_id"), start=1):
        sub = sub.sort_values("class1_xi_high")
        ax.errorbar(sub.class1_xi_high, sub["mean"], yerr=sub.ci95, capsize=3,
                    label=f"Class {cid}",
                    **driver_line_style("no_show", f"Class {cid}", idx))

    ax.set_title(title)
    ax.set_xlabel("Class 1 high no-show probability")
    ax.set_ylabel(ylabel)
    ax.set_xticks(CLASS1_HIGH_NOSHOW_VALUES)
    if y_limits:
        ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Series", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_figures(class_summary, agg_summary):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_overall_and_class(
        class_summary, agg_summary,
        "percent_serviced", "overall_percent_serviced",
        "Served Rate", "Served rate",
        FIGURE_DIR / "percent_serviced_by_class.png",
    )
    plot_overall_and_class(
        class_summary, agg_summary,
        "slot_utilization", "average_utilization",
        "Average Utilization and Class Slot Shares", "Share of available slots",
        FIGURE_DIR / "average_utilization_aggregate.png",
    )


def main():
    for d in (RAW_DIR, SUMMARY_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("Running Class 1 no-show sweep …")
    class_results, agg_results = run_sweep(CLASS1_HIGH_NOSHOW_VALUES, SEEDS)

    class_results.to_csv(RAW_DIR / "class_results.csv", index=False)
    agg_results.to_csv(RAW_DIR / "aggregate_results.csv", index=False)

    class_summary = create_class_summary(class_results)
    agg_summary   = create_aggregate_summary(agg_results)
    class_summary.to_csv(SUMMARY_DIR / "class_summary.csv", index=False)
    agg_summary.to_csv(SUMMARY_DIR / "aggregate_summary.csv", index=False)

    create_figures(class_summary, agg_summary)

    print(f"Done. Figures → {FIGURE_DIR}")


if __name__ == "__main__":
    main()
