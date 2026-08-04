"""
Experiment 3: Booking Horizon Sweep

Tests whether a longer booking horizon amplifies the class access gap through
cancellation compounding. Under pooled FCFS, a longer horizon gives more time
for cancellations to free slots — disproportionately benefiting the class with
lower cancellation rates (Class 1 in the realistic scenario).

All parameters fixed at realistic.yaml; only horizon_days is varied.

horizon_days ∈ {7, 14, 21, 28, 42}  ×  30 seeds = 150 runs.

Outputs
-------
outputs/exp3_horizon_sweep/raw/results.csv
outputs/exp3_horizon_sweep/summary/summary.csv
outputs/exp3_horizon_sweep/figures/class_access_gap.png
outputs/exp3_horizon_sweep/figures/metrics_by_horizon.png
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from simulation.config_loader import load_config
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import SimulationConfig
from analysis.metrics import aggregate_result_row
from analysis.plot_style import (
    CLASS_1_COLOR, CLASS_2_COLOR, BASELINE_COLOR,
    UTILIZATION_COLOR, ACCESS_COLOR,
)

CONFIG_PATH = REPO_DIR / "configs" / "realistic.yaml"
OUTPUT_DIR  = REPO_DIR / "outputs" / "exp3_horizon_sweep"
RAW_DIR     = OUTPUT_DIR / "raw"
SUMMARY_DIR = OUTPUT_DIR / "summary"
FIGURE_DIR  = OUTPUT_DIR / "figures"

SEEDS         = range(1, 31)
HORIZON_VALUES = [7, 14, 21, 28, 42]


# ---------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------

def make_config(base: SimulationConfig, horizon: int, seed: int) -> SimulationConfig:
    return replace(base, horizon_days=int(horizon), seed=int(seed))


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------

def run_experiment() -> pd.DataFrame:
    base = load_config(CONFIG_PATH)
    rows = []
    for horizon in HORIZON_VALUES:
        print(f"  horizon = {horizon} days")
        for seed in SEEDS:
            cfg = make_config(base, horizon, seed)
            result = ClinicAppointmentSimulation(cfg).run()
            row = aggregate_result_row(result, {"horizon_days": horizon, "seed": seed})
            cm = result.class_metrics
            row["class_1_served_rate"] = cm[1].percent_serviced
            row["class_2_served_rate"] = cm[2].percent_serviced
            row["access_gap"]          = cm[1].percent_serviced - cm[2].percent_serviced
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "average_utilization", "overall_percent_serviced",
        "class_1_served_rate", "class_2_served_rate", "access_gap",
        "mean_offered_booking_delay",
    ]
    records = []
    for h, sub in df.groupby("horizon_days"):
        for m in metrics:
            se = sub[m].std() / np.sqrt(len(sub))
            records.append({
                "horizon_days": h,
                "metric": m,
                "mean":   sub[m].mean(),
                "std":    sub[m].std(),
                "se":     se,
                "ci95":   1.96 * se,
                "n":      len(sub),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------

def _pull(summary: pd.DataFrame, metric: str) -> tuple[list, list, list]:
    sub = summary[summary["metric"] == metric].sort_values("horizon_days")
    return sub["horizon_days"].tolist(), sub["mean"].tolist(), sub["ci95"].tolist()


def plot_class_access_gap(summary: pd.DataFrame, path: Path) -> None:
    x, means, cis = _pull(summary, "access_gap")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(x, means, yerr=cis, fmt="o-", capsize=4, color=CLASS_1_COLOR,
                linewidth=2, markersize=6, elinewidth=1.4)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Booking horizon (days)")
    ax.set_ylabel("Class 1 served rate − Class 2 served rate")
    ax.set_title(
        "Class Access Gap vs Booking Horizon\n"
        "λ=24, S=20, realistic behavioral params · 30 seeds · ±95% CI",
        fontsize=10,
    )
    ax.set_xticks(x)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_by_horizon(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, metric, ylabel, color, title in [
        (axes[0], "average_utilization",    "Utilization",         UTILIZATION_COLOR, "Utilization"),
        (axes[1], "overall_percent_serviced","Overall served rate", ACCESS_COLOR,      "Served Rate"),
        (axes[2], "access_gap",             "Gap (C1 − C2)",       CLASS_1_COLOR,     "Class Access Gap"),
    ]:
        x, means, cis = _pull(summary, metric)
        ax.errorbar(x, means, yerr=cis, fmt="o-", capsize=4, color=color,
                    linewidth=2, markersize=5, elinewidth=1.3)
        ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
        ax.set_xlabel("Horizon (days)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.grid(alpha=0.3)

        # Also show C1 and C2 separately on the gap panel
        if metric == "access_gap":
            for cls_metric, cls_color, cls_label in [
                ("class_1_served_rate", CLASS_1_COLOR, "Class 1"),
                ("class_2_served_rate", CLASS_2_COLOR, "Class 2"),
            ]:
                xc, mc, cc = _pull(summary, cls_metric)
                ax.errorbar(xc, mc, yerr=cc, fmt="--s", capsize=3, color=cls_color,
                            linewidth=1.5, markersize=4, elinewidth=1.1, label=cls_label)
            ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Metrics by Booking Horizon  (λ=24, S=20, realistic params, 30 seeds)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    for d in (RAW_DIR, SUMMARY_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("Experiment 3: Horizon sweep …")
    results = run_experiment()
    results.to_csv(RAW_DIR / "results.csv", index=False)

    summary = summarize(results)
    summary.to_csv(SUMMARY_DIR / "summary.csv", index=False)

    plot_class_access_gap(summary, FIGURE_DIR / "class_access_gap.png")
    plot_metrics_by_horizon(summary, FIGURE_DIR / "metrics_by_horizon.png")

    print(f"\nDone. Outputs → {OUTPUT_DIR}")
    print("\nClass access gap by horizon:")
    for _, row in summary[summary["metric"] == "access_gap"].sort_values("horizon_days").iterrows():
        print(f"  H={int(row['horizon_days']):2d}  gap = {row['mean']:+.4f}  ±{row['ci95']:.4f}")


if __name__ == "__main__":
    main()
