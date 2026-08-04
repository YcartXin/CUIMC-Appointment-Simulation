"""
Experiment 4: Arrival Share Sweep at Symmetric Behavior

Tests whether Class 1's access advantage depends on its arrival share when
all behavioral parameters are symmetric (midpoints of the realistic C1/C2
values). Isolates the regression finding that more Class 1 arrivals correlate
negatively with its own access advantage (β = −0.12).

All behavioral parameters at symmetric midpoints. λ_total = 24 fixed (S=20,
H=28). Only the class 1 share of arrivals is varied.

class_1_share ∈ {0.30, 0.40, 0.50, 0.60, 0.70}  ×  30 seeds = 150 runs.

Outputs
-------
outputs/exp4_arrival_share_symmetric/raw/results.csv
outputs/exp4_arrival_share_symmetric/summary/summary.csv
outputs/exp4_arrival_share_symmetric/figures/class_access_gap.png
outputs/exp4_arrival_share_symmetric/figures/served_rates.png
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
from simulation.model import SimulationConfig, ThresholdRule
from analysis.metrics import aggregate_result_row
from analysis.plot_style import CLASS_1_COLOR, CLASS_2_COLOR, BASELINE_COLOR

CONFIG_PATH = REPO_DIR / "configs" / "realistic.yaml"
OUTPUT_DIR  = REPO_DIR / "outputs" / "exp4_arrival_share_symmetric"
RAW_DIR     = OUTPUT_DIR / "raw"
SUMMARY_DIR = OUTPUT_DIR / "summary"
FIGURE_DIR  = OUTPUT_DIR / "figures"

SEEDS           = range(1, 31)
LAMBDA_TOTAL    = 24.0
CLASS_1_SHARES  = [0.30, 0.40, 0.50, 0.60, 0.70]

# Symmetric behavioral midpoints (same as Experiment 1)
_SYM_BALK   = ThresholdRule(threshold=17, low=0.03,  high=0.635)
_SYM_NOSHOW = ThresholdRule(threshold=17, low=0.08,  high=0.41)
_SYM_CANCEL = 0.015


# ---------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------

def make_config(
    base: SimulationConfig,
    class_1_share: float,
    seed: int,
) -> SimulationConfig:
    lambda1 = LAMBDA_TOTAL * class_1_share
    lambda2 = LAMBDA_TOTAL * (1.0 - class_1_share)

    c1 = replace(
        base.classes[1],
        lambda_per_day=lambda1,
        cancel_prob=_SYM_CANCEL,
        balk_prob=_SYM_BALK,
        no_show_prob=_SYM_NOSHOW,
    )
    c2 = replace(
        base.classes[2],
        lambda_per_day=lambda2,
        cancel_prob=_SYM_CANCEL,
        balk_prob=_SYM_BALK,
        no_show_prob=_SYM_NOSHOW,
    )
    return replace(base, classes={1: c1, 2: c2}, seed=int(seed))


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------

def run_experiment() -> pd.DataFrame:
    base = load_config(CONFIG_PATH)
    rows = []
    for share in CLASS_1_SHARES:
        print(f"  class_1_share = {share:.2f}  (λ₁={LAMBDA_TOTAL*share:.1f}, λ₂={LAMBDA_TOTAL*(1-share):.1f})")
        for seed in SEEDS:
            cfg = make_config(base, share, seed)
            result = ClinicAppointmentSimulation(cfg).run()
            row = aggregate_result_row(
                result,
                {"class_1_share": share, "lambda_1": LAMBDA_TOTAL * share,
                 "lambda_2": LAMBDA_TOTAL * (1.0 - share), "seed": seed},
            )
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
    for share, sub in df.groupby("class_1_share"):
        for m in metrics:
            se = sub[m].std() / np.sqrt(len(sub))
            records.append({
                "class_1_share": share,
                "metric":        m,
                "mean":          sub[m].mean(),
                "std":           sub[m].std(),
                "se":            se,
                "ci95":          1.96 * se,
                "n":             len(sub),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------

def _pull(summary: pd.DataFrame, metric: str):
    sub = summary[summary["metric"] == metric].sort_values("class_1_share")
    return sub["class_1_share"].tolist(), sub["mean"].tolist(), sub["ci95"].tolist()


def plot_class_access_gap(summary: pd.DataFrame, path: Path) -> None:
    x, means, cis = _pull(summary, "access_gap")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(x, means, yerr=cis, fmt="o-", capsize=4, color=CLASS_1_COLOR,
                linewidth=2, markersize=6, elinewidth=1.4)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Class 1 share of total arrivals")
    ax.set_ylabel("Class 1 served rate − Class 2 served rate")
    ax.set_title(
        "Class Access Gap vs Arrival Share  (symmetric behavioral parameters)\n"
        "λ_total=24, S=20, H=28 days · 30 seeds · ±95% CI",
        fontsize=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.0%}" for s in x])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_served_rates(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel A: served rates
    ax = axes[0]
    for cls_metric, cls_color, cls_label in [
        ("class_1_served_rate", CLASS_1_COLOR, "Class 1"),
        ("class_2_served_rate", CLASS_2_COLOR, "Class 2"),
    ]:
        x, means, cis = _pull(summary, cls_metric)
        ax.errorbar(x, means, yerr=cis, fmt="o-", capsize=3, color=cls_color,
                    linewidth=2, markersize=5, elinewidth=1.2, label=cls_label)
    ax.set_xlabel("Class 1 share of total arrivals")
    ax.set_ylabel("Served rate")
    ax.set_title("Class Served Rates vs Arrival Share", fontsize=10)
    ax.set_xticks(CLASS_1_SHARES)
    ax.set_xticklabels([f"{s:.0%}" for s in CLASS_1_SHARES])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # Panel B: access gap
    ax = axes[1]
    x, means, cis = _pull(summary, "access_gap")
    ax.errorbar(x, means, yerr=cis, fmt="o-", capsize=4, color=BASELINE_COLOR,
                linewidth=2, markersize=6, elinewidth=1.4)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Class 1 share of total arrivals")
    ax.set_ylabel("Gap (C1 − C2 served rate)")
    ax.set_title("Class Access Gap vs Arrival Share", fontsize=10)
    ax.set_xticks(CLASS_1_SHARES)
    ax.set_xticklabels([f"{s:.0%}" for s in CLASS_1_SHARES])
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Arrival Share Sweep — Symmetric Behavioral Parameters\n"
        "λ_total=24, S=20, H=28 days, 30 seeds",
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

    print("Experiment 4: Arrival share sweep (symmetric behavior) …")
    results = run_experiment()
    results.to_csv(RAW_DIR / "results.csv", index=False)

    summary = summarize(results)
    summary.to_csv(SUMMARY_DIR / "summary.csv", index=False)

    plot_class_access_gap(summary, FIGURE_DIR / "class_access_gap.png")
    plot_served_rates(summary, FIGURE_DIR / "served_rates.png")

    print(f"\nDone. Outputs → {OUTPUT_DIR}")
    print("\nClass access gap by arrival share:")
    for _, row in summary[summary["metric"] == "access_gap"].sort_values("class_1_share").iterrows():
        print(f"  share={row['class_1_share']:.0%}  gap = {row['mean']:+.4f}  ±{row['ci95']:.4f}")


if __name__ == "__main__":
    main()
