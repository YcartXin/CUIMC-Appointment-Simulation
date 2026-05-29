"""
Experiment 1: Asymmetry Attribution Factorial

Decomposes the realistic-scenario class access gap by introducing behavioral
asymmetries one at a time: cancellation gap, balking threshold gap, no-show
gap, and all three together.

Symmetric baseline uses per-parameter midpoints of the realistic C1/C2 values.
Arrival rates and capacity are fixed at realistic.yaml values (lambda_total=24,
S=20, H=28).

5 configurations × 30 seeds = 150 runs.

Outputs
-------
outputs/exp1_asymmetry_attribution/raw/results.csv
outputs/exp1_asymmetry_attribution/summary/summary.csv
outputs/exp1_asymmetry_attribution/figures/class_access_gap.png
outputs/exp1_asymmetry_attribution/figures/served_rates.png
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
from simulation.model import SimulationConfig, ThresholdRule, PatientClassParams
from analysis.metrics import aggregate_result_row, class_result_rows
from analysis.plot_style import CLASS_1_COLOR, CLASS_2_COLOR, BASELINE_COLOR

CONFIG_PATH = REPO_DIR / "configs" / "realistic.yaml"
OUTPUT_DIR  = REPO_DIR / "outputs" / "exp1_asymmetry_attribution"
RAW_DIR     = OUTPUT_DIR / "raw"
SUMMARY_DIR = OUTPUT_DIR / "summary"
FIGURE_DIR  = OUTPUT_DIR / "figures"

SEEDS = range(1, 31)

# ---------------------------------------------------------------
# Symmetric midpoint values (midpoint of C1 and C2 in realistic)
# ---------------------------------------------------------------
_SYM_BALK   = ThresholdRule(threshold=17, low=0.03,  high=0.635)
_SYM_NOSHOW = ThresholdRule(threshold=17, low=0.08,  high=0.41)
_SYM_CANCEL = 0.015

# Realistic asymmetric values
_R_C1_CANCEL = 0.01
_R_C2_CANCEL = 0.02
_R_C1_BALK   = ThresholdRule(threshold=21, low=0.02, high=0.55)
_R_C2_BALK   = ThresholdRule(threshold=14, low=0.04, high=0.72)
_R_C1_NOSHOW = ThresholdRule(threshold=21, low=0.01, high=0.31)
_R_C2_NOSHOW = ThresholdRule(threshold=14, low=0.15, high=0.51)

# Each entry: (label, c1_cancel, c2_cancel, c1_balk, c2_balk, c1_noshow, c2_noshow)
CONFIGS = [
    ("sym",       "Symmetric baseline",          _SYM_CANCEL,   _SYM_CANCEL,   _SYM_BALK,   _SYM_BALK,   _SYM_NOSHOW,   _SYM_NOSHOW),
    ("+cancel",   "+Cancellation gap",           _R_C1_CANCEL,  _R_C2_CANCEL,  _SYM_BALK,   _SYM_BALK,   _SYM_NOSHOW,   _SYM_NOSHOW),
    ("+balk",     "+Balking gap",                _SYM_CANCEL,   _SYM_CANCEL,   _R_C1_BALK,  _R_C2_BALK,  _SYM_NOSHOW,   _SYM_NOSHOW),
    ("+noshow",   "+No-show gap",                _SYM_CANCEL,   _SYM_CANCEL,   _SYM_BALK,   _SYM_BALK,   _R_C1_NOSHOW,  _R_C2_NOSHOW),
    ("realistic", "All gaps (realistic)",        _R_C1_CANCEL,  _R_C2_CANCEL,  _R_C1_BALK,  _R_C2_BALK,  _R_C1_NOSHOW,  _R_C2_NOSHOW),
]
CONFIG_KEYS   = [c[0] for c in CONFIGS]
CONFIG_LABELS = [c[1] for c in CONFIGS]


# ---------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------

def make_config(
    base: SimulationConfig,
    c1_cancel: float, c2_cancel: float,
    c1_balk: ThresholdRule, c2_balk: ThresholdRule,
    c1_noshow: ThresholdRule, c2_noshow: ThresholdRule,
    seed: int,
) -> SimulationConfig:
    c1 = replace(base.classes[1], cancel_prob=c1_cancel,
                 balk_prob=c1_balk, no_show_prob=c1_noshow)
    c2 = replace(base.classes[2], cancel_prob=c2_cancel,
                 balk_prob=c2_balk, no_show_prob=c2_noshow)
    return replace(base, classes={1: c1, 2: c2}, seed=int(seed))


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------

def run_experiment() -> pd.DataFrame:
    base = load_config(CONFIG_PATH)
    rows = []
    for key, label, c1_ca, c2_ca, c1_ba, c2_ba, c1_ns, c2_ns in CONFIGS:
        print(f"  {label}")
        for seed in SEEDS:
            cfg = make_config(base, c1_ca, c2_ca, c1_ba, c2_ba, c1_ns, c2_ns, seed)
            result = ClinicAppointmentSimulation(cfg).run()
            row = aggregate_result_row(result, {"config": key, "label": label, "seed": seed})
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
    metrics = ["average_utilization", "overall_percent_serviced",
               "class_1_served_rate", "class_2_served_rate", "access_gap"]
    records = []
    for key in CONFIG_KEYS:
        sub = df[df["config"] == key]
        for m in metrics:
            records.append({
                "config": key,
                "label":  CONFIGS[CONFIG_KEYS.index(key)][1],
                "metric": m,
                "mean":   sub[m].mean(),
                "std":    sub[m].std(),
                "se":     sub[m].std() / np.sqrt(len(sub)),
                "ci95":   1.96 * sub[m].std() / np.sqrt(len(sub)),
                "n":      len(sub),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------

def _get(summary: pd.DataFrame, config: str, metric: str) -> pd.Series:
    return summary[(summary["config"] == config) & (summary["metric"] == metric)].iloc[0]


def plot_served_rates(summary: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(CONFIG_KEYS))
    width = 0.28
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: served rates
    ax = axes[0]
    c1_means = [_get(summary, k, "class_1_served_rate")["mean"] for k in CONFIG_KEYS]
    c1_ci    = [_get(summary, k, "class_1_served_rate")["ci95"] for k in CONFIG_KEYS]
    c2_means = [_get(summary, k, "class_2_served_rate")["mean"] for k in CONFIG_KEYS]
    c2_ci    = [_get(summary, k, "class_2_served_rate")["ci95"] for k in CONFIG_KEYS]

    ax.bar(x - width/2, c1_means, width, yerr=c1_ci, capsize=3,
           color=CLASS_1_COLOR, alpha=0.85, label="Class 1", error_kw={"linewidth": 1.2})
    ax.bar(x + width/2, c2_means, width, yerr=c2_ci, capsize=3,
           color=CLASS_2_COLOR, alpha=0.85, label="Class 2", error_kw={"linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIG_LABELS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Served rate (completed visits / arrivals)")
    ax.set_title("Class-Level Served Rates")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: access gap
    ax = axes[1]
    gap_means = [_get(summary, k, "access_gap")["mean"] for k in CONFIG_KEYS]
    gap_ci    = [_get(summary, k, "access_gap")["ci95"] for k in CONFIG_KEYS]
    colors = [CLASS_1_COLOR if g >= 0 else CLASS_2_COLOR for g in gap_means]
    ax.bar(x, gap_means, width*2, yerr=gap_ci, capsize=3, color=colors, alpha=0.85,
           error_kw={"linewidth": 1.2})
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIG_LABELS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Class 1 served rate − Class 2 served rate")
    ax.set_title("Class Access Gap Decomposition")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Asymmetry Attribution: Realistic Scenario (λ=24, S=20, H=28 days)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_access_gap(summary: pd.DataFrame, path: Path) -> None:
    """Single-panel gap chart, cleaner for the deck."""
    x = np.arange(len(CONFIG_KEYS))
    gap_means = [_get(summary, k, "access_gap")["mean"] for k in CONFIG_KEYS]
    gap_ci    = [_get(summary, k, "access_gap")["ci95"] for k in CONFIG_KEYS]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#a8c6e8" if g >= 0 else "#f4a582" for g in gap_means]
    ax.bar(x, gap_means, 0.55, yerr=gap_ci, capsize=4, color=colors, alpha=0.9,
           edgecolor="0.3", linewidth=0.7, error_kw={"linewidth": 1.4})
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIG_LABELS, rotation=18, ha="right", fontsize=9.5)
    ax.set_ylabel("Class 1 served rate − Class 2 served rate")
    ax.set_title(
        "Class Access Gap by Source of Asymmetry\n"
        "λ=24, S=20, H=28 days · 30 seeds · ±95% CI",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    for d in (RAW_DIR, SUMMARY_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("Experiment 1: Asymmetry attribution …")
    results = run_experiment()
    results.to_csv(RAW_DIR / "results.csv", index=False)

    summary = summarize(results)
    summary.to_csv(SUMMARY_DIR / "summary.csv", index=False)

    plot_served_rates(summary, FIGURE_DIR / "served_rates.png")
    plot_class_access_gap(summary, FIGURE_DIR / "class_access_gap.png")

    print(f"\nDone. Outputs → {OUTPUT_DIR}")
    print("\nClass access gap by configuration:")
    for key in CONFIG_KEYS:
        row = _get(summary, key, "access_gap")
        print(f"  {row['label']:<35} gap = {row['mean']:+.4f}  ±{row['ci95']:.4f}")


if __name__ == "__main__":
    main()
