"""
Experiment 2: Demand × No-Show Interaction Grid

Sweeps total arrival rate (λ_total) and no-show intensity simultaneously to
find where no-show becomes the binding utilization constraint versus where
demand pressure dominates access.

λ_total ∈ {12, 16, 20, 24, 28, 32}  (λ/S ratios 0.6 – 1.6, S=20 fixed)
no_show_scale ∈ {0.0, 0.5, 1.0, 1.5, 2.0}  (1.0 = realistic values)

Class ratio λ₁:λ₂ = 14:10 held fixed. No-show thresholds and low values
fixed at realistic.yaml; only the high component is scaled.

6 × 5 × 30 = 900 runs.

Outputs
-------
outputs/exp2_demand_noshow_grid/raw/results.csv
outputs/exp2_demand_noshow_grid/summary/grid_summary.csv
outputs/exp2_demand_noshow_grid/figures/utilization_heatmap.png
outputs/exp2_demand_noshow_grid/figures/served_rate_heatmap.png
outputs/exp2_demand_noshow_grid/figures/gap_heatmap.png
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
from analysis.metrics import aggregate_result_row, class_result_rows
from analysis.plot_style import UTILIZATION_CMAP, ACCESS_CMAP, CLASS_GAP_CMAP

CONFIG_PATH = REPO_DIR / "configs" / "realistic.yaml"
OUTPUT_DIR  = REPO_DIR / "outputs" / "exp2_demand_noshow_grid"
RAW_DIR     = OUTPUT_DIR / "raw"
SUMMARY_DIR = OUTPUT_DIR / "summary"
FIGURE_DIR  = OUTPUT_DIR / "figures"

SEEDS = range(1, 31)

LAMBDA_TOTALS    = [12, 16, 20, 24, 28, 32]
NOSHOW_SCALES    = [0.0, 0.5, 1.0, 1.5, 2.0]

# Realistic no-show high values (kept as ratio reference)
_R_C1_NS_HIGH = 0.31
_R_C2_NS_HIGH = 0.51
# Realistic thresholds and lows (held fixed)
_R_C1_NS_THR  = 21
_R_C1_NS_LOW  = 0.01
_R_C2_NS_THR  = 14
_R_C2_NS_LOW  = 0.15

# λ₁:λ_total ratio from realistic
_LAMBDA_RATIO = 14.0 / 24.0


# ---------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------

def make_config(
    base: SimulationConfig,
    lambda_total: float,
    noshow_scale: float,
    seed: int,
) -> SimulationConfig:
    lambda1 = lambda_total * _LAMBDA_RATIO
    lambda2 = lambda_total * (1.0 - _LAMBDA_RATIO)

    c1_ns_high = min(1.0, _R_C1_NS_HIGH * noshow_scale)
    c2_ns_high = min(1.0, _R_C2_NS_HIGH * noshow_scale)

    c1 = replace(
        base.classes[1],
        lambda_per_day=lambda1,
        no_show_prob=ThresholdRule(_R_C1_NS_THR, _R_C1_NS_LOW, c1_ns_high),
    )
    c2 = replace(
        base.classes[2],
        lambda_per_day=lambda2,
        no_show_prob=ThresholdRule(_R_C2_NS_THR, _R_C2_NS_LOW, c2_ns_high),
    )
    return replace(base, classes={1: c1, 2: c2}, seed=int(seed))


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------

def run_experiment() -> pd.DataFrame:
    base = load_config(CONFIG_PATH)
    rows = []
    total = len(LAMBDA_TOTALS) * len(NOSHOW_SCALES)
    done  = 0
    for lam in LAMBDA_TOTALS:
        for scale in NOSHOW_SCALES:
            done += 1
            print(f"  [{done}/{total}] λ_total={lam}  ns_scale={scale:.1f}")
            for seed in SEEDS:
                cfg = make_config(base, lam, scale, seed)
                result = ClinicAppointmentSimulation(cfg).run()
                row = aggregate_result_row(
                    result,
                    {"lambda_total": lam, "noshow_scale": scale, "seed": seed},
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
    metrics = ["average_utilization", "overall_percent_serviced",
               "class_1_served_rate", "class_2_served_rate", "access_gap"]
    records = []
    for (lam, scale), sub in df.groupby(["lambda_total", "noshow_scale"]):
        for m in metrics:
            records.append({
                "lambda_total":  lam,
                "noshow_scale":  scale,
                "lambda_over_s": lam / 20.0,
                "metric":        m,
                "mean":          sub[m].mean(),
                "se":            sub[m].std() / np.sqrt(len(sub)),
                "ci95":          1.96 * sub[m].std() / np.sqrt(len(sub)),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------

def _pivot(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = summary[summary["metric"] == metric].copy()
    return sub.pivot(index="noshow_scale", columns="lambda_total", values="mean")


def _heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    cmap,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".2f",
) -> None:
    im = ax.imshow(
        data.values,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([f"{v}" for v in data.columns], fontsize=9)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in data.index], fontsize=9)
    ax.set_xlabel("Total arrivals λ (S = 20 slots/day)")
    ax.set_ylabel("No-show scale  (1.0 = realistic)")
    ax.set_title(title, fontsize=10)
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.values[i, j]
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                    fontsize=7.5, color="k")


def create_figures(summary: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    util_pivot   = _pivot(summary, "average_utilization")
    served_pivot = _pivot(summary, "overall_percent_serviced")
    gap_pivot    = _pivot(summary, "access_gap")

    # Utilization heatmap
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _heatmap(ax, util_pivot, "Average Utilization (λ/S grid)", UTILIZATION_CMAP, 0.0, 1.0)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "utilization_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Served rate heatmap
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _heatmap(ax, served_pivot, "Overall Served Rate (λ/S grid)", ACCESS_CMAP, 0.0, 1.0)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "served_rate_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Class gap heatmap
    gap_abs = gap_pivot.abs().values.max()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _heatmap(ax, gap_pivot, "Class 1 Access Advantage (C1 − C2 served rate)",
             CLASS_GAP_CMAP, -gap_abs, gap_abs)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "gap_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Combined 3-panel
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _heatmap(axes[0], util_pivot,   "Utilization",  UTILIZATION_CMAP, 0.0, 1.0)
    _heatmap(axes[1], served_pivot, "Served rate",  ACCESS_CMAP,      0.0, 1.0)
    _heatmap(axes[2], gap_pivot,    "Class gap (C1−C2)", CLASS_GAP_CMAP, -gap_abs, gap_abs)
    fig.suptitle("Demand × No-Show Grid  (S=20, 30 seeds per cell)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "grid_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    for d in (RAW_DIR, SUMMARY_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("Experiment 2: Demand × no-show grid …")
    results = run_experiment()
    results.to_csv(RAW_DIR / "results.csv", index=False)

    summary = summarize(results)
    summary.to_csv(SUMMARY_DIR / "grid_summary.csv", index=False)

    create_figures(summary)
    print(f"\nDone. Outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
