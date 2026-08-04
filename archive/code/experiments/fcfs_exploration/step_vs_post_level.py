"""
Within-class step size vs. post-threshold level.

Tests whether a class's served rate is driven more by the step size
(b_1 - b_0) or the absolute post-threshold value (b_1).

Sweep 1 — Fix step at 0.30, vary post-threshold level:
    (0.00, 0.30) → (0.10, 0.40) → (0.20, 0.50) → (0.30, 0.60) → (0.40, 0.70)
    Step is always 0.30; only the level shifts.

Sweep 2 — Fix post-threshold at 0.50, vary step size:
    (0.40, 0.50) → (0.30, 0.50) → (0.20, 0.50) → (0.10, 0.50) → (0.00, 0.50)
    Post-threshold is always 0.50; only the step grows.

Class 2 is held at baseline throughout.

Outputs
-------
outputs/step_vs_post_level/figures/step_vs_post_level.png
outputs/step_vs_post_level/raw/results.csv
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
from simulation.model import ThresholdRule
from analysis.metrics import class_result_rows

CONFIG_PATH = REPO_DIR / "configs" / "baseline.yaml"
FIGURE_DIR = REPO_DIR / "outputs" / "step_vs_post_level" / "figures"
RAW_DIR = REPO_DIR / "outputs" / "step_vs_post_level" / "raw"

SEEDS = range(1, 101)

# Sweep 1: fixed step = 0.30, increasing post-threshold level
SWEEP_FIXED_STEP = [
    {"low": 0.00, "high": 0.30, "label": "b₁=0.30"},
    {"low": 0.10, "high": 0.40, "label": "b₁=0.40"},
    {"low": 0.20, "high": 0.50, "label": "b₁=0.50"},
    {"low": 0.30, "high": 0.60, "label": "b₁=0.60"},
    {"low": 0.40, "high": 0.70, "label": "b₁=0.70"},
]

# Sweep 2: fixed post-threshold = 0.50, increasing step size
SWEEP_FIXED_POST = [
    {"low": 0.40, "high": 0.50, "label": "step=0.10"},
    {"low": 0.30, "high": 0.50, "label": "step=0.20"},
    {"low": 0.20, "high": 0.50, "label": "step=0.30"},
    {"low": 0.10, "high": 0.50, "label": "step=0.40"},
    {"low": 0.00, "high": 0.50, "label": "step=0.50"},
]


def make_config(base_config, low, high, seed):
    class1_params = base_config.classes[1]
    new_class1 = replace(
        class1_params,
        balk_prob=ThresholdRule(
            threshold=class1_params.balk_prob.threshold,
            low=low,
            high=high,
        ),
    )
    new_classes = dict(base_config.classes)
    new_classes[1] = new_class1
    return replace(base_config, classes=new_classes, seed=int(seed))


def run_sweep(sweep_configs, sweep_name, base_config):
    rows = []
    for i, cfg in enumerate(sweep_configs):
        print(f"  {sweep_name}: b_0={cfg['low']:.2f}, b_1={cfg['high']:.2f}")
        for seed in SEEDS:
            config = make_config(base_config, cfg["low"], cfg["high"], seed)
            sim = ClinicAppointmentSimulation(config)
            results = sim.run()
            rows.extend(
                class_result_rows(
                    results,
                    {
                        "sweep": sweep_name,
                        "point": i + 1,
                        "low": cfg["low"],
                        "high": cfg["high"],
                        "step": cfg["high"] - cfg["low"],
                        "label": cfg["label"],
                        "seed": seed,
                    },
                )
            )
    return rows


def summarize_class1(raw):
    c1 = raw[(raw["class_id"] == 1)].copy()
    summary = (
        c1.groupby(["sweep", "point", "label"])["percent_serviced"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["se"]
    return summary


def plot_comparison(summary, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    sweeps = {
        "fixed_step": {
            "color": "#9467bd",
            "marker": "s",
            "label": "Fixed step (0.30), varying post-threshold",
        },
        "fixed_post": {
            "color": "#2ca02c",
            "marker": "^",
            "label": "Fixed post-threshold (0.50), varying step",
        },
    }

    for sweep_name, style in sweeps.items():
        sub = summary[summary["sweep"] == sweep_name].sort_values("point")
        ax.errorbar(
            sub["point"],
            sub["mean"],
            yerr=sub["ci95"],
            capsize=3,
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            label=style["label"],
        )

        # Annotate each point with its label
        for _, row in sub.iterrows():
            ax.annotate(
                row["label"],
                xy=(row["point"], row["mean"]),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=style["color"],
            )

    ax.set_title("Class 1 Served Rate: Step Size vs. Post-Threshold Level")
    ax.set_xlabel("Sweep point")
    ax.set_ylabel("Class 1 served rate")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    base_config = load_config(CONFIG_PATH)

    print("Running sweeps...")
    rows = []
    rows.extend(run_sweep(SWEEP_FIXED_STEP, "fixed_step", base_config))
    rows.extend(run_sweep(SWEEP_FIXED_POST, "fixed_post", base_config))

    raw = pd.DataFrame(rows)
    raw.to_csv(RAW_DIR / "results.csv", index=False)

    summary = summarize_class1(raw)
    plot_comparison(summary, FIGURE_DIR / "step_vs_post_level.png")

    print("Done.")


if __name__ == "__main__":
    main()