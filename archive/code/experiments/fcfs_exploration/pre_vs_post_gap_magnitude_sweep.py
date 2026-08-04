"""
Pre-threshold vs. post-threshold gap: swept across gap magnitudes.

For each gap magnitude g (0.05 to 0.50), two scenarios are compared:

    Pre-threshold gap:  Class 1 (0.00, 0.50), Class 2 (g, 0.50)
    Post-threshold gap: Class 1 (0.00, 0.50), Class 2 (0.00, 0.50 - g)

At every g, the between-class gap magnitude and Class 2 step size
(0.50 - g) are identical. The only difference is whether the gap
sits below or above the threshold.

H7 is strongly supported if the pre-threshold line is consistently
above the post-threshold line across all gap magnitudes.

Outputs
-------
outputs/pre_vs_post_threshold/figures/gap_magnitude_sweep.png
outputs/pre_vs_post_threshold/raw/gap_magnitude_results.csv
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
FIGURE_DIR = REPO_DIR / "outputs" / "pre_vs_post_threshold" / "figures"
RAW_DIR = REPO_DIR / "outputs" / "pre_vs_post_threshold" / "raw"

SEEDS = range(1, 51)

GAP_MAGNITUDES = np.round(np.arange(0.05, 0.55, 0.05), 2)


def make_config(base_config, class2_low, class2_high, seed):
    new_classes = {}
    for cid, params in base_config.classes.items():
        if cid == 1:
            new_classes[cid] = replace(
                params,
                balk_prob=ThresholdRule(
                    threshold=params.balk_prob.threshold,
                    low=0.00,
                    high=0.50,
                ),
            )
        else:
            new_classes[cid] = replace(
                params,
                balk_prob=ThresholdRule(
                    threshold=params.balk_prob.threshold,
                    low=class2_low,
                    high=class2_high,
                ),
            )
    return replace(base_config, classes=new_classes, seed=int(seed))


def run_sweep():
    base_config = load_config(CONFIG_PATH)
    rows = []

    for g in GAP_MAGNITUDES:
        # Pre-threshold gap: Class 2 = (g, 0.50)
        print(f"  gap={g:.2f} pre-threshold: Class 2 = ({g:.2f}, 0.50)")
        for seed in SEEDS:
            config = make_config(base_config, class2_low=g, class2_high=0.50, seed=seed)
            sim = ClinicAppointmentSimulation(config)
            results = sim.run()
            rows.extend(
                class_result_rows(
                    results,
                    {
                        "gap_magnitude": g,
                        "gap_location": "pre-threshold",
                        "class2_low": g,
                        "class2_high": 0.50,
                        "class2_step": 0.50 - g,
                        "seed": seed,
                    },
                )
            )

        # Post-threshold gap: Class 2 = (0.00, 0.50 - g)
        print(f"  gap={g:.2f} post-threshold: Class 2 = (0.00, {0.50 - g:.2f})")
        for seed in SEEDS:
            config = make_config(base_config, class2_low=0.00, class2_high=0.50 - g, seed=seed)
            sim = ClinicAppointmentSimulation(config)
            results = sim.run()
            rows.extend(
                class_result_rows(
                    results,
                    {
                        "gap_magnitude": g,
                        "gap_location": "post-threshold",
                        "class2_low": 0.00,
                        "class2_high": 0.50 - g,
                        "class2_step": 0.50 - g,
                        "seed": seed,
                    },
                )
            )

    return pd.DataFrame(rows)


def compute_gaps(raw):
    """Compute served-rate gap (Class 1 - Class 2) per seed per scenario."""
    pivot = raw.pivot_table(
        index=["gap_magnitude", "gap_location", "seed"],
        columns="class_id",
        values="percent_serviced",
    ).reset_index()

    pivot["served_rate_gap"] = pivot[1] - pivot[2]

    summary = (
        pivot.groupby(["gap_magnitude", "gap_location"])["served_rate_gap"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["se"]

    return summary


def plot_gap_sweep(summary, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    styles = {
        "pre-threshold": {"color": "#9467bd", "marker": "s", "label": "Pre-threshold gap"},
        "post-threshold": {"color": "#2ca02c", "marker": "^", "label": "Post-threshold gap"},
    }

    for location, style in styles.items():
        sub = summary[summary["gap_location"] == location].sort_values("gap_magnitude")
        ax.errorbar(
            sub["gap_magnitude"],
            sub["mean"],
            yerr=sub["ci95"],
            capsize=3,
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            label=style["label"],
        )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Class Served-Rate Gap Across Gap Magnitudes\n(Class 2 step size matched at every point)")
    ax.set_xlabel("Between-class gap magnitude")
    ax.set_ylabel("Served-rate gap (Class 1 − Class 2)")
    ax.set_xticks(GAP_MAGNITUDES)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Running gap magnitude sweep...")
    raw = run_sweep()
    raw.to_csv(RAW_DIR / "gap_magnitude_results.csv", index=False)

    summary = compute_gaps(raw)
    plot_gap_sweep(summary, FIGURE_DIR / "gap_magnitude_sweep.png")

    print("Done.")


if __name__ == "__main__":
    main()