"""
Accepted-delay distributions under different Class 1 balking thresholds.

This script produces:
1. Baseline accepted-delay distribution
2. Accepted-delay distribution when Class 1 balking threshold = 10

Outputs
-------
outputs/class1_balking_threshold/figures/accepted_delay_distribution_baseline.png
outputs/class1_balking_threshold/figures/accepted_delay_distribution_threshold10.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Path setup
# ============================================================

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from simulation.config_loader import load_config
from simulation.engine import ClinicAppointmentSimulation
from analysis.plot_style import driver_line_style

# ============================================================
# Settings
# ============================================================

CONFIG_PATH = REPO_DIR / "configs" / "baseline.yaml"
FIGURE_DIR = REPO_DIR / "outputs" / "class1_balking_threshold" / "figures"
SEEDS = range(1, 101)

# ============================================================
# Helpers
# ============================================================

def with_class1_balking_threshold(config, new_threshold: int):
    """
    Return a copy of config with only Class 1's balking threshold changed.
    """
    class1 = config.classes[1]
    new_balk_rule = replace(class1.balk_prob, threshold=new_threshold)
    new_class1 = replace(class1, balk_prob=new_balk_rule)

    new_classes = dict(config.classes)
    new_classes[1] = new_class1

    return replace(config, classes=new_classes)


def collect_delay_distributions(seeds, threshold_override=None):
    """
    Run the simulation across multiple seeds and aggregate
    accepted_delay_counts per class.

    Parameters
    ----------
    seeds : iterable
        Seeds to run.
    threshold_override : int or None
        If provided, override Class 1's balking threshold.

    Returns
    -------
    dict
        class_id -> {tau: total_count}
    """
    base_config = load_config(CONFIG_PATH)

    if threshold_override is not None:
        base_config = with_class1_balking_threshold(base_config, threshold_override)

    class_ids = sorted(base_config.classes.keys())
    aggregated = {cid: {} for cid in class_ids}

    for seed in seeds:
        config = replace(base_config, seed=int(seed))

        sim = ClinicAppointmentSimulation(config)
        results = sim.run()

        for cid in class_ids:
            delay_counts = results.class_metrics[cid].accepted_delay_counts
            for tau, count in delay_counts.items():
                aggregated[cid][tau] = aggregated[cid].get(tau, 0) + count

    return aggregated


def plot_delay_distribution(aggregated, output_path, threshold_value, title_suffix):
    """
    Plot a stacked bar histogram of accepted booking delays by class.
    """
    all_taus = sorted(set().union(*(d.keys() for d in aggregated.values())))

    fig, ax = plt.subplots(figsize=(8, 5))

    class_ids = sorted(aggregated.keys())
    bottom = np.zeros(len(all_taus))

    for idx, cid in enumerate(class_ids):
        counts = np.array([aggregated[cid].get(tau, 0) for tau in all_taus])
        style = driver_line_style("balking", f"Class {cid}", idx + 1)
        color = style.get("color", None)

        ax.bar(
            all_taus,
            counts,
            bottom=bottom,
            label=f"Class {cid}",
            color=color,
            alpha=0.7,
            width=0.8,
        )
        bottom += counts

    ax.axvline(
        x=threshold_value,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"Class 1 balking threshold = {threshold_value}",
    )

    ax.set_title(f"Accepted Booking Delay Distribution ({title_suffix})")
    ax.set_xlabel("Booking delay τ (days)")
    ax.set_ylabel("Total accepted bookings (100 seeds)")
    ax.set_xticks(all_taus)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 1. Baseline
    # ----------------------------
    print("Collecting baseline accepted-delay distributions across 100 seeds...")
    baseline_aggregated = collect_delay_distributions(SEEDS, threshold_override=None)

    plot_delay_distribution(
        baseline_aggregated,
        FIGURE_DIR / "accepted_delay_distribution_baseline.png",
        threshold_value=9,
        title_suffix="Baseline",
    )

    # ----------------------------
    # 2. Threshold = 10
    # ----------------------------
    print("Collecting threshold=10 accepted-delay distributions across 100 seeds...")
    threshold10_aggregated = collect_delay_distributions(SEEDS, threshold_override=10)

    plot_delay_distribution(
        threshold10_aggregated,
        FIGURE_DIR / "accepted_delay_distribution_threshold10.png",
        threshold_value=10,
        title_suffix="Class 1 threshold = 10",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()