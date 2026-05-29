"""
Baseline accepted-delay distribution.

Runs the baseline configuration across 100 seeds and aggregates
accepted_delay_counts from ClassMetrics to produce a histogram of
booking delays (tau) by class.

This figure serves as evidence for H5 by showing where the high-mass
delay region is. When a balking threshold crosses this region, many
patients' balking decisions change at once, producing the nonlinear
jumps observed in the balking-threshold sweep.

Outputs
-------
outputs/class1_balking_threshold/figures/accepted_delay_distribution.png
"""
from __future__ import annotations

import sys
from pathlib import Path

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
# Run baseline and collect delay counts
# ============================================================

def collect_delay_distributions(seeds):
    """
    Run the baseline simulation across multiple seeds and aggregate
    accepted_delay_counts per class.

    Returns:
        dict mapping class_id -> {tau: total_count}
    """
    base_config = load_config(CONFIG_PATH)
    class_ids = sorted(base_config.classes.keys())

    aggregated = {cid: {} for cid in class_ids}

    for seed in seeds:
        from dataclasses import replace
        config = replace(base_config, seed=int(seed))

        sim = ClinicAppointmentSimulation(config)
        results = sim.run()

        for cid in class_ids:
            delay_counts = results.class_metrics[cid].accepted_delay_counts
            for tau, count in delay_counts.items():
                aggregated[cid][tau] = aggregated[cid].get(tau, 0) + count

    return aggregated


def plot_delay_distribution(aggregated, output_path):
    """
    Plot a stacked bar histogram of accepted booking delays by class.
    """
    all_taus = sorted(
        set().union(*(d.keys() for d in aggregated.values()))
    )

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

    # Mark the baseline balking threshold
    baseline_threshold = 9
    ax.axvline(
        x=baseline_threshold,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
        label=f"Baseline balking threshold ({baseline_threshold})",
    )

    ax.set_title("Accepted Booking Delay Distribution (Baseline)")
    ax.set_xlabel("Booking delay τ (days)")
    ax.set_ylabel("Total accepted bookings (100 seeds)")
    ax.set_xticks(all_taus)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting accepted delay distributions across 100 seeds...")
    aggregated = collect_delay_distributions(SEEDS)

    for cid, counts in sorted(aggregated.items()):
        total = sum(counts.values())
        print(f"  Class {cid}: {total:,} total accepted bookings")
        for tau in sorted(counts.keys()):
            pct = 100 * counts[tau] / total
            print(f"    τ={tau:2d}: {counts[tau]:>7,}  ({pct:5.1f}%)")

    plot_delay_distribution(
        aggregated,
        FIGURE_DIR / "accepted_delay_distribution.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()