"""
Mean offered booking delay vs. balking step at different arrival rates.

Sweeps Class 1 high-balking probability (0.0–0.9) at several per-class
arrival rates. Both classes share the same lambda and balking parameters
so that the only asymmetry is the sweep variable.

The figure shows one line per arrival rate, testing whether demand
pressure changes how balking step interacts with mean offered delay.

Outputs
-------
outputs/balking_step_by_arrival/figures/offered_delay_by_arrival_rate.png
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

# ============================================================
# Path setup
# ============================================================

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from simulation.config_loader import load_config
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import SimulationConfig, ThresholdRule
from analysis.metrics import aggregate_result_row

# ============================================================
# Experiment settings
# ============================================================

CONFIG_PATH = REPO_DIR / "configs" / "baseline.yaml"

OUTPUT_DIR = REPO_DIR / "outputs" / "balking_step_by_arrival"
RAW_DIR = OUTPUT_DIR / "raw"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Per-class arrival rates to test (both classes get the same lambda).
# Baseline is 50 per class (100 total) with 32 slots → λ/S ratios shown.
LAMBDA_VALUES = [20, 35, 50, 65]

BALKING_STEPS = np.round(np.arange(0.0, 1.0, 0.1), 2)

SEEDS = range(1, 31)  # 30 seeds per combination

# ============================================================
# Config modification
# ============================================================

def make_config(
    base_config: SimulationConfig,
    lambda_per_day: float,
    high_balk: float,
    seed: int,
) -> SimulationConfig:
    """
    Set both classes to the same lambda and high-balk value, change seed.
    """
    new_classes = {}
    for cid, params in base_config.classes.items():
        if not isinstance(params.balk_prob, ThresholdRule):
            raise TypeError("Expected ThresholdRule for balk_prob")

        old_rule = params.balk_prob
        new_classes[cid] = replace(
            params,
            lambda_per_day=float(lambda_per_day),
            balk_prob=ThresholdRule(
                threshold=old_rule.threshold,
                low=old_rule.low,
                high=float(high_balk),
            ),
        )

    return replace(base_config, classes=new_classes, seed=int(seed))


# ============================================================
# Run sweep
# ============================================================

def run_sweep() -> pd.DataFrame:
    base_config = load_config(CONFIG_PATH)
    slots = base_config.slots_per_day

    rows = []
    total = len(LAMBDA_VALUES) * len(BALKING_STEPS) * len(SEEDS)
    count = 0

    for lam in LAMBDA_VALUES:
        total_demand = 2 * lam
        ratio = total_demand / slots
        print(f"λ={lam}/class  (total={total_demand}, λ/S={ratio:.2f})")

        for high_balk in BALKING_STEPS:
            for seed in SEEDS:
                config = make_config(base_config, lam, high_balk, seed)
                sim = ClinicAppointmentSimulation(config)
                results = sim.run()

                row = aggregate_result_row(
                    results,
                    {
                        "lambda_per_day": lam,
                        "total_demand": total_demand,
                        "demand_ratio": ratio,
                        "high_balk": high_balk,
                        "seed": seed,
                    },
                )
                rows.append(row)

                count += 1
                if count % 500 == 0:
                    print(f"  {count}/{total} runs complete")

    return pd.DataFrame(rows)


# ============================================================
# Summarize
# ============================================================

def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["lambda_per_day", "demand_ratio", "high_balk"])
        ["mean_offered_booking_delay"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["se"]
    return summary


# ============================================================
# Plot
# ============================================================

def plot_offered_delay(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.viridis
    ratios = sorted(summary["demand_ratio"].unique())
    colors = [cmap(i / (len(ratios) - 1)) for i in range(len(ratios))]

    for (ratio, color) in zip(ratios, colors):
        sub = summary[summary["demand_ratio"] == ratio].sort_values("high_balk")
        lam = sub["lambda_per_day"].iloc[0]

        ax.errorbar(
            sub["high_balk"],
            sub["mean"],
            yerr=sub["ci95"],
            capsize=3,
            marker="o",
            linewidth=2,
            color=color,
            label=f"λ={int(lam)}/class (λ/S={ratio:.2f})",
        )

    ax.set_title("Mean Offered Booking Delay vs. Balking Step")
    ax.set_xlabel("High balking probability")
    ax.set_ylabel("Mean offered booking delay (days)")
    ax.set_xticks(BALKING_STEPS)
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="Arrival rate",
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

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw = run_sweep()
    raw.to_csv(RAW_DIR / "results.csv", index=False)

    summary = summarize(raw)
    summary.to_csv(RAW_DIR / "summary.csv", index=False)

    plot_offered_delay(summary, FIGURE_DIR / "offered_delay_by_arrival_rate.png")

    print("Done.")


if __name__ == "__main__":
    main()