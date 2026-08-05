"""
Pre-threshold vs. post-threshold balking rate: which drives the
class served-rate gap?

Controlled comparison where both scenarios have:
    - The same between-class gap magnitude (0.30)
    - The same Class 2 step size (0.20)
    - Class 1 identical in both (b_0=0.00, b_1=0.50)

Scenario A — Gap is pre-threshold:
    Class 1: b_0=0.00, b_1=0.50  (step=0.50)
    Class 2: b_0=0.30, b_1=0.50  (step=0.20)
    Between-class pre-threshold gap: 0.30
    Between-class post-threshold gap: 0.00

Scenario B — Gap is post-threshold:
    Class 1: b_0=0.00, b_1=0.50  (step=0.50)
    Class 2: b_0=0.00, b_1=0.20  (step=0.20)
    Between-class pre-threshold gap: 0.00
    Between-class post-threshold gap: 0.30

Outputs
-------
outputs/pre_vs_post_threshold/figures/pre_vs_post_comparison.png
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

SEEDS = range(1, 101)

SCENARIOS = {
    "A: Pre-threshold gap\n(0.30 gap below threshold)": {
        1: {"low": 0.00, "high": 0.50},
        2: {"low": 0.30, "high": 0.50},
    },
    "B: Post-threshold gap\n(0.30 gap above threshold)": {
        1: {"low": 0.00, "high": 0.50},
        2: {"low": 0.00, "high": 0.20},
    },
}


def make_config(base_config, class_balking, seed):
    new_classes = {}
    for cid, params in base_config.classes.items():
        balk_cfg = class_balking[cid]
        new_classes[cid] = replace(
            params,
            balk_prob=ThresholdRule(
                threshold=params.balk_prob.threshold,
                low=balk_cfg["low"],
                high=balk_cfg["high"],
            ),
        )
    return replace(base_config, classes=new_classes, seed=int(seed))


def run_scenarios():
    base_config = load_config(CONFIG_PATH)
    rows = []

    for scenario_name, class_balking in SCENARIOS.items():
        print(f"Running: {scenario_name}")
        for seed in SEEDS:
            config = make_config(base_config, class_balking, seed)
            sim = ClinicAppointmentSimulation(config)
            results = sim.run()
            rows.extend(
                class_result_rows(results, {"scenario": scenario_name, "seed": seed})
            )

    return pd.DataFrame(rows)


def plot_comparison(raw, output_path):
    summary = (
        raw.groupby(["scenario", "class_id"])["percent_serviced"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["se"]

    scenarios = list(SCENARIOS.keys())
    x = np.arange(len(scenarios))
    width = 0.30

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, cid in enumerate([1, 2]):
        sub = summary[summary["class_id"] == cid].copy()
        sub = sub.set_index("scenario").loc[scenarios].reset_index()
        ax.bar(
            x + (i - 0.5) * width, sub["mean"], width,
            yerr=sub["ci95"], capsize=4,
            label=f"Class {cid}", alpha=0.8,
        )

    for j, scenario in enumerate(scenarios):
        s = summary[summary["scenario"] == scenario]
        c1 = s[s["class_id"] == 1]["mean"].values[0]
        c2 = s[s["class_id"] == 2]["mean"].values[0]
        gap = abs(c1 - c2)
        mid = (c1 + c2) / 2
        ax.annotate(
            f"gap = {gap:.3f}", xy=(j, mid),
            ha="center", fontsize=10, fontweight="bold",
        )

    ax.set_title(
        "Served-Rate Gap: Pre-Threshold vs. Post-Threshold Difference\n"
        "(same gap magnitude = 0.30, same Class 2 step = 0.20)"
    )
    ax.set_ylabel("Served rate")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw = run_scenarios()
    raw.to_csv(RAW_DIR / "results.csv", index=False)
    plot_comparison(raw, FIGURE_DIR / "pre_vs_post_comparison.png")
    print("Done.")


if __name__ == "__main__":
    main()