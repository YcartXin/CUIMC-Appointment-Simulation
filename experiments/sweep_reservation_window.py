"""
Step 5: Figures for reservation window analysis.

Reads summary CSVs from the reservation window sweep and produces:
    1a. Objective heatmap — equal weights
    1b. Objective heatmap — priority weights
    2a-e. Objective heatmap per balking threshold combo (separate files)
    3. No-offer rate heatmaps by class
    4. Offered delay heatmaps by class
    5. Objective divergence: main vs pooled

Outputs
-------
outputs/reservation_window/figures/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

SUMMARY_DIR = REPO_DIR / "outputs" / "reservation_window" / "summary"
FIGURE_DIR = REPO_DIR / "outputs" / "reservation_window" / "figures"

H_VALUES = [2, 5, 7, 10, 14]

THRESHOLD_COMBOS = [
    ((9, 9),   "symmetric",     "τ=(9,9) Symmetric baseline"),
    ((5, 9),   "c1_sensitive",  "τ=(5,9) Class 1 delay-sensitive"),
    ((9, 5),   "c2_sensitive",  "τ=(9,5) Class 2 delay-sensitive"),
    ((12, 12), "both_tolerant", "τ=(12,12) Both tolerant"),
    ((5, 12),  "max_asymmetry", "τ=(5,12) Maximum asymmetry"),
]


# ============================================================
# Helpers
# ============================================================

def pivot_to_heatmap(df, value_col):
    piv = df.pivot_table(index="h1", columns="h2", values=value_col, aggfunc="first")
    piv = piv.reindex(index=H_VALUES, columns=H_VALUES)
    return piv


def plot_heatmap(ax, matrix, title, cmap="RdYlGn", fmt=".3f", vmin=None, vmax=None):
    im = ax.imshow(matrix.values, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(H_VALUES)))
    ax.set_xticklabels(H_VALUES)
    ax.set_yticks(range(len(H_VALUES)))
    ax.set_yticklabels(H_VALUES)
    ax.set_xlabel("H₂ (Class 2 window)")
    ax.set_ylabel("H₁ (Class 1 window)")
    ax.set_title(title, fontsize=11)

    for i in range(len(H_VALUES)):
        for j in range(len(H_VALUES)):
            val = matrix.values[i, j]
            if not np.isnan(val):
                color = ("white" if abs(val - np.nanmean(matrix.values))
                         > 0.5 * np.nanstd(matrix.values) else "black")
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=8, color=color)

    return im


def single_objective_heatmap(obj_df, regime, balk_high, filename, subtitle):
    """Single heatmap for one weight regime."""
    sub = obj_df[(obj_df["tau1"] == 9) & (obj_df["tau2"] == 9) &
                 (obj_df["balk_high"] == balk_high) &
                 (obj_df["weight_regime"] == regime)].copy()

    matrix = pivot_to_heatmap(sub, "obj_professor_mean")
    w1 = sub["w1"].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_heatmap(ax, matrix, f"w₁={w1:.0f}, w₂=1.0")

    fig.suptitle(f"Main Objective: w₁(Y₁/A₁) + w₂(Y₂/A₂)\n{subtitle}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Figures 1a, 1b: Objective — equal and priority (separate)
# ============================================================

def fig1a_objective_equal(obj_df, balk_high=0.5):
    single_objective_heatmap(
        obj_df, "equal", balk_high,
        "fig1a_objective_equal.png",
        f"Equal weights, τ=(9,9), balk_high={balk_high}",
    )


def fig1b_objective_priority(obj_df, balk_high=0.5):
    single_objective_heatmap(
        obj_df, "priority", balk_high,
        "fig1b_objective_priority.png",
        f"Priority weights, τ=(9,9), balk_high={balk_high}",
    )


# ============================================================
# Figures 2a-e: Objective by threshold combo (one file each)
# ============================================================

def fig2_objective_by_threshold(obj_df, balk_high=0.5):
    sub = obj_df[(obj_df["weight_regime"] == "equal") &
                 (obj_df["balk_high"] == balk_high)].copy()

    for (tau1, tau2), slug, label in THRESHOLD_COMBOS:
        data = sub[(sub["tau1"] == tau1) & (sub["tau2"] == tau2)]
        if len(data) == 0:
            continue

        matrix = pivot_to_heatmap(data, "obj_professor_mean")

        fig, ax = plt.subplots(figsize=(6, 5))
        plot_heatmap(ax, matrix, label)

        fig.suptitle(f"Main Objective by Balking Threshold\n"
                     f"Equal weights (w₁=w₂=1), balk_high={balk_high}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()

        path = FIGURE_DIR / f"fig2_{slug}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ============================================================
# Figure 3: No-offer rate heatmaps by class
# ============================================================

def fig3_no_offer_heatmaps(nof_df, balk_high=0.5):
    sub = nof_df[(nof_df["tau1"] == 9) & (nof_df["tau2"] == 9) &
                 (nof_df["balk_high"] == balk_high)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cid in zip(axes, [1, 2]):
        data = sub[sub["class_id"] == cid]
        matrix = pivot_to_heatmap(data, "mean")
        plot_heatmap(ax, matrix, f"Class {cid} No-Offer Rate",
                     cmap="RdYlGn_r", fmt=".2f")

    fig.suptitle(f"No-Offer Rate by Class\nτ=(9,9), balk_high={balk_high}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = FIGURE_DIR / "fig3_no_offer_rate.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Figure 4: Offered delay heatmaps by class
# ============================================================

def fig4_offered_delay_heatmaps(class_summary, balk_high=0.5):
    sub = class_summary[
        (class_summary["metric"] == "mean_offered_booking_delay") &
        (class_summary["tau1"] == 9) & (class_summary["tau2"] == 9) &
        (class_summary["balk_high"] == balk_high)
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cid in zip(axes, [1, 2]):
        data = sub[sub["class_id"] == cid]
        matrix = pivot_to_heatmap(data, "mean")
        plot_heatmap(ax, matrix, f"Class {cid} Mean Offered Delay",
                     cmap="YlOrRd", fmt=".1f")

    fig.suptitle(f"Mean Offered Booking Delay by Class\nτ=(9,9), balk_high={balk_high}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = FIGURE_DIR / "fig4_offered_delay.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Figure 5: Objective divergence — main vs pooled
# ============================================================

def fig5_objective_divergence(obj_df, balk_high=0.5):
    sub = obj_df[(obj_df["tau1"] == 9) & (obj_df["tau2"] == 9) &
                 (obj_df["balk_high"] == balk_high)].copy()

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = {"equal": "#2ca02c", "priority": "#9467bd"}
    markers = {"equal": "o", "priority": "s"}

    for regime in ["equal", "priority"]:
        data = sub[sub["weight_regime"] == regime]
        ax.scatter(
            data["obj_pooled_mean"],
            data["obj_professor_mean"],
            c=colors[regime], marker=markers[regime],
            s=50, alpha=0.7,
            label=f"w₁={'1.0' if regime == 'equal' else '2.0'}",
        )

        for _, row in data.iterrows():
            h1, h2 = int(row["h1"]), int(row["h2"])
            if (h1, h2) in [(2, 14), (14, 2), (14, 14), (7, 7), (2, 2)]:
                ax.annotate(
                    f"({h1},{h2})",
                    xy=(row["obj_pooled_mean"], row["obj_professor_mean"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7, color=colors[regime],
                )

    all_vals = sub[["obj_pooled_mean", "obj_professor_mean"]].values.flatten()
    lo, hi = min(all_vals) * 0.95, max(all_vals) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Pooled objective: (w₁Y₁+w₂Y₂)/(w₁A₁+w₂A₂)")
    ax.set_ylabel("Main objective: w₁(Y₁/A₁)+w₂(Y₂/A₂)")
    ax.set_title(f"Objective Function Divergence\nτ=(9,9), balk_high={balk_high}")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIGURE_DIR / "fig5_objective_divergence.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Figure 6: Objective by balking rate
# ============================================================

def fig6_objective_by_balk_rate(obj_df):
    """One heatmap per balk_high value at baseline threshold, equal weights."""
    sub = obj_df[(obj_df["weight_regime"] == "equal") &
                 (obj_df["tau1"] == 9) & (obj_df["tau2"] == 9)].copy()

    balk_highs = sorted(sub["balk_high"].unique())

    for bh in balk_highs:
        data = sub[sub["balk_high"] == bh]
        matrix = pivot_to_heatmap(data, "obj_professor_mean")

        fig, ax = plt.subplots(figsize=(6, 5))
        plot_heatmap(ax, matrix, f"balk_high = {bh}")

        fig.suptitle("Main Objective by Post-Threshold Balking Rate\n"
                     "Equal weights (w₁=w₂=1), τ=(9,9)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()

        path = FIGURE_DIR / f"fig6_balk_rate_{bh:.1f}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    obj_df = pd.read_csv(SUMMARY_DIR / "objectives.csv")
    class_summary = pd.read_csv(SUMMARY_DIR / "class_summary.csv")
    nof_df = pd.read_csv(SUMMARY_DIR / "no_offer_rates.csv")

    balk_highs = sorted(obj_df["balk_high"].unique())
    baseline_balk = 0.5 if 0.5 in balk_highs else balk_highs[0]
    print(f"Using balk_high={baseline_balk} for baseline figures")
    print()

    fig1a_objective_equal(obj_df, baseline_balk)
    fig1b_objective_priority(obj_df, baseline_balk)
    fig2_objective_by_threshold(obj_df, baseline_balk)
    fig3_no_offer_heatmaps(nof_df, baseline_balk)
    fig4_offered_delay_heatmaps(class_summary, baseline_balk)
    fig5_objective_divergence(obj_df, baseline_balk)
    fig6_objective_by_balk_rate(obj_df)

    print()
    print(f"All figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()