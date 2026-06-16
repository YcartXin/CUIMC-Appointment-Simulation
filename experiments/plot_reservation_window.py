"""
Step 5: Figures for reservation window analysis.

Reads summary CSVs from the reservation window sweep and produces:
    1. Objective function heatmaps across (H1, H2) — equal vs priority weights
    2. Objective heatmaps by balking threshold combo
    3. No-offer rate heatmaps by class
    4. Offered delay heatmaps by class
    5. Objective divergence: professor's vs pooled

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
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

SUMMARY_DIR = REPO_DIR / "outputs" / "reservation_window" / "summary"
FIGURE_DIR = REPO_DIR / "outputs" / "reservation_window" / "figures"

H_VALUES = [2, 5, 7, 10, 14]

THRESHOLD_LABELS = {
    (9, 9): "τ=(9,9) symmetric",
    (5, 9): "τ=(5,9) C1 sensitive",
    (9, 5): "τ=(9,5) C2 sensitive",
    (12, 12): "τ=(12,12) both tolerant",
    (5, 12): "τ=(5,12) max asymmetry",
}


# ============================================================
# Helpers
# ============================================================

def pivot_to_heatmap(df, value_col):
    """Pivot a filtered dataframe into a (H1 × H2) matrix for heatmap."""
    piv = df.pivot_table(index="h1", columns="h2", values=value_col, aggfunc="first")
    piv = piv.reindex(index=H_VALUES, columns=H_VALUES)
    return piv


def plot_heatmap(ax, matrix, title, cmap="RdYlGn", fmt=".3f", vmin=None, vmax=None):
    """Plot a single heatmap on an axis."""
    im = ax.imshow(matrix.values, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(H_VALUES)))
    ax.set_xticklabels(H_VALUES)
    ax.set_yticks(range(len(H_VALUES)))
    ax.set_yticklabels(H_VALUES)
    ax.set_xlabel("H₂ (Class 2 window)")
    ax.set_ylabel("H₁ (Class 1 window)")
    ax.set_title(title, fontsize=10)

    # Annotate cells
    for i in range(len(H_VALUES)):
        for j in range(len(H_VALUES)):
            val = matrix.values[i, j]
            if not np.isnan(val):
                color = "white" if abs(val - np.nanmean(matrix.values)) > 0.5 * np.nanstd(matrix.values) else "black"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=7, color=color)

    return im


# ============================================================
# Figure 1: Objective heatmaps — equal vs priority (baseline threshold)
# ============================================================

def fig1_objective_equal_vs_priority(obj_df, balk_high=0.5):
    """Two heatmaps: professor's objective at equal and priority weights."""
    sub = obj_df[(obj_df["tau1"] == 9) & (obj_df["tau2"] == 9) &
                 (obj_df["balk_high"] == balk_high)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, regime in zip(axes, ["equal", "priority"]):
        data = sub[sub["weight_regime"] == regime]
        matrix = pivot_to_heatmap(data, "obj_professor_mean")

        w1 = data["w1"].iloc[0]
        title = f"w₁={w1:.0f}, w₂=1.0"
        plot_heatmap(ax, matrix, title)

    fig.suptitle("Professor's Objective: w₁(Y₁/A₁) + w₂(Y₂/A₂)\n"
                 f"τ=(9,9), balk_high={balk_high}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = FIGURE_DIR / "fig1_objective_equal_vs_priority.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Figure 2: Objective by threshold combo (equal weights)
# ============================================================

def fig2_objective_by_threshold(obj_df, balk_high=0.5):
    """One heatmap per threshold combo, equal weights."""
    sub = obj_df[(obj_df["weight_regime"] == "equal") &
                 (obj_df["balk_high"] == balk_high)].copy()

    combos = list(THRESHOLD_LABELS.keys())
    fig, axes = plt.subplots(1, len(combos), figsize=(4 * len(combos), 4))

    # Find global min/max for consistent colorscale
    all_vals = []
    for tau1, tau2 in combos:
        data = sub[(sub["tau1"] == tau1) & (sub["tau2"] == tau2)]
        if len(data) > 0:
            all_vals.extend(data["obj_professor_mean"].values)
    vmin, vmax = min(all_vals), max(all_vals)

    for ax, (tau1, tau2) in zip(axes, combos):
        data = sub[(sub["tau1"] == tau1) & (sub["tau2"] == tau2)]
        matrix = pivot_to_heatmap(data, "obj_professor_mean")
        im = plot_heatmap(ax, matrix, THRESHOLD_LABELS[(tau1, tau2)],
                          vmin=vmin, vmax=vmax)

    fig.suptitle("Professor's Objective by Balking Threshold\n"
                 f"Equal weights (w₁=w₂=1), balk_high={balk_high}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = FIGURE_DIR / "fig2_objective_by_threshold.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Figure 3: No-offer rate heatmaps by class
# ============================================================

def fig3_no_offer_heatmaps(nof_df, balk_high=0.5):
    """No-offer rate heatmap per class at baseline threshold."""
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
    """Offered delay heatmap per class at baseline threshold."""
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
# Figure 5: Objective divergence — professor's vs pooled
# ============================================================

def fig5_objective_divergence(obj_df, balk_high=0.5):
    """Scatter: professor's vs pooled objective, colored by weight regime."""
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
            c=colors[regime],
            marker=markers[regime],
            s=50,
            alpha=0.7,
            label=f"w₁={'1.0' if regime == 'equal' else '2.0'}",
        )

        # Annotate a few interesting points
        for _, row in data.iterrows():
            h1, h2 = int(row["h1"]), int(row["h2"])
            if (h1, h2) in [(2, 14), (14, 2), (14, 14), (7, 7), (2, 2)]:
                ax.annotate(
                    f"({h1},{h2})",
                    xy=(row["obj_pooled_mean"], row["obj_professor_mean"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7, color=colors[regime],
                )

    # Reference line
    all_vals = sub[["obj_pooled_mean", "obj_professor_mean"]].values.flatten()
    lo, hi = min(all_vals) * 0.95, max(all_vals) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Pooled objective: (w₁Y₁+w₂Y₂)/(w₁A₁+w₂A₂)")
    ax.set_ylabel("Professor's objective: w₁(Y₁/A₁)+w₂(Y₂/A₂)")
    ax.set_title("Objective Function Divergence\n"
                 f"τ=(9,9), balk_high={balk_high}")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    path = FIGURE_DIR / "fig5_objective_divergence.png"
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

    # Get available balk_high values
    balk_highs = sorted(obj_df["balk_high"].unique())
    baseline_balk = 0.5 if 0.5 in balk_highs else balk_highs[0]
    print(f"Using balk_high={baseline_balk} for baseline figures")
    print()

    fig1_objective_equal_vs_priority(obj_df, baseline_balk)
    fig2_objective_by_threshold(obj_df, baseline_balk)
    fig3_no_offer_heatmaps(nof_df, baseline_balk)
    fig4_offered_delay_heatmaps(class_summary, baseline_balk)
    fig5_objective_divergence(obj_df, baseline_balk)

    print()
    print(f"All figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()