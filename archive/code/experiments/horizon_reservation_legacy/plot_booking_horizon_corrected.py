"""
Step 3: Figures for the class-specific booking-horizon analysis.

Reads the summary CSVs and produces, for each objective:
    1a/1b/1c  Objective heatmap over (H1, H2) — equal weights, Class-1
              weight 2, and Class-1 weight 3, each at arrival rates 50
              and 25 per class
    2a-e   Objective heatmap per balking-threshold combination
    6      Objective heatmap per post-threshold balking rate
    7      Class-2-threshold counterbalance: how much Class-1 weight is needed,
           at tau=(9,5), to pull the optimum back toward the symmetric optimum
    8      Balking-rate x weight interaction at tau=(9,9)
    9      Arrival-rate comparison at tau=(9,9)
Objective-independent diagnostics (written once):
    3      No-offer rate by class
    4      Offered delay by class

Outputs
-------
outputs/booking_horizon/figures/<objective>/*.png   (objective-specific)
outputs/booking_horizon/figures/*.png                (shared diagnostics)
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

OUTPUT_DIR = REPO_DIR / "outputs" / "booking_horizon"
SUMMARY_DIR = OUTPUT_DIR / "summary"
FIGURE_DIR = OUTPUT_DIR / "figures"

BASELINE_BALK = 0.5
BASELINE_LAMBDA = 50

OBJECTIVES = {
    "service_rate": dict(
        mean="obj_service_rate_mean", slug="service_rate",
        name="Objective - Service Rate", formula="w\u2081(Y\u2081/A\u2081) + w\u2082(Y\u2082/A\u2082)"),
    "slot_util": dict(
        mean="obj_slot_util_mean", slug="slot_utilization",
        name="Objective - Average Slot Utilization", formula="w\u2081U\u2081 + w\u2082U\u2082"),
}

THRESHOLD_COMBOS = [
    ((9, 9), "symmetric", "\u03c4=(9,9) Symmetric baseline"),
    ((5, 9), "c1_sensitive", "\u03c4=(5,9) Class 1 delay-sensitive"),
    ((9, 5), "c2_sensitive", "\u03c4=(9,5) Class 2 delay-sensitive"),
    ((12, 12), "both_tolerant", "\u03c4=(12,12) Both tolerant"),
    ((5, 12), "max_asymmetry", "\u03c4=(5,12) Maximum asymmetry"),
]

HEADLINE_WEIGHTS = [0.25, 1.0, 2.0, 3.0]      # balk x weight grid
COUNTERBALANCE_WEIGHTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
FIG1_ARRIVAL_RATES = [BASELINE_LAMBDA, 25]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def h_axis(df):
    return sorted(df["h1"].unique())


def matrix(df, value_col, h_values):
    piv = df.pivot_table(index="h1", columns="h2", values=value_col, aggfunc="first")
    return piv.reindex(index=h_values, columns=h_values)


def draw(ax, mat, h_values, title, cmap="RdYlGn", star=True, annotate=None):
    im = ax.imshow(mat.values, cmap=cmap, aspect="equal", origin="upper")
    ax.set_xticks(range(len(h_values)))
    ax.set_xticklabels(h_values, fontsize=7)
    ax.set_yticks(range(len(h_values)))
    ax.set_yticklabels(h_values, fontsize=7)
    ax.set_xlabel("H\u2082 (Class 2 window)", fontsize=8)
    ax.set_ylabel("H\u2081 (Class 1 window)", fontsize=8)
    ax.set_title(title, fontsize=10)
    if star:
        vals = mat.values
        if not np.all(np.isnan(vals)):
            i, j = np.unravel_index(np.nanargmax(vals), vals.shape)
            ax.scatter(j, i, marker="*", s=160, c="black",
                       edgecolors="white", linewidths=0.6, zorder=5)
    if annotate is not None:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:{annotate}}", ha="center", va="center",
                            fontsize=6, color="black")
    return im


def argmax_cell(df, value_col, h_values):
    mat = matrix(df, value_col, h_values)
    vals = mat.values
    i, j = np.unravel_index(np.nanargmax(vals), vals.shape)
    return h_values[i], h_values[j]


def base_slice(obj, tau1, tau2, balk, lam, w1=None):
    sub = obj[(obj.tau1 == tau1) & (obj.tau2 == tau2) &
              (obj.balk_high == balk) & (obj.arrival_rate == lam)]
    if w1 is not None:
        sub = sub[np.isclose(sub.w1, w1)]
    return sub


# ----------------------------------------------------------------------
# Objective-specific figures
# ----------------------------------------------------------------------

def fig1_equal_priority(obj, cfg, fdir):
    """Create equal-, w1=2-, and w1=3-weight heatmaps at arrival rates 50 and 25."""
    hv = h_axis(obj)
    panels = [
        (1.0, "Equal weights (w\u2081=1)", "fig1a_objective_equal"),
        (2.0, "Class-1 priority (w\u2081=2)", "fig1b_objective_priority"),
        (3.0, "Class-1 priority (w\u2081=3)", "fig1c_objective_priority_w3"),
    ]

    for lam in FIG1_ARRIVAL_RATES:
        suffix = "" if np.isclose(lam, BASELINE_LAMBDA) else f"_arrival{float(lam):g}"

        for w1, label, stem in panels:
            sub = base_slice(obj, 9, 9, BASELINE_BALK, lam, w1)
            if sub.empty:
                raise ValueError(
                    "No objective-summary rows found for "
                    f"tau=(9,9), balk={BASELINE_BALK}, arrival_rate={lam}, "
                    f"w1={w1}. Re-run summarize_booking_horizon.py and confirm "
                    "that objectives.csv contains this weight and arrival rate."
                )

            fig, ax = plt.subplots(figsize=(6, 5))
            im = draw(
                ax,
                matrix(sub, cfg["mean"], hv),
                hv,
                label,
                annotate=".3f" if len(hv) <= 5 else None,
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(
                f"{cfg['name']}: {cfg['formula']}\n"
                f"\u03c4=(9,9), balk={BASELINE_BALK}, \u03bb={float(lam):g}",
                fontsize=11,
                fontweight="bold",
            )
            fig.tight_layout()
            _save(fig, fdir / f"{stem}{suffix}.png")


def fig2_by_threshold(obj, cfg, fdir):
    hv = h_axis(obj)
    for (tau1, tau2), slug, label in THRESHOLD_COMBOS:
        sub = base_slice(obj, tau1, tau2, BASELINE_BALK, BASELINE_LAMBDA, 1.0)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = draw(ax, matrix(sub, cfg["mean"], hv), hv, label,
                  annotate=".3f" if len(hv) <= 5 else None)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{cfg['name']} by balking threshold\n"
                     f"Equal weights, balk={BASELINE_BALK}, \u03bb={BASELINE_LAMBDA}",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        _save(fig, fdir / f"fig2_{slug}.png")


def fig6_by_balk_rate(obj, cfg, fdir):
    hv = h_axis(obj)
    for bh in sorted(obj.balk_high.unique()):
        sub = base_slice(obj, 9, 9, bh, BASELINE_LAMBDA, 1.0)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = draw(ax, matrix(sub, cfg["mean"], hv), hv, f"balk = {bh}",
                  annotate=".3f" if len(hv) <= 5 else None)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{cfg['name']} by post-threshold balking rate\n"
                     f"Equal weights, \u03c4=(9,9), \u03bb={BASELINE_LAMBDA}",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        _save(fig, fdir / f"fig6_balk_rate_{bh:.1f}.png")


def fig7_counterbalance(obj, cfg, fdir):
    """tau=(9,5): how much Class-1 weight pulls the optimum back."""
    hv = h_axis(obj)
    weights = [w for w in COUNTERBALANCE_WEIGHTS
               if np.any(np.isclose(obj.w1.unique()[:, None], w))]
    n = len(weights)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.0))
    if n == 1:
        axes = [axes]
    vmaxs, mats = [], []
    for w1 in weights:
        sub = base_slice(obj, 9, 5, BASELINE_BALK, BASELINE_LAMBDA, w1)
        mats.append(matrix(sub, cfg["mean"], hv))
    vmin = min(np.nanmin(m.values) for m in mats)
    vmax = max(np.nanmax(m.values) for m in mats)
    for ax, w1, mat in zip(axes, weights, mats):
        im = ax.imshow(mat.values, cmap="RdYlGn", origin="upper", vmin=vmin, vmax=vmax)
        i, j = np.unravel_index(np.nanargmax(mat.values), mat.values.shape)
        ax.scatter(j, i, marker="*", s=120, c="black", edgecolors="white", linewidths=0.6)
        ax.set_title(f"w\u2081={w1:g}\nopt=({hv[i]},{hv[j]})", fontsize=9)
        ax.set_xticks(range(len(hv)))
        ax.set_xticklabels(hv, fontsize=6)
        ax.set_yticks(range(len(hv)))
        ax.set_yticklabels(hv, fontsize=6)
        ax.set_xlabel("H\u2082", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("H\u2081", fontsize=7)
    fig.suptitle(f"{cfg['name']}: Class-1 weight counterbalance at \u03c4=(9,5)\n"
                 f"balk={BASELINE_BALK}, \u03bb={BASELINE_LAMBDA} \u00b7 star = optimum",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, fdir / "fig7_counterbalance.png")

    # companion: optimal H1*, H2* versus w1
    h1s, h2s = [], []
    for w1 in weights:
        sub = base_slice(obj, 9, 5, BASELINE_BALK, BASELINE_LAMBDA, w1)
        a, b = argmax_cell(sub, cfg["mean"], hv)
        h1s.append(a)
        h2s.append(b)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(weights, h1s, "o-", label="optimal H\u2081", linewidth=2)
    ax.plot(weights, h2s, "s--", label="optimal H\u2082", linewidth=2)
    ax.set_xlabel("Class-1 weight w\u2081")
    ax.set_ylabel("Optimal booking horizon (days)")
    ax.set_title(f"{cfg['name']}: optimal horizon vs Class-1 weight\n\u03c4=(9,5)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, fdir / "fig7b_counterbalance_optimum.png")


def fig8_balk_weight(obj, cfg, fdir):
    """tau=(9,9): balking rate x Class-1 weight interaction."""
    hv = h_axis(obj)
    balks = sorted(obj.balk_high.unique())
    weights = [w for w in HEADLINE_WEIGHTS
               if np.any(np.isclose(obj.w1.unique()[:, None], w))]
    nr, nc = len(balks), len(weights)
    fig, axes = plt.subplots(nr, nc, figsize=(2.5 * nc, 2.6 * nr))
    axes = np.atleast_2d(axes)
    mats = {}
    for bi, bh in enumerate(balks):
        for wi, w1 in enumerate(weights):
            sub = base_slice(obj, 9, 9, bh, BASELINE_LAMBDA, w1)
            mats[(bi, wi)] = matrix(sub, cfg["mean"], hv)
    for bi, bh in enumerate(balks):
        row = [mats[(bi, wi)] for wi in range(nc)]
        vmin = min(np.nanmin(m.values) for m in row)
        vmax = max(np.nanmax(m.values) for m in row)
        for wi, w1 in enumerate(weights):
            ax = axes[bi, wi]
            mat = mats[(bi, wi)]
            ax.imshow(mat.values, cmap="RdYlGn", origin="upper", vmin=vmin, vmax=vmax)
            i, j = np.unravel_index(np.nanargmax(mat.values), mat.values.shape)
            ax.scatter(j, i, marker="*", s=90, c="black", edgecolors="white", linewidths=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if bi == 0:
                ax.set_title(f"w\u2081={w1:g}", fontsize=9)
            if wi == 0:
                ax.set_ylabel(f"balk={bh}", fontsize=9)
            ax.set_xlabel(f"opt=({hv[i]},{hv[j]})", fontsize=7)
    fig.suptitle(f"{cfg['name']}: balking rate \u00d7 Class-1 weight at \u03c4=(9,9)\n"
                 f"\u03bb={BASELINE_LAMBDA} \u00b7 rows = balk rate, cols = weight \u00b7 star = optimum",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, fdir / "fig8_balk_weight.png")


def fig9_arrival_rate(obj, cfg, fdir):
    hv = h_axis(obj)
    lams = sorted(obj.arrival_rate.unique())
    if len(lams) < 2:
        return
    fig, axes = plt.subplots(1, len(lams), figsize=(5.5 * len(lams), 5))
    axes = np.atleast_1d(axes)
    mats = [matrix(base_slice(obj, 9, 9, BASELINE_BALK, lam, 1.0), cfg["mean"], hv)
            for lam in lams]
    vmin = min(np.nanmin(m.values) for m in mats)
    vmax = max(np.nanmax(m.values) for m in mats)
    for ax, lam, mat in zip(axes, lams, mats):
        im = ax.imshow(mat.values, cmap="RdYlGn", origin="upper", vmin=vmin, vmax=vmax)
        i, j = np.unravel_index(np.nanargmax(mat.values), mat.values.shape)
        ax.scatter(j, i, marker="*", s=140, c="black", edgecolors="white", linewidths=0.6)
        ax.set_title(f"\u03bb={lam}  opt=({hv[i]},{hv[j]})", fontsize=10)
        ax.set_xticks(range(len(hv))); ax.set_xticklabels(hv, fontsize=7)
        ax.set_yticks(range(len(hv))); ax.set_yticklabels(hv, fontsize=7)
        ax.set_xlabel("H\u2082", fontsize=8)
        ax.set_ylabel("H\u2081", fontsize=8)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.046, pad=0.04)
    fig.suptitle(f"{cfg['name']} by arrival rate\nEqual weights, \u03c4=(9,9), balk={BASELINE_BALK}",
                 fontsize=11, fontweight="bold")
    _save(fig, fdir / "fig9_arrival_rate.png")


# ----------------------------------------------------------------------
# Objective-independent diagnostics
# ----------------------------------------------------------------------

def fig3_no_offer(nof):
    hv = sorted(nof["h1"].unique())
    sub = nof[(nof.tau1 == 9) & (nof.tau2 == 9) &
              (nof.balk_high == BASELINE_BALK) & (nof.arrival_rate == BASELINE_LAMBDA)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cid in zip(axes, [1, 2]):
        im = draw(ax, matrix(sub[sub.class_id == cid], "mean", hv), hv,
                  f"Class {cid} no-offer rate", cmap="RdYlGn_r", star=False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"No-offer rate by class\n\u03c4=(9,9), balk={BASELINE_BALK}, \u03bb={BASELINE_LAMBDA}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, FIGURE_DIR / "fig3_no_offer_rate.png")


def fig4_offered_delay(cls_sum):
    hv = sorted(cls_sum["h1"].unique())
    sub = cls_sum[(cls_sum.metric == "mean_offered_booking_delay") &
                  (cls_sum.tau1 == 9) & (cls_sum.tau2 == 9) &
                  (cls_sum.balk_high == BASELINE_BALK) &
                  (cls_sum.arrival_rate == BASELINE_LAMBDA)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cid in zip(axes, [1, 2]):
        im = draw(ax, matrix(sub[sub.class_id == cid], "mean", hv), hv,
                  f"Class {cid} mean offered delay", cmap="YlOrRd", star=False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Mean offered booking delay by class\n\u03c4=(9,9), balk={BASELINE_BALK}, \u03bb={BASELINE_LAMBDA}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, FIGURE_DIR / "fig4_offered_delay.png")


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(REPO_DIR)}")


def main():
    obj = pd.read_csv(SUMMARY_DIR / "objectives.csv")
    cls_sum = pd.read_csv(SUMMARY_DIR / "class_summary.csv")
    nof = pd.read_csv(SUMMARY_DIR / "no_offer_rates.csv")

    for key, cfg in OBJECTIVES.items():
        fdir = FIGURE_DIR / cfg["slug"]
        print(f"{cfg['name']} -> {fdir.relative_to(REPO_DIR)}")
        fig1_equal_priority(obj, cfg, fdir)
        fig2_by_threshold(obj, cfg, fdir)
        fig6_by_balk_rate(obj, cfg, fdir)
        fig7_counterbalance(obj, cfg, fdir)
        fig8_balk_weight(obj, cfg, fdir)
        fig9_arrival_rate(obj, cfg, fdir)

    print("Shared diagnostics:")
    fig3_no_offer(nof)
    fig4_offered_delay(cls_sum)
    print("\nAll figures written under", FIGURE_DIR.relative_to(REPO_DIR))


if __name__ == "__main__":
    main()
