"""
H4 combined comparison: balking vs. no-show loss timing.

Reads existing summary CSVs from the Class 1 balking sweep and the
Class 1 no-show sweep and produces a single two-panel figure:

    Panel A — Overall utilization (flat for balking, falling for no-show)
    Panel B — Class 1 served rate (falling in both)

No sweeps are rerun; this script only reads and plots.

Outputs
-------
outputs/h4_balking_vs_noshow/figures/h4_loss_timing_comparison.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# Path setup
# ============================================================

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

# ============================================================
# Style constants (from analysis/plot_style.py)
# ============================================================

BALKING_COLOR = "#9467bd"
NO_SHOW_COLOR = "#2ca02c"

BALKING_STYLE = dict(color=BALKING_COLOR, linewidth=2, marker="s")
NO_SHOW_STYLE = dict(color=NO_SHOW_COLOR, linewidth=2, marker="^")

# ============================================================
# Data paths
# ============================================================

BALKING_AGG = (
    REPO_DIR / "outputs" / "class1_balking" / "summary" / "aggregate_summary.csv"
)
BALKING_CLASS = (
    REPO_DIR / "outputs" / "class1_balking" / "summary" / "class_summary.csv"
)
NO_SHOW_AGG = (
    REPO_DIR / "outputs" / "class1_no_show" / "summary" / "aggregate_summary.csv"
)
NO_SHOW_CLASS = (
    REPO_DIR / "outputs" / "class1_no_show" / "summary" / "class_summary.csv"
)

FIGURE_DIR = REPO_DIR / "outputs" / "h4_balking_vs_noshow" / "figures"


# ============================================================
# Data loading helpers
# ============================================================

def load_metric(path: Path, x_col: str, metric: str) -> pd.DataFrame:
    """Load one metric from an aggregate summary CSV."""
    df = pd.read_csv(path)
    df = df[df["metric"] == metric].copy()
    df = df.sort_values(x_col).reset_index(drop=True)
    return df


def load_class_metric(
    path: Path, x_col: str, metric: str, class_id: int
) -> pd.DataFrame:
    """Load one metric for one class from a class summary CSV."""
    df = pd.read_csv(path)
    df = df[(df["metric"] == metric) & (df["class_id"] == class_id)].copy()
    df = df.sort_values(x_col).reset_index(drop=True)
    return df


# ============================================================
# Plotting
# ============================================================

def create_h4_figure(output_path: Path) -> None:
    """Create the two-panel H4 comparison figure."""

    # --- Load data ---
    balking_util = load_metric(
        BALKING_AGG, "class1_high_balk", "average_utilization"
    )
    noshow_util = load_metric(
        NO_SHOW_AGG, "class1_xi_high", "average_utilization"
    )
    balking_served = load_class_metric(
        BALKING_CLASS, "class1_high_balk", "percent_serviced", class_id=1
    )
    noshow_served = load_class_metric(
        NO_SHOW_CLASS, "class1_xi_high", "percent_serviced", class_id=1
    )

    # --- Create figure ---
    fig, (ax_util, ax_served) = plt.subplots(
        1, 2, figsize=(12, 5), sharey=False
    )

    # Panel A: Overall Utilization
    ax_util.errorbar(
        balking_util["class1_high_balk"],
        balking_util["mean"],
        yerr=balking_util["ci95"],
        capsize=3,
        label="Balking sweep",
        **BALKING_STYLE,
    )
    ax_util.errorbar(
        noshow_util["class1_xi_high"],
        noshow_util["mean"],
        yerr=noshow_util["ci95"],
        capsize=3,
        label="No-show sweep",
        **NO_SHOW_STYLE,
    )
    ax_util.set_title("Overall Utilization")
    ax_util.set_xlabel("High probability")
    ax_util.set_ylabel("Average utilization")
    ax_util.grid(True, alpha=0.3)
    ax_util.legend(frameon=False)

    # Panel B: Class 1 Served Rate
    ax_served.errorbar(
        balking_served["class1_high_balk"],
        balking_served["mean"],
        yerr=balking_served["ci95"],
        capsize=3,
        label="Balking sweep",
        **BALKING_STYLE,
    )
    ax_served.errorbar(
        noshow_served["class1_xi_high"],
        noshow_served["mean"],
        yerr=noshow_served["ci95"],
        capsize=3,
        label="No-show sweep",
        **NO_SHOW_STYLE,
    )
    ax_served.set_title("Class 1 Served Rate")
    ax_served.set_xlabel("High probability")
    ax_served.set_ylabel("Served rate")
    ax_served.grid(True, alpha=0.3)
    ax_served.legend(frameon=False)

    fig.suptitle(
        "H4: Loss Timing — Balking (pre-booking) vs. No-Show (service-day)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    create_h4_figure(FIGURE_DIR / "h4_loss_timing_comparison.png")
    print("Done.")


if __name__ == "__main__":
    main()