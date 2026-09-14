"""Create explanatory figures used in the FCFS presentation.

Outputs
-------
figures/slide_05_threshold_magnitude_definition.png
figures/slide_05_threshold_magnitude_definition.svg
figures/slide_09_class_access_gap_revised_yaxis.png
figures/slide_09_class_access_gap_revised_yaxis.svg

The Slide 9 values reproduce the means and interval endpoints displayed in
the existing presentation figure. The revised figure changes the y-axis copy
and retains the per-10-percentage-point interpretation as a figure note.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GRID = "#d9d9d9"
TEXT = "#111111"


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 25,
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "axes.edgecolor": TEXT,
            "axes.linewidth": 1.3,
        }
    )


def _save(fig: plt.Figure, stem: str, *, dpi: int = 200) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_threshold_magnitude_definition() -> None:
    """Illustrate the two parameters of the presentation's threshold rule."""

    threshold = 6
    pre_probability = 0.05
    post_probability = 0.30

    fig, ax = plt.subplots(figsize=(10.24, 5.76))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.16)

    # The step changes at the threshold. The post-threshold level itself is
    # the magnitude; it is deliberately not drawn as the size of the jump.
    x = np.array([0, threshold, threshold, 14], dtype=float)
    y = np.array([pre_probability, pre_probability, post_probability, post_probability])
    ax.plot(x, y, color=BLUE, linewidth=4, solid_capstyle="round", zorder=3)
    ax.scatter([threshold], [post_probability], color=BLUE, s=90, zorder=4)

    ax.axvline(threshold, color="#6b6b6b", linestyle="--", linewidth=2, zorder=1)
    ax.axvspan(threshold, 14, color=ORANGE, alpha=0.10, zorder=0)

    ax.annotate(
        "Delay = threshold\n6 days",
        xy=(threshold, 0.17),
        xytext=(4.55, 0.17),
        ha="right",
        va="center",
        fontsize=16,
        color=TEXT,
        arrowprops={"arrowstyle": "-|>", "color": "#6b6b6b", "lw": 1.8},
    )
    ax.annotate(
        "Magnitude = post-threshold probability (30%)",
        xy=(10.4, post_probability),
        xytext=(8.2, 0.382),
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=ORANGE,
        arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 2},
    )
    ax.text(
        2.9,
        pre_probability + 0.018,
        "Pre-threshold probability (5%)",
        ha="center",
        va="bottom",
        fontsize=15,
        color="#4c4c4c",
    )
    ax.text(
        10.1,
        0.025,
        "Appointments offered after the threshold\nuse the post-threshold probability",
        ha="center",
        va="bottom",
        fontsize=14,
        color="#4c4c4c",
    )
    ax.text(
        0.25,
        0.405,
        "Illustrative values",
        ha="left",
        va="top",
        fontsize=13,
        color="#666666",
    )

    ax.set_title("Magnitude and delay in the threshold model", pad=17, fontsize=24, fontweight="semibold")
    ax.set_xlabel("Offered booking delay (days)", labelpad=10)
    ax.set_ylabel("Probability of balking or no-show", labelpad=12, fontsize=17)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 0.42)
    ax.set_xticks(np.arange(0, 15, 2))
    ax.set_yticks(np.arange(0, 0.41, 0.10))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=1, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)

    _save(fig, "slide_05_threshold_magnitude_definition")


def create_slide_09_revised_yaxis() -> None:
    """Recreate the existing Slide 9 figure with revised y-axis wording."""

    demand = np.arange(3)
    demand_labels = ["Low demand", "Moderate demand", "High demand"]

    balk_mean = np.array([-0.52, -0.76, -1.00])
    balk_low = np.array([-0.88, -0.96, -1.21])
    balk_high = np.array([-0.15, -0.55, -0.79])

    noshow_mean = np.array([-2.32, -1.83, -1.34])
    noshow_low = np.array([-2.67, -2.07, -1.55])
    noshow_high = np.array([-1.96, -1.62, -1.13])

    fig, ax = plt.subplots(figsize=(10.24, 6.025))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.91, bottom=0.16)
    fig.text(
        0.16,
        0.965,
        "Effect of a 10 pp increase in the C1-C2 behavior gap; shaded bands show 95% CIs",
        ha="left",
        va="top",
        fontsize=12,
        color="#555555",
    )

    ax.axhline(0, color=BLUE, linewidth=1.5)
    ax.fill_between(demand, balk_low, balk_high, color=BLUE, alpha=0.16, linewidth=0)
    ax.plot(
        demand,
        balk_mean,
        color=BLUE,
        marker="o",
        linewidth=3,
        markersize=9,
        label="Balking probability gap",
    )
    ax.fill_between(demand, noshow_low, noshow_high, color=ORANGE, alpha=0.16, linewidth=0)
    ax.plot(
        demand,
        noshow_mean,
        color=ORANGE,
        marker="o",
        linewidth=3,
        markersize=9,
        label="No-show probability gap",
    )

    ax.set_xticks(demand, demand_labels)
    ax.set_xlabel("Demand level", labelpad=9)
    ax.set_ylabel("Difference in served rate\nbetween C1 and C2 (pp)", labelpad=12)
    ax.set_ylim(-2.8, 0.13)
    ax.set_yticks(np.arange(-2.5, 0.1, 0.5))
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    _save(fig, "slide_09_class_access_gap_revised_yaxis")


def main() -> None:
    _set_style()
    create_threshold_magnitude_definition()
    create_slide_09_revised_yaxis()


if __name__ == "__main__":
    main()
