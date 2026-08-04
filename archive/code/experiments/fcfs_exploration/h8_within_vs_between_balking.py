"""
H8: Within-class balking step vs. between-class post-threshold gap.

The experiment fixes Class 1's post-threshold balking rate at 0.50 and varies:

    S1 = b1_1 - b0_1        (Class 1 within-class step)
    G1 = b1_1 - b1_2        (between-class post-threshold gap)

S1 changes by lowering Class 1 b0. G1 changes by lowering Class 2 b1.
Both vary from 0.00 to 0.50 in increments of 0.10. Each of the 36 cells is
run with the same 100 seeds.

For every common starting cell and seed, the script compares equal 0.10
increases:

    Delta_step = R1(S1 + 0.10, G1) - R1(S1, G1)
    Delta_gap  = R1(S1, G1 + 0.10) - R1(S1, G1)

Since the two effects may have opposite signs, the primary H8 comparison is:

    abs(Delta_gap) - abs(Delta_step)

Run from the repository root:

    python experiments/h8_within_vs_between_balking.py
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

from analysis.metrics import class_result_rows
from simulation.config_loader import load_config
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import ThresholdRule


CONFIG_PATH = REPO_DIR / "configs" / "baseline.yaml"
OUTPUT_DIR = REPO_DIR / "outputs" / "h8_within_vs_between_balking"
RAW_DIR = OUTPUT_DIR / "raw"
SUMMARY_DIR = OUTPUT_DIR / "summary"
FIGURE_DIR = OUTPUT_DIR / "figures"

SEEDS = range(1, 101)
LEVEL_INDICES = range(6)  # 0.00, 0.10, ..., 0.50
INCREMENT = 0.10

CLASS_1_POST = 0.50
CLASS_2_PRE = 0.00


def rate(index: int) -> float:
    return round(index * INCREMENT, 2)


def make_config(base_config, step_index: int, gap_index: int, seed: int):
    """Create one factorial-cell configuration."""
    within_step = rate(step_index)
    post_gap = rate(gap_index)

    class_1_low = round(CLASS_1_POST - within_step, 2)
    class_2_high = round(CLASS_1_POST - post_gap, 2)

    c1 = base_config.classes[1]
    c2 = base_config.classes[2]
    classes = dict(base_config.classes)

    classes[1] = replace(
        c1,
        balk_prob=ThresholdRule(
            threshold=c1.balk_prob.threshold,
            low=class_1_low,
            high=CLASS_1_POST,
        ),
    )
    classes[2] = replace(
        c2,
        balk_prob=ThresholdRule(
            threshold=c2.balk_prob.threshold,
            low=CLASS_2_PRE,
            high=class_2_high,
        ),
    )

    return replace(base_config, classes=classes, seed=int(seed))


def run_sweep() -> pd.DataFrame:
    """Run the 6 x 6 factorial grid with common seeds."""
    base_config = load_config(CONFIG_PATH)
    rows = []
    cell = 0

    for step_index in LEVEL_INDICES:
        for gap_index in LEVEL_INDICES:
            cell += 1
            within_step = rate(step_index)
            post_gap = rate(gap_index)
            class_1_low = round(CLASS_1_POST - within_step, 2)
            class_2_high = round(CLASS_1_POST - post_gap, 2)

            print(
                f"Cell {cell:02d}/36: S1={within_step:.2f}, G1={post_gap:.2f}; "
                f"C1=({class_1_low:.2f}, {CLASS_1_POST:.2f}), "
                f"C2=({CLASS_2_PRE:.2f}, {class_2_high:.2f})"
            )

            for seed in SEEDS:
                config = make_config(base_config, step_index, gap_index, seed)
                results = ClinicAppointmentSimulation(config).run()
                rows.extend(
                    class_result_rows(
                        results,
                        {
                            "seed": seed,
                            "step_index": step_index,
                            "gap_index": gap_index,
                            "within_step": within_step,
                            "post_gap": post_gap,
                            "class_1_low": class_1_low,
                            "class_1_high": CLASS_1_POST,
                            "class_2_low": CLASS_2_PRE,
                            "class_2_high": class_2_high,
                        },
                    )
                )

    return pd.DataFrame(rows)


def add_ci(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["se"] = summary["std"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["se"]
    summary["ci_low"] = summary["mean"] - summary["ci95"]
    summary["ci_high"] = summary["mean"] + summary["ci95"]
    return summary


def summarize_grid(raw: pd.DataFrame) -> pd.DataFrame:
    c1 = raw.loc[raw["class_id"] == 1].copy()
    summary = (
        c1.groupby(
            ["step_index", "gap_index", "within_step", "post_gap"],
            as_index=False,
        )
        .agg(
            mean=("percent_serviced", "mean"),
            std=("percent_serviced", "std"),
            n=("percent_serviced", "count"),
        )
    )
    return add_ci(summary)


def compute_paired_effects(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute both 0.10 effects from the same starting cell and seed."""
    c1 = raw.loc[
        raw["class_id"] == 1,
        ["seed", "step_index", "gap_index", "within_step", "post_gap", "percent_serviced"],
    ].rename(columns={"percent_serviced": "served_base"})

    # Both neighboring cells exist only when the starting indices are 0 to 4.
    base = c1.loc[(c1["step_index"] < 5) & (c1["gap_index"] < 5)].copy()

    step_up = c1[["seed", "step_index", "gap_index", "served_base"]].copy()
    step_up["step_index"] -= 1
    step_up = step_up.rename(columns={"served_base": "served_step_up"})

    gap_up = c1[["seed", "step_index", "gap_index", "served_base"]].copy()
    gap_up["gap_index"] -= 1
    gap_up = gap_up.rename(columns={"served_base": "served_gap_up"})

    keys = ["seed", "step_index", "gap_index"]
    paired = base.merge(step_up, on=keys, validate="one_to_one")
    paired = paired.merge(gap_up, on=keys, validate="one_to_one")

    paired["delta_step"] = paired["served_step_up"] - paired["served_base"]
    paired["delta_gap"] = paired["served_gap_up"] - paired["served_base"]
    paired["abs_delta_step"] = paired["delta_step"].abs()
    paired["abs_delta_gap"] = paired["delta_gap"].abs()
    paired["effect_difference"] = paired["abs_delta_gap"] - paired["abs_delta_step"]
    paired["gap_effect_larger"] = paired["effect_difference"] > 0

    return paired.sort_values(["seed", "step_index", "gap_index"])


def summarize_by_seed(paired: pd.DataFrame) -> pd.DataFrame:
    """Average grid cells within each seed before inference."""
    return (
        paired.groupby("seed", as_index=False)
        .agg(
            signed_step_effect=("delta_step", "mean"),
            signed_gap_effect=("delta_gap", "mean"),
            absolute_step_effect=("abs_delta_step", "mean"),
            absolute_gap_effect=("abs_delta_gap", "mean"),
            absolute_effect_difference=("effect_difference", "mean"),
            share_gap_larger=("gap_effect_larger", "mean"),
        )
    )


def summarize_effects(seed_summary: pd.DataFrame) -> pd.DataFrame:
    measures = {
        "Within-class step: signed effect": "signed_step_effect",
        "Between-class post-threshold gap: signed effect": "signed_gap_effect",
        "Within-class step: absolute effect": "absolute_step_effect",
        "Between-class post-threshold gap: absolute effect": "absolute_gap_effect",
        "Between minus within: absolute-effect difference": "absolute_effect_difference",
        "Share of comparisons where between-class effect is larger": "share_gap_larger",
    }

    rows = []
    for label, column in measures.items():
        values = seed_summary[column]
        mean = values.mean()
        std = values.std(ddof=1)
        n = values.count()
        se = std / np.sqrt(n)
        ci95 = 1.96 * se
        rows.append(
            {
                "measure": label,
                "mean": mean,
                "std": std,
                "n": n,
                "se": se,
                "ci95": ci95,
                "ci_low": mean - ci95,
                "ci_high": mean + ci95,
            }
        )

    return pd.DataFrame(rows)


def summarize_cells(paired: pd.DataFrame) -> pd.DataFrame:
    summary = (
        paired.groupby(
            ["step_index", "gap_index", "within_step", "post_gap"],
            as_index=False,
        )
        .agg(
            mean_delta_step=("delta_step", "mean"),
            mean_delta_gap=("delta_gap", "mean"),
            mean_abs_step=("abs_delta_step", "mean"),
            mean_abs_gap=("abs_delta_gap", "mean"),
            mean_difference=("effect_difference", "mean"),
            share_seeds_gap_larger=("gap_effect_larger", "mean"),
        )
    )
    return summary


def effect_row(effect_summary: pd.DataFrame, label: str) -> pd.Series:
    return effect_summary.loc[effect_summary["measure"] == label].iloc[0]


def plot_effect_comparison(effect_summary: pd.DataFrame, output_path: Path) -> None:
    labels = [
        "Within-class step\n$S_1=b_{1,1}-b_{0,1}$",
        "Between-class post-threshold gap\n$G_1=b_{1,1}-b_{1,2}$",
    ]
    rows = effect_summary.set_index("measure").loc[
        [
            "Within-class step: absolute effect",
            "Between-class post-threshold gap: absolute effect",
        ]
    ]

    means = rows["mean"].to_numpy() * 100
    cis = rows["ci95"].to_numpy() * 100
    x = np.arange(2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, means, yerr=cis, fmt="o", markersize=8, capsize=5, linewidth=2)

    for xi, mean, ci in zip(x, means, cis):
        ax.annotate(
            f"{mean:.3f} pp\n95% CI [{mean-ci:.3f}, {mean+ci:.3f}]",
            (xi, mean),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Absolute change in Class 1 served rate (percentage points)")
    ax.set_title("H8: Marginal Effect of an Equal 0.10 Increase")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_difference_heatmap(cell_summary: pd.DataFrame, output_path: Path) -> None:
    matrix = (
        cell_summary.pivot(
            index="post_gap",
            columns="within_step",
            values="mean_difference",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    values = matrix.to_numpy() * 100
    limit = np.nanmax(np.abs(values)) or 1.0

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_xticks(range(len(matrix.columns)), [f"{x:.1f}" for x in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [f"{y:.1f}" for y in matrix.index])
    ax.set_xlabel("Starting within-class step $S_1$")
    ax.set_ylabel("Starting between-class post-threshold gap $G_1$")
    ax.set_title(
        "Between-Class Minus Within-Class Absolute Effect\n"
        "Positive values favor the between-class post-threshold gap"
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Difference in absolute effects (percentage points)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_qmd_summary(
    effect_summary: pd.DataFrame,
    cell_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    step_signed = effect_row(effect_summary, "Within-class step: signed effect")
    gap_signed = effect_row(effect_summary, "Between-class post-threshold gap: signed effect")
    step_abs = effect_row(effect_summary, "Within-class step: absolute effect")
    gap_abs = effect_row(effect_summary, "Between-class post-threshold gap: absolute effect")
    difference = effect_row(
        effect_summary,
        "Between minus within: absolute-effect difference",
    )

    cells_gap_larger = int((cell_summary["mean_difference"] > 0).sum())
    total_cells = len(cell_summary)

    if difference["ci_low"] > 0:
        verdict = "Supported"
        conclusion = "The between-class post-threshold gap has the larger average absolute effect."
    elif difference["ci_high"] < 0:
        verdict = "Not supported"
        conclusion = "The within-class step has the larger average absolute effect."
    else:
        verdict = "Qualified / inconclusive"
        conclusion = "The paired confidence interval includes zero, so neither contrast clearly dominates."

    text = f"""# H8 results for the evidence section

## Equal 0.10 marginal changes

- Within-class step, signed effect: {step_signed['mean']*100:.4f} percentage points
  (95% CI {step_signed['ci_low']*100:.4f} to {step_signed['ci_high']*100:.4f}).
- Between-class post-threshold gap, signed effect: {gap_signed['mean']*100:.4f} percentage points
  (95% CI {gap_signed['ci_low']*100:.4f} to {gap_signed['ci_high']*100:.4f}).
- Within-class step, absolute effect: {step_abs['mean']*100:.4f} percentage points
  (95% CI {step_abs['ci_low']*100:.4f} to {step_abs['ci_high']*100:.4f}).
- Between-class post-threshold gap, absolute effect: {gap_abs['mean']*100:.4f} percentage points
  (95% CI {gap_abs['ci_low']*100:.4f} to {gap_abs['ci_high']*100:.4f}).
- Paired difference, between minus within: {difference['mean']*100:.4f} percentage points
  (95% CI {difference['ci_low']*100:.4f} to {difference['ci_high']*100:.4f}).

The between-class post-threshold gap had the larger mean absolute effect in
{cells_gap_larger} of {total_cells} common starting cells
({cells_gap_larger/total_cells:.1%}).

**Automated verdict: {verdict}.** {conclusion}

The confidence interval is calculated across 100 seed-level averages. The 25
starting grid cells are averaged within each seed before inference, so the grid
cells are not treated as independent replications.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    for directory in (RAW_DIR, SUMMARY_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    raw = run_sweep()
    raw.to_csv(RAW_DIR / "raw_results.csv", index=False)

    grid_summary = summarize_grid(raw)
    grid_summary.to_csv(SUMMARY_DIR / "class1_grid_summary.csv", index=False)

    paired = compute_paired_effects(raw)
    paired.to_csv(SUMMARY_DIR / "paired_marginal_effects.csv", index=False)

    seed_summary = summarize_by_seed(paired)
    seed_summary.to_csv(SUMMARY_DIR / "seed_level_effect_summary.csv", index=False)

    effect_summary = summarize_effects(seed_summary)
    effect_summary.to_csv(SUMMARY_DIR / "marginal_effect_summary.csv", index=False)

    cell_summary = summarize_cells(paired)
    cell_summary.to_csv(SUMMARY_DIR / "dominance_by_grid_cell.csv", index=False)

    plot_effect_comparison(
        effect_summary,
        FIGURE_DIR / "h8_marginal_effect_comparison.png",
    )
    plot_difference_heatmap(
        cell_summary,
        FIGURE_DIR / "h8_effect_difference_heatmap.png",
    )
    write_qmd_summary(
        effect_summary,
        cell_summary,
        SUMMARY_DIR / "h8_results_for_qmd.md",
    )

    print("\nH8 experiment complete.")
    print(f"Read: {SUMMARY_DIR / 'h8_results_for_qmd.md'}")


if __name__ == "__main__":
    main()
