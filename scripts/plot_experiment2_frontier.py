"""Generate six Experiment 2 frontier figures from exported background tables.

Run from any directory: python /path/to/repo/scripts/plot_experiment2_frontier.py
No simulation, selection, or CSV mutation is performed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {"reservation_only": "Reservation only", "both_flexible": "Both flexible"}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]


def load_data(input_dir: Path):
    targets = pd.read_csv(input_dir / "background_policy_targets.csv")
    cells = pd.read_csv(input_dir / "minimum_qualifying_cells.csv")
    required = {"background_id", "policy", "rho", "class1_share", "min_budget_5pp", "achieves_5pp"}
    if not required.issubset(targets.columns):
        raise ValueError(f"Missing target columns: {sorted(required - set(targets.columns))}")
    required_cells = {"background_id", "policy", "target", "min_tested_c2_budget_pp", "actual_c2_loss_pp", "c1_gain_pp"}
    if not required_cells.issubset(cells.columns):
        raise ValueError(f"Missing qualifying-cell columns: {sorted(required_cells - set(cells.columns))}")
    if targets.duplicated(["background_id", "policy"]).any():
        raise ValueError("Duplicate background/policy targets")
    if set(targets.policy) != set(POLICIES):
        raise ValueError("Expected reservation_only and both_flexible policies")
    flags = targets.achieves_5pp.astype(str).str.lower().map({"true": True, "false": False})
    if flags.isna().any():
        raise ValueError("achieves_5pp must contain only True/False")
    targets["success"] = flags
    budget = pd.to_numeric(targets.min_budget_5pp, errors="raise")
    if (flags != budget.notna()).any() or (budget.dropna() < 0).any() or (budget.dropna() > .10 + 1e-9).any():
        raise ValueError("Success flags and minimum budgets disagree or exceed the 10pp ceiling")
    targets["budget_pp"] = budget * 100
    for col in ["rho", "class1_share"]:
        if not np.isfinite(targets[col]).all():
            raise ValueError(f"Nonfinite {col}")
    wide = targets.pivot(index="background_id", columns="policy", values="success")
    if wide.isna().any().any():
        raise ValueError("Every background must have both policies")
    metadata = targets.pivot(index="background_id", columns="policy", values=["rho", "class1_share"])
    for col in ["rho", "class1_share"]:
        if not metadata[col].iloc[:, 0].equals(metadata[col].iloc[:, 1]):
            raise ValueError(f"Policy rows disagree on {col}")
    qualifying = cells.loc[cells.target == "5pp"].copy()
    matched = targets.loc[targets.success].merge(qualifying, on=["background_id", "policy"], how="outer", validate="one_to_one", indicator=True)
    if not matched._merge.eq("both").all():
        raise ValueError("Successful targets and qualifying cells do not match")
    for col in ["min_tested_c2_budget_pp", "actual_c2_loss_pp", "c1_gain_pp"]:
        if not np.isfinite(matched[col]).all():
            raise ValueError(f"Nonfinite qualifying-cell {col}")
    if not np.allclose(matched.budget_pp, matched.min_tested_c2_budget_pp):
        raise ValueError("Minimum budgets disagree between exports")
    if (matched.c1_gain_pp < 5 - 1e-8).any() or (matched.actual_c2_loss_pp > matched.budget_pp + 1e-8).any():
        raise ValueError("Qualifying cells violate gain target or loss budget")
    return targets, qualifying, wide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs" / "experiment2_frontier")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "experiment2_frontier")
    parser.add_argument("--figure", choices=["all", "budget"], default="all", help="Generate all figures or only the median budget figure")
    args = parser.parse_args()
    targets, cells, wide = load_data(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 15,
                         "axes.labelsize": 12, "svg.hashsalt": "experiment2_frontier"})
    bf = targets.loc[targets.policy == "both_flexible"]
    shares = sorted(bf.class1_share.unique())
    rhos = sorted(bf.rho.unique())
    grouped = bf.groupby(["class1_share", "rho"])
    rates = grouped.success.mean().unstack("rho").reindex(index=shares, columns=rhos) * 100
    budget_stats = bf.loc[bf.success].groupby(["class1_share", "rho"]).budget_pp.agg(["count", "min", "median", "max"])
    budget_stats = budget_stats.reindex(pd.MultiIndex.from_product([shares, rhos], names=["class1_share", "rho"]))
    budget_stats["count"] = budget_stats["count"].fillna(0).astype(int)
    note = "Target: Class 1 gain >=5pp; Class 2 loss ceiling: 10pp. Each background has equal weight."

    def save(fig, stem, caption=note):
        if caption:
            fig.text(.5, .015, caption, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .055 if caption else 0, 1, 1))
        for ext in ("png", "svg"):
            path = args.output_dir / f"{stem}.{ext}"
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Generated {stem}.png / .svg")

    def lines(table, title, ylabel, stem, percent=False):
        fig, ax = plt.subplots(figsize=(10.5, 6))
        for i, rho in enumerate(rhos):
            ax.plot(shares, table[rho], marker=["o", "s", "^", "D", "v", "P"][i % 6], color=COLORS[i % len(COLORS)], label=fr"$\rho={rho:g}$")
        ax.set(title=title, xlabel="Class 1 patient share", ylabel=ylabel, xticks=shares)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.25)
        ax.legend(title="Demand / capacity")
        if percent:
            ax.set_ylim(-3, 103)
            ax.yaxis.set_major_formatter(PercentFormatter(100))
        else:
            ax.set_ylim(0, 10.5)
        save(fig, stem, note if percent else note + "\nMedians among successful backgrounds only; gaps mean no successes.")

    def budget_ranges():
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        # Stagger series within each share group so overlapping range bars remain
        # visible. These offsets are display positions, not different share values.
        offsets = np.linspace(-.012, .012, len(rhos))
        for i, rho in enumerate(rhos):
            stats = budget_stats.xs(rho, level="rho").reindex(shares)
            valid = stats["count"] > 0
            med = stats["median"].to_numpy(dtype=float)
            lo = stats["min"].to_numpy(dtype=float)
            hi = stats["max"].to_numpy(dtype=float)
            assert np.all(lo[valid] <= med[valid]) and np.all(med[valid] <= hi[valid])
            color = COLORS[i % len(COLORS)]
            positions = np.asarray(shares) + offsets[i]
            ax.plot(positions, med, color=color, linewidth=1.5, alpha=.8)
            artists = ax.errorbar(positions[valid], med[valid],
                        yerr=np.vstack([med[valid] - lo[valid], hi[valid] - med[valid]]),
                        fmt=["o", "s", "^", "D", "v", "P"][i % 6],
                        color=color, ecolor=matplotlib.colors.to_rgba(color, .28), markersize=7, capsize=3,
                        elinewidth=.8, capthick=.8, label=fr"$\rho={rho:g}$")
            for collection in artists[2]:
                collection.set_linestyle("--")
                collection.set_zorder(1)
        ax.set(title="Minimum tested Class 2 budget: median and min–max range",
               xlabel="Class 1 patient share", ylabel="Minimum tested Class 2 budget (pp)",
               xticks=shares, ylim=(-.3, 10.6), xlim=(min(shares)-.04, max(shares)+.04))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.25)
        ax.legend(title="Demand / capacity", ncol=2)
        save(fig, "fig2_bf_min_budget_by_share_and_rho", caption=None)

    budget_ranges()
    if args.figure == "budget":
        return
    lines(rates, "Both-flexible success by patient mix and demand", "Successful backgrounds (%)", "fig1_bf_success_rate_by_share_and_rho", True)

    def heatmap(table, title, label, stem, percent=False):
        fig, ax = plt.subplots(figsize=(10.5, 6))
        values = table.to_numpy(dtype=float).T
        im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap="Blues", vmin=0, vmax=100 if percent else max(1, np.nanmax(values)))
        ax.set(xticks=range(len(shares)), xticklabels=[f"{s:.0%}" for s in shares], yticks=range(len(rhos)), yticklabels=[f"{r:g}" for r in rhos], xlabel="Class 1 patient share", ylabel=r"Demand / capacity ($\rho$)", title=title)
        for row in range(len(rhos)):
            for col in range(len(shares)):
                v = values[row, col]
                text = "N/A" if not np.isfinite(v) else (f"{v:.0f}%" if percent else f"{v:.0f}")
                ax.text(col, row, text, ha="center", va="center", color="white" if np.isfinite(v) and v > im.norm.vmax * .6 else "black")
        fig.colorbar(im, ax=ax, label=label)
        save(fig, stem)

    heatmap(100 - rates, "Both-flexible failure under the 10pp ceiling", "Failed backgrounds (%)", "fig3_bf_failure_heatmap_under_10pp_ceiling", True)
    outcomes = wide.join(bf.set_index("background_id")[["class1_share", "rho"]])
    conditions = [wide.both_flexible & wide.reservation_only, wide.both_flexible & ~wide.reservation_only, ~wide.both_flexible & wide.reservation_only]
    outcomes["outcome"] = np.select(conditions, ["Both policies work", "Both flexible only", "Reservation only works"], default="Neither works")
    counts = pd.crosstab(outcomes.class1_share, outcomes.outcome).reindex(index=shares, fill_value=0)
    order = ["Both policies work", "Both flexible only", "Reservation only works", "Neither works"]
    counts = counts.reindex(columns=order, fill_value=0)
    fig, ax = plt.subplots(figsize=(10.5, 6))
    counts.plot.bar(stacked=True, ax=ax, color=[COLORS[0], COLORS[1], COLORS[3], "#bdbdbd"], rot=0)
    ax.set(title="Policy success across background environments", xlabel="Class 1 patient share", ylabel="Number of backgrounds", xticklabels=[f"{s:.0%}" for s in shares])
    ax.legend(fontsize=9)
    save(fig, "fig4_background_outcomes_by_class1_share")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, col, title in zip(axes, ["min_tested_c2_budget_pp", "actual_c2_loss_pp"], ["Minimum tested budget", "Actual evaluated loss at that budget"]):
        for i, (policy, label) in enumerate(POLICIES.items()):
            values = np.sort(cells.loc[cells.policy == policy, col].to_numpy())
            if len(values):
                ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=COLORS[i], label=f"{label} (n={len(values)})")
        ax.set(title=title, xlabel="Class 2 loss / budget (pp)", ylabel="Cumulative share of successful backgrounds", ylim=(0, 1.02))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.25)
        ax.legend(fontsize=9, loc="lower right")
    save(fig, "fig5_required_budget_distribution_by_policy", note + "\nConditional distributions: each policy's successful backgrounds; samples may differ. Negative actual loss means a Class 2 gain.")
    only = outcomes.loc[outcomes.outcome == "Both flexible only"]
    breakdown = pd.crosstab(only.class1_share, only.rho).reindex(index=shares, columns=rhos, fill_value=0)
    heatmap(breakdown, f"Both-flexible-only success: {len(only)} backgrounds", "Number of backgrounds", "fig6_bf_only_backgrounds_breakdown")
    print(f"Validated {len(wide)} backgrounds; outcome counts: {outcomes.outcome.value_counts().to_dict()}")


if __name__ == "__main__":
    main()

