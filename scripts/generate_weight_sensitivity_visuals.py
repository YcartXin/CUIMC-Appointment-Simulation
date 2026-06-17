from __future__ import annotations

import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_DIR / "notebooks" / "temp" / "reservation_playground_cache" / "92a74cbf5416"
CLASS_CACHE_PATH = CACHE_DIR / "class_df.pkl"


def resolve_utilization_config() -> tuple[str, str, str, str, Path]:
    requested = os.environ.get("UTILIZATION_DEFINITION", "booked").strip().lower()
    if requested in {"booked", "booked_slot", "booked-slot"}:
        return (
            "booked",
            "booked_slot_utilization",
            "booked-slot utilization",
            "booked slots divided by available slots",
            REPO_DIR / "docs" / "reports" / "reservation_visual_objectives" / "weight_sensitivity",
        )
    if requested in {"old", "served", "served_slot", "served-slot", "slot", "attended"}:
        return (
            "old",
            "slot_utilization",
            "old served-slot utilization",
            "completed visits divided by available slots",
            REPO_DIR / "docs" / "reports" / "reservation_visual_objectives_old_utilization" / "weight_sensitivity",
        )
    raise ValueError("UTILIZATION_DEFINITION must be 'booked' or 'old'.")


(
    UTILIZATION_DEFINITION,
    UTILIZATION_COLUMN,
    UTILIZATION_LABEL,
    UTILIZATION_DEFINITION_TEXT,
    OUTPUT_DIR,
) = resolve_utilization_config()
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
README_PATH = OUTPUT_DIR / "README.md"

STRICT_POLICY = "Strict C1 reservation"
FCFS_POLICY = "Pooled FCFS"
SCENARIO_TYPE = "symmetric_baseline"
DEMAND_LABEL = "lambda1=lambda2=25"
SEEDS = [5101, 5102, 5103, 5104, 5105]
Q_VALUES = [0, 4, 8, 12, 16, 20, 24, 28, 32]
SERVED_RATE_FLOOR = 0.50
WEIGHT_SETS = [
    (1.0, 1.0),
    (1.25, 1.0),
    (1.5, 1.0),
    (2.0, 1.0),
    (3.0, 1.0),
]


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def weight_label(w1: float, w2: float) -> str:
    return f"w1={w1:g}, w2={w2:g}"


def load_baseline_class_data() -> pd.DataFrame:
    if not CLASS_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing cached class data: {CLASS_CACHE_PATH}")

    class_df = pd.read_pickle(CLASS_CACHE_PATH)
    subset = class_df[
        (class_df["scenario_type"] == SCENARIO_TYPE)
        & (class_df["demand_label"] == DEMAND_LABEL)
        & (class_df["seed"].isin(SEEDS))
        & (class_df["Q"].isin(Q_VALUES))
        & (class_df["policy"].isin([FCFS_POLICY, STRICT_POLICY]))
    ].copy()
    if subset.empty:
        raise RuntimeError("No rows found for the requested baseline slice.")

    strict_q = sorted(subset.loc[subset["policy"] == STRICT_POLICY, "Q"].unique().tolist())
    if strict_q != Q_VALUES:
        raise RuntimeError(f"Strict-reservation Q values do not match the requested grid: {strict_q}")
    if subset[(subset["policy"] == FCFS_POLICY) & (subset["Q"] == 0)].empty:
        raise RuntimeError("FCFS Q=0 baseline is missing.")
    return subset


def base_run_rows(class_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy, seed, q), group in class_df.groupby(["policy", "seed", "Q"]):
        class_1 = group[group["class_id"] == 1]
        class_2 = group[group["class_id"] == 2]
        if class_1.empty or class_2.empty:
            continue

        c1 = class_1.iloc[0]
        c2 = class_2.iloc[0]
        offered_1 = float(c1["offered"])
        offered_2 = float(c2["offered"])
        served_rate_1 = float(c1["served_rate"])
        served_rate_2 = float(c2["served_rate"])
        arrivals_1 = float(c1["arrivals"])
        arrivals_2 = float(c2["arrivals"])
        served_1 = float(c1["served"])
        served_2 = float(c2["served"])
        total_arrivals = arrivals_1 + arrivals_2
        overall_served_rate = (served_1 + served_2) / total_arrivals if total_arrivals > 0 else np.nan
        min_served_rate = min(served_rate_1, served_rate_2)
        rows.append(
            {
                "policy": policy,
                "seed": int(seed),
                "Q": int(q),
                "rho_1": float(c1[UTILIZATION_COLUMN]),
                "rho_2": float(c2[UTILIZATION_COLUMN]),
                "tau_1": float(c1["mean_offered_booking_delay"]) if offered_1 > 0 else np.nan,
                "tau_2": float(c2["mean_offered_booking_delay"]) if offered_2 > 0 else np.nan,
                "offered_1": offered_1,
                "offered_2": offered_2,
                "total_offered_delay_1": float(c1["total_offered_booking_delay"]),
                "total_offered_delay_2": float(c2["total_offered_booking_delay"]),
                "served_rate_1": served_rate_1,
                "served_rate_2": served_rate_2,
                "arrivals_1": arrivals_1,
                "arrivals_2": arrivals_2,
                "served_1": served_1,
                "served_2": served_2,
                "overall_served_rate": overall_served_rate,
                "min_served_rate": min_served_rate,
                "served_rate_flag": min_served_rate < SERVED_RATE_FLOOR,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("Could not build base run rows.")
    return result


def score_weight_sets(base_rows: pd.DataFrame) -> pd.DataFrame:
    scored_frames = []
    for w1, w2 in WEIGHT_SETS:
        scored = base_rows.copy()
        scored["w1"] = w1
        scored["w2"] = w2
        scored["weight_set"] = weight_label(w1, w2)
        scored["weighted_utilization"] = w1 * scored["rho_1"] + w2 * scored["rho_2"]
        wait_denominator = w1 * scored["offered_1"] + w2 * scored["offered_2"]
        scored["weighted_offered_wait"] = np.where(
            wait_denominator > 0,
            (w1 * scored["total_offered_delay_1"] + w2 * scored["total_offered_delay_2"]) / wait_denominator,
            np.nan,
        )
        scored_frames.append(scored)
    return pd.concat(scored_frames, ignore_index=True)


def strict_with_fcfs_deltas(scored: pd.DataFrame) -> pd.DataFrame:
    strict = scored[(scored["policy"] == STRICT_POLICY) & (scored["Q"].isin(Q_VALUES))].copy()
    fcfs = scored[(scored["policy"] == FCFS_POLICY) & (scored["Q"] == 0)][
        [
            "seed",
            "w1",
            "w2",
            "weighted_utilization",
            "weighted_offered_wait",
            "served_rate_1",
            "served_rate_2",
            "overall_served_rate",
        ]
    ].rename(
        columns={
            "weighted_utilization": "fcfs_weighted_utilization",
            "weighted_offered_wait": "fcfs_weighted_offered_wait",
            "served_rate_1": "fcfs_served_rate_1",
            "served_rate_2": "fcfs_served_rate_2",
            "overall_served_rate": "fcfs_overall_served_rate",
        }
    )
    merged = strict.merge(fcfs, on=["seed", "w1", "w2"], how="left", validate="many_to_one")
    if merged["fcfs_weighted_utilization"].isna().any() or merged["fcfs_weighted_offered_wait"].isna().any():
        raise RuntimeError("Some strict-reservation rows are missing a matched FCFS baseline.")

    merged["delta_weighted_utilization"] = (
        merged["weighted_utilization"] - merged["fcfs_weighted_utilization"]
    )
    merged["delta_weighted_offered_wait"] = (
        merged["weighted_offered_wait"] - merged["fcfs_weighted_offered_wait"]
    )
    return merged


def summarize(deltas: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "rho_1",
        "rho_2",
        "tau_1",
        "tau_2",
        "weighted_utilization",
        "weighted_offered_wait",
        "fcfs_weighted_utilization",
        "fcfs_weighted_offered_wait",
        "delta_weighted_utilization",
        "delta_weighted_offered_wait",
        "served_rate_1",
        "served_rate_2",
        "overall_served_rate",
        "fcfs_served_rate_1",
        "fcfs_served_rate_2",
        "fcfs_overall_served_rate",
        "min_served_rate",
    ]
    summary = (
        deltas.groupby(["weight_set", "w1", "w2", "Q"], dropna=False)[numeric_cols]
        .mean()
        .reset_index()
        .sort_values(["w1", "w2", "Q"])
    )
    summary["served_rate_flag"] = summary["min_served_rate"] < SERVED_RATE_FLOOR
    return summary


def save_tables(by_seed: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Path]:
    table_paths = {
        "by_seed": TABLE_DIR / "weight_sensitivity_by_seed.csv",
        "summary": TABLE_DIR / "weight_sensitivity_summary.csv",
    }
    by_seed.to_csv(table_paths["by_seed"], index=False)
    summary.to_csv(table_paths["summary"], index=False)
    return table_paths


def load_saved_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    table_paths = {
        "by_seed": TABLE_DIR / "weight_sensitivity_by_seed.csv",
        "summary": TABLE_DIR / "weight_sensitivity_summary.csv",
    }
    missing = [path for path in table_paths.values() if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing cached simulation data and saved weight-sensitivity tables. "
            f"Expected cache at {CLASS_CACHE_PATH} or saved tables: {missing_text}"
        )
    return pd.read_csv(table_paths["by_seed"]), pd.read_csv(table_paths["summary"]), table_paths


def shade_flagged_q(ax: plt.Axes, summary: pd.DataFrame) -> None:
    flagged_q = sorted(summary.loc[summary["served_rate_flag"], "Q"].unique().tolist())
    for q in flagged_q:
        ax.axvspan(q - 1.0, q + 1.0, color="0.92", zorder=0)


def mark_class_served_rate_flags(ax: plt.Axes, plot_df: pd.DataFrame) -> None:
    class_1_flagged = plot_df[plot_df["served_rate_1"] < SERVED_RATE_FLOOR]
    class_2_flagged = plot_df[plot_df["served_rate_2"] < SERVED_RATE_FLOOR]
    if not class_1_flagged.empty:
        ax.scatter(
            class_1_flagged["Q"],
            class_1_flagged["served_rate_1"],
            marker="X",
            color="black",
            edgecolors="white",
            linewidths=0.7,
            s=95,
            zorder=8,
            label="class below 50%",
        )
    if not class_2_flagged.empty:
        ax.scatter(
            class_2_flagged["Q"],
            class_2_flagged["served_rate_2"],
            marker="X",
            color="black",
            edgecolors="white",
            linewidths=0.7,
            s=95,
            zorder=8,
            label="class below 50%" if class_1_flagged.empty else "_nolegend_",
        )


def plot_weighted_utilization(summary: pd.DataFrame) -> Path:
    figure_name = (
        "weighted_utilization_by_weight.png"
        if UTILIZATION_DEFINITION == "booked"
        else "weighted_old_utilization_by_weight.png"
    )
    output_path = FIGURE_DIR / figure_name
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    shade_flagged_q(ax, summary)

    for idx, (label, group) in enumerate(summary.groupby("weight_set", sort=False)):
        color = f"C{idx}"
        group = group.sort_values("Q")
        ax.plot(group["Q"], group["weighted_utilization"], marker="o", color=color, label=label)
        ax.axhline(group["fcfs_weighted_utilization"].iloc[0], color=color, linestyle="--", alpha=0.45, linewidth=1.0)

    ax.set_title(f"Weighted {UTILIZATION_LABEL} for different class weights")
    ax.set_xlabel("reserved Class 1 slots per day, Q")
    ax.set_ylabel("U(Q)")
    ax.grid(axis="y", alpha=0.25)
    flag_patch = mpatches.Patch(color="0.92", label="served-rate flag zone")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [flag_patch], labels + ["served-rate flag zone"], frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_weighted_wait(summary: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "weighted_offered_wait_by_weight.png"
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    shade_flagged_q(ax, summary)

    for idx, (label, group) in enumerate(summary.groupby("weight_set", sort=False)):
        color = f"C{idx}"
        group = group.sort_values("Q")
        ax.plot(group["Q"], group["weighted_offered_wait"], marker="o", color=color, label=label)
        ax.axhline(group["fcfs_weighted_offered_wait"].iloc[0], color=color, linestyle="--", alpha=0.45, linewidth=1.0)

    ax.set_title("Weighted offered waiting time for different class weights")
    ax.set_xlabel("reserved Class 1 slots per day, Q")
    ax.set_ylabel("T(Q), days")
    ax.grid(axis="y", alpha=0.25)
    flag_patch = mpatches.Patch(color="0.92", label="served-rate flag zone")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [flag_patch], labels + ["served-rate flag zone"], frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_utilization_delta(summary: pd.DataFrame) -> Path:
    figure_name = (
        "weighted_utilization_delta_vs_fcfs_by_weight.png"
        if UTILIZATION_DEFINITION == "booked"
        else "weighted_old_utilization_delta_vs_fcfs_by_weight.png"
    )
    output_path = FIGURE_DIR / figure_name
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    shade_flagged_q(ax, summary)

    for idx, (label, group) in enumerate(summary.groupby("weight_set", sort=False)):
        group = group.sort_values("Q")
        ax.plot(group["Q"], group["delta_weighted_utilization"], marker="o", color=f"C{idx}", label=label)

    ax.axhline(0, color="0.20", linewidth=1.0)
    ax.set_title(f"Strict minus FCFS weighted {UTILIZATION_LABEL}")
    ax.set_xlabel("reserved Class 1 slots per day, Q")
    ax.set_ylabel("U(Q) - U(FCFS)")
    ax.grid(axis="y", alpha=0.25)
    flag_patch = mpatches.Patch(color="0.92", label="served-rate flag zone")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [flag_patch], labels + ["served-rate flag zone"], frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_wait_delta(summary: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "weighted_wait_delta_vs_fcfs_by_weight.png"
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    shade_flagged_q(ax, summary)

    for idx, (label, group) in enumerate(summary.groupby("weight_set", sort=False)):
        group = group.sort_values("Q")
        ax.plot(group["Q"], group["delta_weighted_offered_wait"], marker="o", color=f"C{idx}", label=label)

    ax.axhline(0, color="0.20", linewidth=1.0)
    ax.set_title("Strict minus FCFS weighted offered waiting time")
    ax.set_xlabel("reserved Class 1 slots per day, Q")
    ax.set_ylabel("T(Q) - T(FCFS), days")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.01,
        0.03,
        "Below zero means lower offered waiting time than FCFS.",
        transform=ax.transAxes,
        fontsize=8,
        color="0.25",
    )
    flag_patch = mpatches.Patch(color="0.92", label="served-rate flag zone")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [flag_patch], labels + ["served-rate flag zone"], frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_served_rate_drop(summary: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "served_rate_drop_overall.png"
    first_weight = summary[["w1", "w2"]].drop_duplicates().sort_values(["w1", "w2"]).iloc[0]
    plot_df = summary[
        summary["w1"].eq(first_weight["w1"]) & summary["w2"].eq(first_weight["w2"])
    ].sort_values("Q")

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    shade_flagged_q(ax, plot_df)

    ax.plot(plot_df["Q"], plot_df["served_rate_1"], marker="o", color="tab:blue", label="Class 1 served rate")
    ax.plot(plot_df["Q"], plot_df["served_rate_2"], marker="s", color="tab:orange", label="Class 2 served rate")
    ax.plot(
        plot_df["Q"],
        plot_df["overall_served_rate"],
        marker="^",
        color="0.20",
        linewidth=2.2,
        label="Overall served rate",
    )
    ax.axhline(plot_df["fcfs_served_rate_1"].iloc[0], color="tab:blue", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axhline(plot_df["fcfs_served_rate_2"].iloc[0], color="tab:orange", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axhline(plot_df["fcfs_overall_served_rate"].iloc[0], color="0.20", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.axhline(SERVED_RATE_FLOOR, color="0.45", linestyle=":", linewidth=1.1, alpha=0.8, label="50% class floor")
    mark_class_served_rate_flags(ax, plot_df)

    ax.set_title("Served rates by class and overall")
    ax.set_xlabel("reserved Class 1 slots per day, Q")
    ax.set_ylabel("served rate")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(axis="y", alpha=0.25)
    flag_patch = mpatches.Patch(color="0.92", label="at least one class below 50%")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [flag_patch], labels + ["at least one class below 50%"], frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_figures(summary: pd.DataFrame) -> dict[str, Path]:
    return {
        "weighted_utilization": plot_weighted_utilization(summary),
        "weighted_offered_wait": plot_weighted_wait(summary),
        "weighted_utilization_delta": plot_utilization_delta(summary),
        "weighted_wait_delta": plot_wait_delta(summary),
        "served_rate_drop": plot_served_rate_drop(summary),
    }


def write_readme(figures: dict[str, Path], tables: dict[str, Path]) -> None:
    figure_lines = "\n".join(f"- `{path.relative_to(OUTPUT_DIR)}`" for path in figures.values())
    table_lines = "\n".join(f"- `{path.relative_to(OUTPUT_DIR)}`" for path in tables.values())
    weights = ", ".join(weight_label(w1, w2) for w1, w2 in WEIGHT_SETS)
    README_PATH.write_text(
        "\n".join(
            [
                "# Weight Sensitivity For Strict Reservation",
                "",
                "This folder scores the same baseline simulation runs under several class-weight choices.",
                "No new booking policy is introduced here; the weights only change the objective values.",
                f"Here, utilization means {UTILIZATION_DEFINITION_TEXT}.",
                "",
                f"Baseline slice: `{SCENARIO_TYPE}`, `{DEMAND_LABEL}`, seeds `{SEEDS[0]}-{SEEDS[-1]}`, Q grid `{Q_VALUES}`.",
                f"Weight sets: {weights}.",
                "",
                "Gray vertical bands mark Q values where at least one class has served rate below 50%.",
                "On the served-rate plot, black X markers are placed on the class line that is below 50%.",
                "Dashed horizontal lines in the absolute plots are the pooled FCFS references for the same weight set.",
                "The served-rate plot is not repeated by weight because weights do not change which patients are served.",
                "",
                "## Figures",
                figure_lines,
                "",
                "## Tables",
                table_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def print_summary(figures: dict[str, Path], tables: dict[str, Path], summary: pd.DataFrame) -> None:
    print("Weight-sensitivity simulation summary")
    print(
        "Baseline slice: "
        f"{SCENARIO_TYPE}, {DEMAND_LABEL}, seeds {SEEDS[0]}-{SEEDS[-1]}, Q={Q_VALUES}"
    )
    print(f"Utilization definition: {UTILIZATION_COLUMN} ({UTILIZATION_DEFINITION_TEXT})")
    print("Weight sets:")
    for w1, w2 in WEIGHT_SETS:
        print(f"  - {weight_label(w1, w2)}")
    print("Figures created:")
    for path in figures.values():
        print(f"  - {path}")
    print("Tables created:")
    for path in tables.values():
        print(f"  - {path}")
    print(f"README: {README_PATH}")
    print(f"Served-rate flag Q values: {sorted(summary.loc[summary['served_rate_flag'], 'Q'].unique().tolist())}")
    print("Note: weights rescore the same simulation runs; they do not change appointment booking behavior.")


def main() -> None:
    ensure_dirs()
    if CLASS_CACHE_PATH.exists():
        class_df = load_baseline_class_data()
        base_rows = base_run_rows(class_df)
        scored = score_weight_sets(base_rows)
        by_seed = strict_with_fcfs_deltas(scored)
        summary = summarize(by_seed)
        tables = save_tables(by_seed, summary)
    else:
        by_seed, summary, tables = load_saved_tables()
    figures = make_figures(summary)
    write_readme(figures, tables)
    print_summary(figures, tables, summary)


if __name__ == "__main__":
    main()
