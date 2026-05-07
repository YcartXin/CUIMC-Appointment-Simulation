from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from engine_files.config_loader import load_config
from engine_files.engine import ClinicAppointmentSimulation
from engine_files.model import ThresholdRule


OUT_DIR = Path(__file__).resolve().parent / "metric_analysis_files"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CONFIG = load_config(REPO_DIR / "configs" / "baseline.yaml")
SCENARIO_2_CONFIG = load_config(REPO_DIR / "configs" / "scenario_2.yaml")

FINE_SEED = 2027
SCENARIO_SEEDS = [2027, 2028, 2029]

STEP_GRID = np.linspace(0.0, 1.0, 21)
PROB_GRID = np.linspace(0.0, 0.30, 21)
THRESHOLD_GRID = list(range(BASE_CONFIG.horizon_days))
ARRIVAL_MULTIPLIERS = np.linspace(0.5, 1.5, 11)
CLASS_1_SHARES = np.linspace(0.1, 0.9, 17)
CLASS_ARRIVAL_MULTIPLIERS = np.linspace(0.5, 1.5, 21)


def run_result(config, seed):
    seeded = replace(config, seed=seed)
    return ClinicAppointmentSimulation(seeded).run()


def aggregate_delay_metrics(result):
    booked = sum(m.booked for m in result.class_metrics.values())
    offered = sum(m.offered for m in result.class_metrics.values())
    accepted_delay = sum(m.total_booking_delay for m in result.class_metrics.values())
    offered_delay = sum(m.total_offered_booking_delay for m in result.class_metrics.values())
    return {
        "mean_accepted_booking_delay": accepted_delay / booked if booked else 0.0,
        "mean_offered_booking_delay": offered_delay / offered if offered else 0.0,
    }


def result_metrics(config, seed=FINE_SEED):
    result = run_result(config, seed)
    c1 = result.class_metrics[1]
    c2 = result.class_metrics[2]

    class_1_delay = c1.mean_offered_booking_delay
    class_2_delay = c2.mean_offered_booking_delay

    return {
        "average_utilization": result.average_utilization,
        "overall_percent_serviced": result.overall_percent_serviced,
        **aggregate_delay_metrics(result),
        "class_1_percent_serviced": c1.percent_serviced,
        "class_2_percent_serviced": c2.percent_serviced,
        "class_1_mean_offered_booking_delay": class_1_delay,
        "class_2_mean_offered_booking_delay": class_2_delay,
        "access_advantage_class_1": c1.percent_serviced - c2.percent_serviced,
        "delay_advantage_class_1": class_2_delay - class_1_delay,
    }


def mean_metrics(config, seeds):
    rows = [result_metrics(config, seed) for seed in seeds]
    return pd.DataFrame(rows).mean(numeric_only=True).to_dict()


def make_step_rule(old_rule, threshold=None, step=None):
    threshold = old_rule.threshold if threshold is None else int(threshold)
    low = old_rule.low
    step = old_rule.high - old_rule.low if step is None else float(step)
    return ThresholdRule(threshold=threshold, low=low, high=min(low + step, 1.0))


def update_classes(config, changes):
    classes = {}
    for class_id, params in config.classes.items():
        classes[class_id] = replace(params, **changes.get(class_id, {}))
    return replace(config, classes=classes)


def set_balk_steps(config, class_1_step, class_2_step):
    return update_classes(
        config,
        {
            1: {"balk_prob": make_step_rule(config.classes[1].balk_prob, step=class_1_step)},
            2: {"balk_prob": make_step_rule(config.classes[2].balk_prob, step=class_2_step)},
        },
    )


def set_balk_thresholds(config, class_1_threshold, class_2_threshold):
    return update_classes(
        config,
        {
            1: {"balk_prob": make_step_rule(config.classes[1].balk_prob, threshold=class_1_threshold)},
            2: {"balk_prob": make_step_rule(config.classes[2].balk_prob, threshold=class_2_threshold)},
        },
    )


def set_balk_threshold_jump(config, threshold, jump_level):
    return update_classes(
        config,
        {
            1: {"balk_prob": make_step_rule(config.classes[1].balk_prob, threshold=threshold, step=jump_level)},
            2: {"balk_prob": make_step_rule(config.classes[2].balk_prob, threshold=threshold, step=jump_level)},
        },
    )


def set_no_show_steps(config, class_1_step, class_2_step):
    return update_classes(
        config,
        {
            1: {"no_show_prob": make_step_rule(config.classes[1].no_show_prob, step=class_1_step)},
            2: {"no_show_prob": make_step_rule(config.classes[2].no_show_prob, step=class_2_step)},
        },
    )


def set_no_show_thresholds(config, class_1_threshold, class_2_threshold):
    return update_classes(
        config,
        {
            1: {"no_show_prob": make_step_rule(config.classes[1].no_show_prob, threshold=class_1_threshold)},
            2: {"no_show_prob": make_step_rule(config.classes[2].no_show_prob, threshold=class_2_threshold)},
        },
    )


def set_no_show_threshold_jump(config, threshold, jump_level):
    return update_classes(
        config,
        {
            1: {"no_show_prob": make_step_rule(config.classes[1].no_show_prob, threshold=threshold, step=jump_level)},
            2: {"no_show_prob": make_step_rule(config.classes[2].no_show_prob, threshold=threshold, step=jump_level)},
        },
    )


def set_cancel_probs(config, class_1_cancel, class_2_cancel):
    return update_classes(
        config,
        {
            1: {"cancel_prob": float(class_1_cancel)},
            2: {"cancel_prob": float(class_2_cancel)},
        },
    )


def set_arrival_mix(config, lambda_total, class_1_share):
    return update_classes(
        config,
        {
            1: {"lambda_per_day": class_1_share * lambda_total},
            2: {"lambda_per_day": (1.0 - class_1_share) * lambda_total},
        },
    )


def set_class_arrival(config, target_class, lambda_per_day):
    return update_classes(config, {target_class: {"lambda_per_day": float(lambda_per_day)}})


def grid_records(x_values, y_values, x_name, y_name, config_builder):
    rows = []
    for x_value, y_value in product(x_values, y_values):
        config = config_builder(x_value, y_value)
        rows.append({x_name: x_value, y_name: y_value, **result_metrics(config)})
    return pd.DataFrame(rows)


def pivot(df, x_name, y_name, metric):
    return df.pivot(index=y_name, columns=x_name, values=metric).sort_index().sort_index(axis=1)


def heatmap_panel(ax, table, title, xlabel, ylabel, diverging=False):
    if diverging:
        max_abs = float(np.nanmax(np.abs(table.values)))
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs) if max_abs > 0 else None
        image = ax.imshow(table.values, origin="lower", aspect="auto", cmap="RdBu_r", norm=norm)
    else:
        image = ax.imshow(table.values, origin="lower", aspect="auto", cmap="viridis")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([f"{v:.2f}" if isinstance(v, float) else str(v) for v in table.columns], rotation=45)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([f"{v:.2f}" if isinstance(v, float) else str(v) for v in table.index])
    return image


def draw_four_panel(df, x_name, y_name, xlabel, ylabel, filename, title):
    specs = [
        ("average_utilization", "Average utilization", False),
        ("overall_percent_serviced", "Overall percent serviced", False),
        ("access_advantage_class_1", "Class 1 access advantage", True),
        ("delay_advantage_class_1", "Class 1 delay advantage", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    for ax, (metric, panel_title, diverging) in zip(axes.ravel(), specs):
        table = pivot(df, x_name, y_name, metric)
        image = heatmap_panel(ax, table, panel_title, xlabel, ylabel, diverging=diverging)
        fig.colorbar(image, ax=ax, shrink=0.85)

    fig.savefig(OUT_DIR / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)


def draw_threshold_jump_panel(df, x_name, y_name, xlabel, ylabel, filename, title):
    specs = [
        ("average_utilization", "Average utilization", False),
        ("overall_percent_serviced", "Overall percent serviced", False),
        ("mean_offered_booking_delay", "Mean offered delay", False),
        ("access_advantage_class_1", "Class 1 access advantage", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    for ax, (metric, panel_title, diverging) in zip(axes.ravel(), specs):
        table = pivot(df, x_name, y_name, metric)
        image = heatmap_panel(ax, table, panel_title, xlabel, ylabel, diverging=diverging)
        fig.colorbar(image, ax=ax, shrink=0.85)

    fig.savefig(OUT_DIR / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)


def draw_scenario_comparison():
    rows = []
    for name, config in [("Baseline", BASE_CONFIG), ("Scenario 2", SCENARIO_2_CONFIG)]:
        rows.append({"scenario": name, **mean_metrics(config, SCENARIO_SEEDS)})

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    df.set_index("scenario")[["average_utilization", "overall_percent_serviced"]].plot(kind="bar", ax=axes[0], rot=0)
    axes[0].set_title("Aggregate rates")
    axes[0].set_ylabel("rate")
    axes[0].set_ylim(0, 1.05)
    df.set_index("scenario")[["mean_accepted_booking_delay", "mean_offered_booking_delay"]].plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_title("Booking-delay metrics")
    axes[1].set_ylabel("days")
    df.set_index("scenario")[["access_advantage_class_1", "delay_advantage_class_1"]].plot(kind="bar", ax=axes[2], rot=0)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Class 1 advantage")
    axes[2].set_ylabel("difference")
    fig.savefig(OUT_DIR / "scenario_metric_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return df


def draw_arrival_mix():
    lambda_total_base = sum(params.lambda_per_day for params in BASE_CONFIG.classes.values())
    df = grid_records(
        ARRIVAL_MULTIPLIERS,
        CLASS_1_SHARES,
        "arrival_multiplier",
        "class_1_share",
        lambda multiplier, share: set_arrival_mix(BASE_CONFIG, lambda_total_base * multiplier, share),
    )
    draw_four_panel(
        df,
        "class_1_share",
        "arrival_multiplier",
        "class 1 share p",
        "total lambda multiplier",
        "arrival_mix_benefit_heatmaps.png",
        "Arrival pressure and class mix",
    )
    return df


def draw_class_arrival_lines():
    rows = []
    for target_class in sorted(BASE_CONFIG.classes):
        base_lambda = BASE_CONFIG.classes[target_class].lambda_per_day
        for multiplier in CLASS_ARRIVAL_MULTIPLIERS:
            lambda_per_day = base_lambda * multiplier
            config = set_class_arrival(BASE_CONFIG, target_class, lambda_per_day)
            metrics = result_metrics(config)
            rows.append(
                {
                    "target_class": target_class,
                    "lambda_per_day": lambda_per_day,
                    **metrics,
                }
            )
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for row_index, target_class in enumerate(sorted(BASE_CONFIG.classes)):
        data = df[df.target_class == target_class].sort_values("lambda_per_day")

        ax = axes[row_index, 0]
        ax.plot(data.lambda_per_day, data.average_utilization, marker="o", label="average utilization")
        ax.plot(data.lambda_per_day, data.overall_percent_serviced, marker="o", label="overall percent serviced")
        ax.plot(data.lambda_per_day, data[f"class_{target_class}_percent_serviced"], marker="o", label=f"class {target_class} percent serviced")
        ax.set_title(f"Class {target_class}: rates vs own arrival rate")
        ax.set_xlabel("lambda per day")
        ax.set_ylabel("rate")
        ax.legend()

        ax = axes[row_index, 1]
        ax.plot(data.lambda_per_day, data.access_advantage_class_1, marker="o", label="class 1 access advantage")
        ax.plot(data.lambda_per_day, data.delay_advantage_class_1, marker="o", label="class 1 delay advantage")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Class {target_class}: class advantage vs own arrival rate")
        ax.set_xlabel("lambda per day")
        ax.set_ylabel("difference")
        ax.legend()

    fig.savefig(OUT_DIR / "class_arrival_rate_benefit_lines.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return df


def summarize_direction(df, name):
    abs_access = df["access_advantage_class_1"].abs().max()
    abs_delay = df["delay_advantage_class_1"].abs().max()
    best_access = df.loc[df["access_advantage_class_1"].idxmax()]
    worst_access = df.loc[df["access_advantage_class_1"].idxmin()]
    best_delay = df.loc[df["delay_advantage_class_1"].idxmax()]
    worst_delay = df.loc[df["delay_advantage_class_1"].idxmin()]
    print(f"\n{name}")
    print(f"max |access advantage| = {abs_access:.4f}")
    print(f"max |delay advantage|  = {abs_delay:.4f}")
    print("best class 1 access row:")
    print(best_access.to_string())
    print("best class 2 access row:")
    print(worst_access.to_string())
    print("best class 1 delay row:")
    print(best_delay.to_string())
    print("best class 2 delay row:")
    print(worst_delay.to_string())


def main():
    plt.style.use("default")

    scenario_df = draw_scenario_comparison()

    balk_step_df = grid_records(
        STEP_GRID,
        STEP_GRID,
        "class_1_step",
        "class_2_step",
        lambda c1, c2: set_balk_steps(BASE_CONFIG, c1, c2),
    )
    draw_four_panel(
        balk_step_df,
        "class_1_step",
        "class_2_step",
        "class 1 balking step",
        "class 2 balking step",
        "balking_step_benefit_heatmaps.png",
        "Balking step size by class",
    )

    balk_threshold_df = grid_records(
        THRESHOLD_GRID,
        THRESHOLD_GRID,
        "class_1_threshold",
        "class_2_threshold",
        lambda c1, c2: set_balk_thresholds(BASE_CONFIG, c1, c2),
    )
    draw_four_panel(
        balk_threshold_df,
        "class_1_threshold",
        "class_2_threshold",
        "class 1 balking threshold",
        "class 2 balking threshold",
        "balking_threshold_benefit_heatmaps.png",
        "Balking threshold by class",
    )

    balk_threshold_jump_df = grid_records(
        STEP_GRID,
        THRESHOLD_GRID,
        "jump_level",
        "threshold",
        lambda jump_level, threshold: set_balk_threshold_jump(BASE_CONFIG, threshold, jump_level),
    )
    draw_threshold_jump_panel(
        balk_threshold_jump_df,
        "jump_level",
        "threshold",
        "balking jump level",
        "balking threshold",
        "balking_threshold_jump_heatmaps.png",
        "Balking threshold and jump level",
    )

    no_show_step_df = grid_records(
        STEP_GRID,
        STEP_GRID,
        "class_1_step",
        "class_2_step",
        lambda c1, c2: set_no_show_steps(BASE_CONFIG, c1, c2),
    )
    draw_four_panel(
        no_show_step_df,
        "class_1_step",
        "class_2_step",
        "class 1 no-show step",
        "class 2 no-show step",
        "no_show_step_benefit_heatmaps.png",
        "No-show step size by class",
    )

    no_show_threshold_df = grid_records(
        THRESHOLD_GRID,
        THRESHOLD_GRID,
        "class_1_threshold",
        "class_2_threshold",
        lambda c1, c2: set_no_show_thresholds(BASE_CONFIG, c1, c2),
    )
    draw_four_panel(
        no_show_threshold_df,
        "class_1_threshold",
        "class_2_threshold",
        "class 1 no-show threshold",
        "class 2 no-show threshold",
        "no_show_threshold_benefit_heatmaps.png",
        "No-show threshold by class",
    )

    no_show_threshold_jump_df = grid_records(
        STEP_GRID,
        THRESHOLD_GRID,
        "jump_level",
        "threshold",
        lambda jump_level, threshold: set_no_show_threshold_jump(BASE_CONFIG, threshold, jump_level),
    )
    draw_threshold_jump_panel(
        no_show_threshold_jump_df,
        "jump_level",
        "threshold",
        "no-show jump level",
        "no-show threshold",
        "no_show_threshold_jump_heatmaps.png",
        "No-show threshold and jump level",
    )

    cancel_df = grid_records(
        PROB_GRID,
        PROB_GRID,
        "class_1_cancel_prob",
        "class_2_cancel_prob",
        lambda c1, c2: set_cancel_probs(BASE_CONFIG, c1, c2),
    )
    draw_four_panel(
        cancel_df,
        "class_1_cancel_prob",
        "class_2_cancel_prob",
        "class 1 cancellation probability",
        "class 2 cancellation probability",
        "cancellation_benefit_heatmaps.png",
        "Cancellation probability by class",
    )

    arrival_df = draw_arrival_mix()
    class_arrival_df = draw_class_arrival_lines()

    print("\nScenario comparison")
    print(
        scenario_df[
            [
                "scenario",
                "average_utilization",
                "overall_percent_serviced",
                "mean_accepted_booking_delay",
                "mean_offered_booking_delay",
                "access_advantage_class_1",
                "delay_advantage_class_1",
            ]
        ].to_string(index=False)
    )

    summarize_direction(balk_step_df, "Balking step")
    summarize_direction(balk_threshold_df, "Balking threshold")
    summarize_direction(balk_threshold_jump_df, "Balking threshold and jump level")
    summarize_direction(no_show_step_df, "No-show step")
    summarize_direction(no_show_threshold_df, "No-show threshold")
    summarize_direction(no_show_threshold_jump_df, "No-show threshold and jump level")
    summarize_direction(cancel_df, "Cancellation")
    summarize_direction(arrival_df, "Arrival mix")
    summarize_direction(class_arrival_df, "Class arrival rate")


if __name__ == "__main__":
    main()
