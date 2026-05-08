from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import platform
import subprocess
import time
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
from analysis.metrics import (
    outcome_rates_from_result,
    result_metrics_from_result,
)
from analysis.plot_style import (
    ACCEPTED_WAIT_COLOR,
    ACCESS_CMAP,
    ACCESS_COLOR,
    ARRIVAL_COLOR,
    BALKING_CMAP,
    BALKING_COLOR,
    BASELINE_COLOR,
    CANCELLATION_COLOR,
    CLASS_1_COLOR,
    CLASS_2_COLOR,
    CLASS_GAP_CMAP,
    DRIVER_COLORS,
    NO_SHOW_COLOR,
    OVERALL_COLOR,
    UTILIZATION_CMAP,
    UTILIZATION_COLOR,
    WAIT_CMAP,
    WAIT_COLOR,
    blend_color,
    driver_from_text,
    driver_heatmap_cmap,
    plot_driver_line,
)


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
FCFS_STRESS_MULTIPLIERS = np.linspace(0.3, 1.7, 15)
REGRESSION_SCENARIOS = 240
REGRESSION_SEEDS = [4101, 4102]
REGRESSION_RANDOM_SEED = 9137

REGRESSION_FEATURES = [
    "lambda_total",
    "class_1_share",
    "balk_step_mean",
    "balk_step_gap_c1_minus_c2",
    "balk_threshold_mean",
    "balk_threshold_gap_c1_minus_c2",
    "no_show_step_mean",
    "no_show_step_gap_c1_minus_c2",
    "no_show_threshold_mean",
    "no_show_threshold_gap_c1_minus_c2",
    "cancel_prob_mean",
    "cancel_prob_gap_c1_minus_c2",
]

REGRESSION_TARGETS = {
    "average_utilization": "Average utilization",
    "overall_percent_serviced": "Overall percent serviced",
    "mean_offered_booking_delay": "Mean offered delay",
    "access_advantage_class_1": "Class 1 access advantage",
}

FEATURE_LABELS = {
    "lambda_total": "total arrival rate",
    "class_1_share": "class 1 arrival share",
    "balk_step_mean": "avg balking step",
    "balk_step_gap_c1_minus_c2": "balking step gap",
    "balk_threshold_mean": "avg balking threshold",
    "balk_threshold_gap_c1_minus_c2": "balking threshold gap",
    "no_show_step_mean": "avg no-show step",
    "no_show_step_gap_c1_minus_c2": "no-show step gap",
    "no_show_threshold_mean": "avg no-show threshold",
    "no_show_threshold_gap_c1_minus_c2": "no-show threshold gap",
    "cancel_prob_mean": "avg cancellation prob.",
    "cancel_prob_gap_c1_minus_c2": "cancellation prob. gap",
}

def run_result(config, seed):
    seeded = replace(config, seed=seed)
    return ClinicAppointmentSimulation(seeded).run()


def result_metrics(config, seed=FINE_SEED):
    return result_metrics_from_result(run_result(config, seed))


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


def set_regression_parameters(config, params):
    return update_classes(
        config,
        {
            1: {
                "lambda_per_day": params["lambda_total"] * params["class_1_share"],
                "cancel_prob": params["class_1_cancel_prob"],
                "balk_prob": ThresholdRule(
                    threshold=int(params["class_1_balk_threshold"]),
                    low=0.0,
                    high=params["class_1_balk_step"],
                ),
                "no_show_prob": ThresholdRule(
                    threshold=int(params["class_1_no_show_threshold"]),
                    low=0.0,
                    high=params["class_1_no_show_step"],
                ),
            },
            2: {
                "lambda_per_day": params["lambda_total"] * (1.0 - params["class_1_share"]),
                "cancel_prob": params["class_2_cancel_prob"],
                "balk_prob": ThresholdRule(
                    threshold=int(params["class_2_balk_threshold"]),
                    low=0.0,
                    high=params["class_2_balk_step"],
                ),
                "no_show_prob": ThresholdRule(
                    threshold=int(params["class_2_no_show_threshold"]),
                    low=0.0,
                    high=params["class_2_no_show_step"],
                ),
            },
        },
    )


def grid_records(x_values, y_values, x_name, y_name, config_builder):
    rows = []
    for x_value, y_value in product(x_values, y_values):
        config = config_builder(x_value, y_value)
        rows.append({x_name: x_value, y_name: y_value, **result_metrics(config)})
    return pd.DataFrame(rows)


def pivot(df, x_name, y_name, metric):
    return df.pivot(index=y_name, columns=x_name, values=metric).sort_index().sort_index(axis=1)


def format_axis_value(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def tick_positions(values, max_ticks=8):
    values = list(values)
    if len(values) <= max_ticks:
        return list(range(len(values)))
    positions = np.linspace(0, len(values) - 1, max_ticks)
    return sorted(set(int(round(position)) for position in positions))


def heatmap_panel(
    ax,
    table,
    title,
    xlabel,
    ylabel,
    diverging=False,
    vmin=None,
    vmax=None,
    cmap=None,
    max_ticks=8,
):
    if diverging:
        max_abs = float(np.nanmax(np.abs(table.values)))
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs) if max_abs > 0 else None
        image = ax.imshow(table.values, origin="lower", aspect="auto", cmap=cmap or CLASS_GAP_CMAP, norm=norm)
    else:
        image = ax.imshow(table.values, origin="lower", aspect="auto", cmap=cmap or ACCESS_CMAP, vmin=vmin, vmax=vmax)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    x_positions = tick_positions(table.columns, max_ticks=max_ticks)
    y_positions = tick_positions(table.index, max_ticks=max_ticks)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_axis_value(table.columns[position]) for position in x_positions], rotation=35, ha="right")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([format_axis_value(table.index[position]) for position in y_positions])
    ax.tick_params(axis="both", labelsize=9)
    return image


def nearest_position(values, target):
    numeric_values = [float(value) for value in values]
    return min(range(len(numeric_values)), key=lambda index: abs(numeric_values[index] - float(target)))


def mark_heatmap_slice(ax, table, fixed_y=None, fixed_x=None):
    if fixed_y is not None:
        y_pos = nearest_position(table.index, fixed_y)
        ax.axhline(y_pos, color="black", linestyle="--", linewidth=3.0)
        ax.axhline(y_pos, color="white", linestyle="--", linewidth=1.7)
    if fixed_x is not None:
        x_pos = nearest_position(table.columns, fixed_x)
        ax.axvline(x_pos, color="black", linestyle="--", linewidth=3.0)
        ax.axvline(x_pos, color="white", linestyle="--", linewidth=1.7)


def draw_four_panel(df, x_name, y_name, xlabel, ylabel, filename, title, fixed_y=None, fixed_x=None, subtitle=None, driver=None):
    driver = driver or driver_from_text(filename, title, xlabel, ylabel, x_name, y_name)
    specs = [
        ("average_utilization", "Average utilization", False),
        ("overall_percent_serviced", "Overall percent serviced", False),
        ("access_advantage_class_1", "Class 1 access advantage", True),
        ("delay_advantage_class_1", "Class 1 delay advantage", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
    fig.suptitle(f"{title}\n{subtitle}" if subtitle else title, fontsize=14)

    for ax, (metric, panel_title, diverging) in zip(axes.ravel(), specs):
        table = pivot(df, x_name, y_name, metric)
        _, colorbar_label = metric_heatmap_style(metric)
        cmap = driver_heatmap_cmap(driver, diverging=diverging)
        image = heatmap_panel(ax, table, panel_title, xlabel, ylabel, diverging=diverging, cmap=cmap)
        mark_heatmap_slice(ax, table, fixed_y=fixed_y, fixed_x=fixed_x)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
        colorbar.set_label(colorbar_label)

    fig.savefig(OUT_DIR / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)


def draw_behavior_panel(
    df,
    x_name,
    y_name,
    xlabel,
    ylabel,
    filename,
    title,
    fixed_y=None,
    fixed_x=None,
    subtitle=None,
    driver=None,
):
    driver = driver or driver_from_text(filename, title, xlabel, ylabel, x_name, y_name)
    specs = [
        ("overall_percent_serviced", "Overall served rate", "rate", False),
        ("class_1_percent_serviced", "Class 1 served rate", "rate", False),
        ("class_2_percent_serviced", "Class 2 served rate", "rate", False),
        ("mean_offered_booking_delay", "Overall offered wait", "days", False),
        ("class_1_mean_offered_booking_delay", "Class 1 offered wait", "days", False),
        ("class_2_mean_offered_booking_delay", "Class 2 offered wait", "days", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), constrained_layout=True)
    fig.suptitle(f"{title}\n{subtitle}" if subtitle else title, fontsize=14)

    rate_metrics = [metric for metric, _, units, _ in specs if units == "rate"]
    wait_metrics = [metric for metric, _, units, _ in specs if units == "days"]
    rate_vmin = float(df[rate_metrics].min().min())
    rate_vmax = float(df[rate_metrics].max().max())
    wait_vmin = float(df[wait_metrics].min().min())
    wait_vmax = float(df[wait_metrics].max().max())

    for ax, (metric, panel_title, units, diverging) in zip(axes.ravel(), specs):
        table = pivot(df, x_name, y_name, metric)
        cmap = driver_heatmap_cmap(driver, diverging=diverging)
        if units == "rate":
            image = heatmap_panel(
                ax,
                table,
                panel_title,
                xlabel,
                ylabel,
                diverging=diverging,
                vmin=rate_vmin,
                vmax=rate_vmax,
                cmap=cmap,
            )
        else:
            image = heatmap_panel(
                ax,
                table,
                panel_title,
                xlabel,
                ylabel,
                diverging=diverging,
                vmin=wait_vmin,
                vmax=wait_vmax,
                cmap=cmap,
            )
        mark_heatmap_slice(ax, table, fixed_y=fixed_y, fixed_x=fixed_x)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
        colorbar.set_label(units)

    fig.savefig(OUT_DIR / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)


def metric_heatmap_style(metric):
    if "advantage" in metric or "gap" in metric:
        return CLASS_GAP_CMAP, "Class 1 - Class 2"
    if "balking_rate" in metric:
        return BALKING_CMAP, "rate"
    if metric == "average_utilization":
        return UTILIZATION_CMAP, "rate"
    if metric in {"overall_percent_serviced", "class_1_percent_serviced", "class_2_percent_serviced"}:
        return ACCESS_CMAP, "rate"
    if "booking_delay" in metric:
        return WAIT_CMAP, "days"
    return ACCESS_CMAP, "value"


def draw_two_metric_heatmap(df, x_name, y_name, xlabel, ylabel, filename, title, panels, fixed_y=None, fixed_x=None, driver=None):
    driver = driver or driver_from_text(filename, title, xlabel, ylabel, x_name, y_name)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)
    fig.suptitle(title, fontsize=15)

    for ax, panel in zip(axes, panels):
        table = pivot(df, x_name, y_name, panel["metric"])
        cmap, default_label = metric_heatmap_style(panel["metric"])
        diverging = panel.get("diverging", False)
        panel_cmap = driver_heatmap_cmap(driver, diverging=diverging) if driver else panel.get("cmap", CLASS_GAP_CMAP if diverging else cmap)
        image = heatmap_panel(
            ax,
            table,
            panel["title"],
            xlabel,
            ylabel,
            diverging=diverging,
            cmap=panel_cmap,
            max_ticks=panel.get("max_ticks", 8),
        )
        mark_heatmap_slice(ax, table, fixed_y=fixed_y, fixed_x=fixed_x)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.88)
        colorbar.set_label(panel.get("label", default_label))

    fig.savefig(OUT_DIR / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)


def nearest_value(values, target):
    values = sorted(values)
    return values[nearest_position(values, target)]


def slice_with_fixed(df, x_name, y_name, fixed_axis, fixed_value):
    if fixed_axis == y_name:
        actual_value = nearest_value(df[y_name].unique(), fixed_value)
        return df[df[y_name] == actual_value].sort_values(x_name).copy(), actual_value
    actual_value = nearest_value(df[x_name].unique(), fixed_value)
    return df[df[x_name] == actual_value].sort_values(y_name).copy(), actual_value


def diagonal_slice(df, x_name, y_name, common_name):
    mask = np.isclose(df[x_name].astype(float), df[y_name].astype(float))
    data = df[mask].copy()
    data[common_name] = data[x_name]
    return data.sort_values(common_name)


def style_line_axis(ax, xlabel, ylabel, y_range=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_range is not None:
        ax.set_ylim(*y_range)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_slice_metric_figure(
    df,
    x_name,
    y_name,
    baseline_x,
    baseline_y,
    xlabel,
    ylabel,
    filename,
    title,
    metrics,
    metric_ylabel,
    y_range=None,
    driver=None,
):
    driver = driver or driver_from_text(filename, title, xlabel, ylabel, x_name, y_name)
    left_df, left_fixed = slice_with_fixed(df, x_name, y_name, y_name, baseline_y)
    right_df, right_fixed = slice_with_fixed(df, x_name, y_name, x_name, baseline_x)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True, sharey=y_range is not None)
    fig.suptitle(title, fontsize=15)

    for index, (metric, label, color) in enumerate(metrics):
        plot_driver_line(axes[0], left_df[x_name], left_df[metric], label, driver=driver, color=color, index=index)
    axes[0].axvline(baseline_x, color=BASELINE_COLOR, linestyle="--", linewidth=1.2, label="baseline")
    axes[0].set_title(f"Class 1 varies; Class 2 fixed at {format_axis_value(left_fixed)}")
    style_line_axis(axes[0], xlabel, metric_ylabel, y_range=y_range)

    for index, (metric, label, color) in enumerate(metrics):
        plot_driver_line(axes[1], right_df[y_name], right_df[metric], label, driver=driver, color=color, index=index)
    axes[1].axvline(baseline_y, color=BASELINE_COLOR, linestyle="--", linewidth=1.2, label="baseline")
    axes[1].set_title(f"Class 2 varies; Class 1 fixed at {format_axis_value(right_fixed)}")
    style_line_axis(axes[1], ylabel, metric_ylabel, y_range=y_range)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False)
    fig.savefig(OUT_DIR / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)


def draw_behavior_slice_figures(df, x_name, y_name, baseline_x, baseline_y, xlabel, ylabel, prefix, title):
    driver = driver_from_text(prefix, title, xlabel, ylabel, x_name, y_name)
    draw_slice_metric_figure(
        df,
        x_name,
        y_name,
        baseline_x,
        baseline_y,
        xlabel,
        ylabel,
        f"{prefix}_slice_access.png",
        f"{title}: served rate slices",
        [
            ("overall_percent_serviced", "overall", OVERALL_COLOR),
            ("class_1_percent_serviced", "Class 1", CLASS_1_COLOR),
            ("class_2_percent_serviced", "Class 2", CLASS_2_COLOR),
        ],
        "served rate",
        y_range=(0, 1.05),
        driver=driver,
    )
    draw_slice_metric_figure(
        df,
        x_name,
        y_name,
        baseline_x,
        baseline_y,
        xlabel,
        ylabel,
        f"{prefix}_slice_utilization.png",
        f"{title}: utilization slices",
        [("average_utilization", "utilization", UTILIZATION_COLOR)],
        "utilization",
        y_range=(0, 1.05),
        driver=driver,
    )
    draw_slice_metric_figure(
        df,
        x_name,
        y_name,
        baseline_x,
        baseline_y,
        xlabel,
        ylabel,
        f"{prefix}_slice_wait.png",
        f"{title}: offered-wait slices",
        [
            ("mean_offered_booking_delay", "overall", OVERALL_COLOR),
            ("class_1_mean_offered_booking_delay", "Class 1", CLASS_1_COLOR),
            ("class_2_mean_offered_booking_delay", "Class 2", CLASS_2_COLOR),
        ],
        "offered wait (days)",
        driver=driver,
    )
    if prefix.startswith("balking"):
        draw_slice_metric_figure(
            df,
            x_name,
            y_name,
            baseline_x,
            baseline_y,
            xlabel,
            ylabel,
            f"{prefix}_slice_balking_rate.png",
            f"{title}: balking-rate slices",
            [
                ("overall_balking_rate", "overall", OVERALL_COLOR),
                ("class_1_balking_rate", "Class 1", CLASS_1_COLOR),
                ("class_2_balking_rate", "Class 2", CLASS_2_COLOR),
            ],
            "balking rate among offered patients",
            y_range=(0, 1.05),
            driver=driver,
        )


def draw_arrival_rate_slice_figures(df, baseline_share):
    slice_df, actual_share = slice_with_fixed(df, "lambda_total", "class_1_share", "class_1_share", baseline_share)
    specs = [
        (
            "arrival_rate_slice_access.png",
            "Arrival rate at 50/50 mix: served rate",
            [
                ("overall_percent_serviced", "overall", OVERALL_COLOR),
                ("class_1_percent_serviced", "Class 1", CLASS_1_COLOR),
                ("class_2_percent_serviced", "Class 2", CLASS_2_COLOR),
            ],
            "served rate",
            (0, 1.05),
        ),
        (
            "arrival_rate_slice_utilization.png",
            "Arrival rate at 50/50 mix: utilization",
            [("average_utilization", "utilization", UTILIZATION_COLOR)],
            "utilization",
            (0, 1.05),
        ),
        (
            "arrival_rate_slice_wait.png",
            "Arrival rate at 50/50 mix: offered wait",
            [
                ("mean_offered_booking_delay", "overall", OVERALL_COLOR),
                ("class_1_mean_offered_booking_delay", "Class 1", CLASS_1_COLOR),
                ("class_2_mean_offered_booking_delay", "Class 2", CLASS_2_COLOR),
            ],
            "offered wait (days)",
            None,
        ),
    ]

    for filename, title, metrics, ylabel, y_range in specs:
        driver = "arrival"
        fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
        for index, (metric, label, color) in enumerate(metrics):
            plot_driver_line(ax, slice_df["lambda_total"], slice_df[metric], label, driver=driver, color=color, index=index)
        ax.axvline(sum(params.lambda_per_day for params in BASE_CONFIG.classes.values()), color=BASELINE_COLOR, linestyle="--", linewidth=1.2, label="baseline")
        ax.set_title(f"{title}\nClass 1 share fixed at {actual_share:.2f}")
        style_line_axis(ax, "total arrivals per day", ylabel, y_range=y_range)
        ax.legend(frameon=False)
        fig.savefig(OUT_DIR / filename, dpi=190, bbox_inches="tight")
        plt.close(fig)


def draw_single_metric_heatmap(
    df,
    x_name,
    y_name,
    metric,
    xlabel,
    ylabel,
    filename,
    title,
    colorbar_label,
    fixed_y=None,
    fixed_x=None,
    vmin=None,
    vmax=None,
    driver=None,
):
    driver = driver or driver_from_text(filename, title, xlabel, ylabel, x_name, y_name)
    table = pivot(df, x_name, y_name, metric)
    cmap = driver_heatmap_cmap(driver, diverging=False) if driver else metric_heatmap_style(metric)[0]

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    image = heatmap_panel(
        ax,
        table,
        title,
        xlabel,
        ylabel,
        diverging=False,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    mark_heatmap_slice(ax, table, fixed_y=fixed_y, fixed_x=fixed_x)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label(colorbar_label)
    fig.savefig(OUT_DIR / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)


def draw_class_service_pair_heatmaps(
    df,
    x_name,
    y_name,
    xlabel,
    ylabel,
    class_1_filename,
    class_2_filename,
    class_1_title,
    class_2_title,
    fixed_y=None,
    fixed_x=None,
):
    vmin = float(df[["class_1_percent_serviced", "class_2_percent_serviced"]].min().min())
    vmax = float(df[["class_1_percent_serviced", "class_2_percent_serviced"]].max().max())

    draw_single_metric_heatmap(
        df,
        x_name,
        y_name,
        "class_1_percent_serviced",
        xlabel,
        ylabel,
        class_1_filename,
        class_1_title,
        "class 1 percent serviced",
        fixed_y=fixed_y,
        fixed_x=fixed_x,
        vmin=vmin,
        vmax=vmax,
    )
    draw_single_metric_heatmap(
        df,
        x_name,
        y_name,
        "class_2_percent_serviced",
        xlabel,
        ylabel,
        class_2_filename,
        class_2_title,
        "class 2 percent serviced",
        fixed_y=fixed_y,
        fixed_x=fixed_x,
        vmin=vmin,
        vmax=vmax,
    )


def draw_balking_class_service_heatmaps(df, baseline_balk_step):
    draw_class_service_pair_heatmaps(
        df,
        "class_1_step",
        "class_2_step",
        "class 1 balking step",
        "class 2 balking step",
        "balking_step_class1_service_heatmap.png",
        "balking_step_class2_service_heatmap.png",
        "Class 1 served rate under balking step changes",
        "Class 2 served rate under balking step changes",
        fixed_y=baseline_balk_step,
    )


def draw_balking_threshold_class_service_heatmaps(df, baseline_balk_threshold):
    draw_class_service_pair_heatmaps(
        df,
        "class_1_threshold",
        "class_2_threshold",
        "class 1 balking threshold",
        "class 2 balking threshold",
        "balking_threshold_class1_service_heatmap.png",
        "balking_threshold_class2_service_heatmap.png",
        "Class 1 served rate under balking threshold changes",
        "Class 2 served rate under balking threshold changes",
        fixed_y=baseline_balk_threshold,
    )


def draw_balking_slice_lines(df, x_name, y_name, fixed_y, baseline_x, xlabel, fixed_label, filename, title):
    driver = "balking"
    y_values = sorted(df[y_name].unique())
    nearest_y = y_values[nearest_position(y_values, fixed_y)]
    slice_df = df[df[y_name] == nearest_y].copy()
    slice_df = slice_df.sort_values(x_name)
    slice_df.to_csv(OUT_DIR / filename.replace(".png", ".csv"), index=False)

    specs = [
        ("average_utilization", "Average utilization", "rate"),
        ("overall_percent_serviced", "Overall percent serviced", "rate"),
        ("mean_offered_booking_delay", "Mean offered delay", "days"),
        ("access_advantage_class_1", "Class 1 access advantage", "difference"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    fig.suptitle(f"{title}\n{fixed_label}; all other parameters fixed at baseline", fontsize=14)

    for index, (ax, (metric, panel_title, ylabel)) in enumerate(zip(axes.ravel(), specs)):
        plot_driver_line(ax, slice_df[x_name], slice_df[metric], panel_title, driver=driver, index=index)
        ax.axvline(baseline_x, color=BASELINE_COLOR, linestyle="--", linewidth=1.1, label="class 1 baseline")
        if "advantage" in metric:
            ax.axhline(0, color=BASELINE_COLOR, linewidth=0.9)
        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

    fig.savefig(OUT_DIR / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return slice_df


def draw_threshold_jump_panel(df, x_name, y_name, xlabel, ylabel, filename, title, driver=None):
    driver = driver or driver_from_text(filename, title, xlabel, ylabel, x_name, y_name)
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
        _, colorbar_label = metric_heatmap_style(metric)
        cmap = driver_heatmap_cmap(driver, diverging=diverging)
        image = heatmap_panel(ax, table, panel_title, xlabel, ylabel, diverging=diverging, cmap=cmap)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
        colorbar.set_label(colorbar_label)

    fig.savefig(OUT_DIR / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)


def draw_scenario_comparison():
    rows = []
    for name, config in [("Baseline", BASE_CONFIG), ("Scenario 2", SCENARIO_2_CONFIG)]:
        rows.append({"scenario": name, **mean_metrics(config, SCENARIO_SEEDS)})

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    df.set_index("scenario")[["average_utilization", "overall_percent_serviced"]].plot(
        kind="bar",
        ax=axes[0],
        rot=0,
        color=[UTILIZATION_COLOR, ACCESS_COLOR],
    )
    axes[0].set_title("Aggregate rates")
    axes[0].set_ylabel("rate")
    axes[0].set_ylim(0, 1.05)
    df.set_index("scenario")[["mean_accepted_booking_delay", "mean_offered_booking_delay"]].plot(
        kind="bar",
        ax=axes[1],
        rot=0,
        color=[ACCEPTED_WAIT_COLOR, WAIT_COLOR],
    )
    axes[1].set_title("Booking-delay metrics")
    axes[1].set_ylabel("days")
    df.set_index("scenario")[["access_advantage_class_1", "delay_advantage_class_1"]].plot(
        kind="bar",
        ax=axes[2],
        rot=0,
        color=[CLASS_1_COLOR, CLASS_2_COLOR],
    )
    axes[2].axhline(0, color=BASELINE_COLOR, linewidth=0.8)
    axes[2].set_title("Class 1 advantage")
    axes[2].set_ylabel("difference")
    fig.savefig(OUT_DIR / "scenario_metric_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return df


def draw_arrival_mix():
    lambda_total_base = sum(params.lambda_per_day for params in BASE_CONFIG.classes.values())
    lambda_total_values = lambda_total_base * ARRIVAL_MULTIPLIERS
    df = grid_records(
        lambda_total_values,
        CLASS_1_SHARES,
        "lambda_total",
        "class_1_share",
        lambda lambda_total, share: set_arrival_mix(BASE_CONFIG, lambda_total, share),
    )
    df["arrival_multiplier"] = df["lambda_total"] / lambda_total_base
    draw_four_panel(
        df,
        "class_1_share",
        "lambda_total",
        "class 1 share p",
        "total arrivals per day",
        "arrival_mix_benefit_heatmaps.png",
        "Arrival pressure and class mix",
    )
    draw_behavior_panel(
        df,
        "class_1_share",
        "lambda_total",
        "class 1 share p",
        "total arrivals per day",
        "arrival_mix_behavior_panels.png",
        "Arrival rate and class mix",
    )
    draw_two_metric_heatmap(
        df,
        "class_1_share",
        "lambda_total",
        "Class 1 arrival share",
        "total arrivals per day",
        "arrival_mix_interaction_heatmap.png",
        "Arrival rate and class mix",
        [
            {"metric": "overall_percent_serviced", "title": "Overall served rate", "label": "rate", "cmap": ACCESS_CMAP},
            {"metric": "mean_offered_booking_delay", "title": "Overall offered wait", "label": "days", "cmap": WAIT_CMAP},
        ],
    )
    baseline_share = BASE_CONFIG.classes[1].lambda_per_day / lambda_total_base
    draw_arrival_rate_slice_figures(df, baseline_share)
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
        plot_driver_line(ax, data.lambda_per_day, data.average_utilization, "average utilization", driver="arrival", index=0)
        plot_driver_line(ax, data.lambda_per_day, data.overall_percent_serviced, "overall percent serviced", driver="arrival", index=1)
        plot_driver_line(ax, data.lambda_per_day, data[f"class_{target_class}_percent_serviced"], f"class {target_class} percent serviced", driver="arrival", index=2)
        ax.set_title(f"Class {target_class}: rates vs own arrival rate")
        ax.set_xlabel("lambda per day")
        ax.set_ylabel("rate")
        ax.legend()

        ax = axes[row_index, 1]
        plot_driver_line(ax, data.lambda_per_day, data.access_advantage_class_1, "class 1 access advantage", driver="arrival", index=0)
        plot_driver_line(ax, data.lambda_per_day, data.delay_advantage_class_1, "class 1 delay advantage", driver="arrival", index=1)
        ax.axhline(0, color=BASELINE_COLOR, linewidth=0.8)
        ax.set_title(f"Class {target_class}: class advantage vs own arrival rate")
        ax.set_xlabel("lambda per day")
        ax.set_ylabel("difference")
        ax.legend()

    fig.savefig(OUT_DIR / "class_arrival_rate_benefit_lines.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return df


def draw_fcfs_capacity_stress():
    lambda_total_base = sum(params.lambda_per_day for params in BASE_CONFIG.classes.values())
    class_1_share = BASE_CONFIG.classes[1].lambda_per_day / lambda_total_base

    rows = []
    for multiplier in FCFS_STRESS_MULTIPLIERS:
        lambda_total = lambda_total_base * multiplier
        config = set_arrival_mix(BASE_CONFIG, lambda_total, class_1_share)
        seed_rows = []
        for seed in SCENARIO_SEEDS:
            result = run_result(config, seed)
            seed_rows.append(
                {
                    "arrival_multiplier": multiplier,
                    "lambda_total": lambda_total,
                    **result_metrics_from_result(result),
                    **outcome_rates_from_result(result),
                }
            )
        rows.append(pd.DataFrame(seed_rows).mean(numeric_only=True).to_dict())

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "fcfs_capacity_stress_results.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    baseline_lambda_total = lambda_total_base
    plot_driver_line(ax, df.lambda_total, df.average_utilization, "average utilization", driver="arrival", index=0)
    plot_driver_line(ax, df.lambda_total, df.overall_percent_serviced, "overall percent serviced", driver="arrival", index=1)
    ax.axvline(baseline_lambda_total, color=BASELINE_COLOR, linewidth=0.8, linestyle="--")
    ax.set_title("Capacity and access")
    ax.set_xlabel("total arrivals per day")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.legend()

    ax = axes[0, 1]
    plot_driver_line(ax, df.lambda_total, df.mean_accepted_booking_delay, "accepted delay", driver="arrival", index=0)
    plot_driver_line(ax, df.lambda_total, df.mean_offered_booking_delay, "offered delay", driver="arrival", index=1)
    ax.axvline(baseline_lambda_total, color=BASELINE_COLOR, linewidth=0.8, linestyle="--")
    ax.set_title("Booking delay")
    ax.set_xlabel("total arrivals per day")
    ax.set_ylabel("days")
    ax.legend()

    ax = axes[1, 0]
    plot_driver_line(ax, df.lambda_total, df.class_1_percent_serviced, "class 1", driver="arrival", index=0)
    plot_driver_line(ax, df.lambda_total, df.class_2_percent_serviced, "class 2", driver="arrival", index=1)
    ax.axvline(baseline_lambda_total, color=BASELINE_COLOR, linewidth=0.8, linestyle="--")
    ax.set_title("Class access under FCFS")
    ax.set_xlabel("total arrivals per day")
    ax.set_ylabel("percent serviced")
    ax.set_ylim(0, 1.05)
    ax.legend()

    ax = axes[1, 1]
    plot_driver_line(ax, df.lambda_total, df.balked_rate, "balked", driver="balking", index=0)
    plot_driver_line(ax, df.lambda_total, df.no_offer_rate, "no offer", driver="arrival", index=1)
    plot_driver_line(ax, df.lambda_total, df.lost_after_booking_rate, "lost after booking", driver="cancellation", index=2)
    ax.axvline(baseline_lambda_total, color=BASELINE_COLOR, linewidth=0.8, linestyle="--")
    ax.set_title("Main loss channels")
    ax.set_xlabel("total arrivals per day")
    ax.set_ylabel("share of arrivals")
    ax.set_ylim(0, 1.05)
    ax.legend()

    fig.savefig(OUT_DIR / "fcfs_capacity_stress_curves.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)
    plot_driver_line(axes[0], df.lambda_total, df.average_utilization, "utilization", driver="arrival", index=0)
    plot_driver_line(axes[0], df.lambda_total, df.overall_percent_serviced, "overall served rate", driver="arrival", index=1)
    axes[0].axvline(baseline_lambda_total, color=BASELINE_COLOR, linewidth=1.2, linestyle="--", label="baseline")
    axes[0].set_title("Capacity use vs access")
    style_line_axis(axes[0], "total arrivals per day", "rate", y_range=(0, 1.05))
    axes[0].legend(frameon=False)

    plot_driver_line(axes[1], df.lambda_total, df.mean_offered_booking_delay, "offered wait", driver="arrival", index=0)
    plot_driver_line(axes[1], df.lambda_total, df.mean_accepted_booking_delay, "accepted wait", driver="arrival", index=1)
    axes[1].axvline(baseline_lambda_total, color=BASELINE_COLOR, linewidth=1.2, linestyle="--", label="baseline")
    axes[1].set_title("Waits under demand pressure")
    style_line_axis(axes[1], "total arrivals per day", "days")
    axes[1].legend(frameon=False)
    fig.savefig(OUT_DIR / "fcfs_stress_access_wait.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    outcomes = [
        ("served_rate", "served"),
        ("no_show_rate", "no-show"),
        ("canceled_rate", "canceled"),
        ("unresolved_booked_rate", "unresolved booked"),
        ("balked_rate", "balked"),
        ("no_offer_rate", "no offer"),
    ]
    ax.stackplot(
        df.lambda_total,
        *[df[column] for column, _ in outcomes],
        labels=[label for _, label in outcomes],
        colors=[
            ARRIVAL_COLOR,
            NO_SHOW_COLOR,
            CANCELLATION_COLOR,
            blend_color(BASELINE_COLOR, amount=0.35),
            BALKING_COLOR,
            blend_color(ARRIVAL_COLOR, amount=0.45),
        ],
        alpha=0.9,
    )
    ax.axvline(baseline_lambda_total, color=BASELINE_COLOR, linewidth=0.8, linestyle="--")
    ax.set_title("Final outcome decomposition per arrival")
    ax.set_xlabel("total arrivals per day")
    ax.set_ylabel("share of arrivals")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(OUT_DIR / "fcfs_arrival_outcome_decomposition.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    return df


def draw_metric_driver_panel(
    ax,
    data,
    x_name,
    series,
    title,
    xlabel,
    ylabel,
    baseline_x=None,
    y_range=None,
    zero_line=False,
    driver=None,
):
    driver = driver or driver_from_text(title, xlabel, x_name)
    for index, (metric, label, color) in enumerate(series):
        plot_driver_line(ax, data[x_name], data[metric], label, driver=driver, color=color, index=index)
    if baseline_x is not None:
        ax.axvline(baseline_x, color=BASELINE_COLOR, linestyle="--", linewidth=1.2, label="baseline")
    if zero_line:
        ax.axhline(0, color=BASELINE_COLOR, linewidth=0.9)
    ax.set_title(title)
    style_line_axis(ax, xlabel, ylabel, y_range=y_range)
    if len(series) > 1 or baseline_x is not None:
        ax.legend(frameon=False, fontsize=9)


def draw_metric_driver_figure(filename, title, panels):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), constrained_layout=True)
    fig.suptitle(title, fontsize=15)

    for ax, panel in zip(axes.ravel(), panels):
        draw_metric_driver_panel(ax=ax, **panel)

    fig.savefig(OUT_DIR / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)


def draw_metric_driver_figures(
    class_arrival_df,
    balk_step_df,
    balk_threshold_df,
    no_show_step_df,
    no_show_threshold_df,
    cancel_df,
    baseline_balk_step,
    baseline_balk_threshold,
    baseline_no_show_step,
    baseline_no_show_threshold,
    baseline_cancel_prob,
):
    baseline_class_1_arrival = BASE_CONFIG.classes[1].lambda_per_day
    class_1_arrival = class_arrival_df[class_arrival_df["target_class"] == 1].sort_values("lambda_per_day").copy()
    balk_step_class_1, _ = slice_with_fixed(
        balk_step_df,
        "class_1_step",
        "class_2_step",
        "class_2_step",
        baseline_balk_step,
    )
    balk_threshold_class_1, _ = slice_with_fixed(
        balk_threshold_df,
        "class_1_threshold",
        "class_2_threshold",
        "class_2_threshold",
        baseline_balk_threshold,
    )
    no_show_step_class_1, _ = slice_with_fixed(
        no_show_step_df,
        "class_1_step",
        "class_2_step",
        "class_2_step",
        baseline_no_show_step,
    )
    no_show_threshold_class_1, _ = slice_with_fixed(
        no_show_threshold_df,
        "class_1_threshold",
        "class_2_threshold",
        "class_2_threshold",
        baseline_no_show_threshold,
    )
    cancel_class_1, _ = slice_with_fixed(
        cancel_df,
        "class_1_cancel_prob",
        "class_2_cancel_prob",
        "class_2_cancel_prob",
        baseline_cancel_prob,
    )

    slot_utilization_series = [
        ("average_utilization", "overall", OVERALL_COLOR),
        ("class_1_slot_utilization", "Class 1", CLASS_1_COLOR),
        ("class_2_slot_utilization", "Class 2", CLASS_2_COLOR),
    ]
    access_series = [
        ("overall_percent_serviced", "overall", OVERALL_COLOR),
        ("class_1_percent_serviced", "Class 1", CLASS_1_COLOR),
        ("class_2_percent_serviced", "Class 2", CLASS_2_COLOR),
    ]
    wait_series = [
        ("mean_offered_booking_delay", "overall", OVERALL_COLOR),
        ("class_1_mean_offered_booking_delay", "Class 1", CLASS_1_COLOR),
        ("class_2_mean_offered_booking_delay", "Class 2", CLASS_2_COLOR),
    ]
    balking_series = [
        ("overall_balking_rate", "overall", OVERALL_COLOR),
        ("class_1_balking_rate", "Class 1", CLASS_1_COLOR),
        ("class_2_balking_rate", "Class 2", CLASS_2_COLOR),
    ]

    draw_metric_driver_figure(
        "metric_utilization_drivers.png",
        "Average utilization: Class 1 slices with Class 2 fixed",
        [
            {
                "data": class_1_arrival,
                "x_name": "lambda_per_day",
                "series": slot_utilization_series,
                "title": "Class 1 arrival rate changes",
                "xlabel": "Class 1 arrivals per day",
                "ylabel": "slot share",
                "baseline_x": baseline_class_1_arrival,
                "y_range": (0, 1.05),
            },
            {
                "data": no_show_step_class_1,
                "x_name": "class_1_step",
                "series": slot_utilization_series,
                "title": "Class 1 no-show step changes",
                "xlabel": "Class 1 no-show high-delay probability",
                "ylabel": "slot share",
                "baseline_x": baseline_no_show_step,
                "y_range": (0, 1.05),
            },
            {
                "data": no_show_threshold_class_1,
                "x_name": "class_1_threshold",
                "series": slot_utilization_series,
                "title": "Class 1 no-show threshold changes",
                "xlabel": "Class 1 no-show threshold (days)",
                "ylabel": "slot share",
                "baseline_x": baseline_no_show_threshold,
                "y_range": (0, 1.05),
            },
            {
                "data": cancel_class_1,
                "x_name": "class_1_cancel_prob",
                "series": slot_utilization_series,
                "title": "Class 1 cancellation changes",
                "xlabel": "Class 1 cancellation probability",
                "ylabel": "slot share",
                "baseline_x": baseline_cancel_prob,
                "y_range": (0, 1.05),
            },
        ],
    )

    draw_metric_driver_figure(
        "metric_access_drivers.png",
        "Served rate: Class 1 slices with Class 2 fixed",
        [
            {
                "data": class_1_arrival,
                "x_name": "lambda_per_day",
                "series": access_series,
                "title": "Class 1 arrival rate changes",
                "xlabel": "Class 1 arrivals per day",
                "ylabel": "served rate",
                "baseline_x": baseline_class_1_arrival,
                "y_range": (0, 1.05),
            },
            {
                "data": balk_step_class_1,
                "x_name": "class_1_step",
                "series": access_series,
                "title": "Class 1 balking step changes",
                "xlabel": "Class 1 balking high-delay probability",
                "ylabel": "served rate",
                "baseline_x": baseline_balk_step,
                "y_range": (0, 1.05),
            },
            {
                "data": no_show_step_class_1,
                "x_name": "class_1_step",
                "series": access_series,
                "title": "Class 1 no-show step changes",
                "xlabel": "Class 1 no-show high-delay probability",
                "ylabel": "served rate",
                "baseline_x": baseline_no_show_step,
                "y_range": (0, 1.05),
            },
            {
                "data": cancel_class_1,
                "x_name": "class_1_cancel_prob",
                "series": access_series,
                "title": "Class 1 cancellation changes",
                "xlabel": "Class 1 cancellation probability",
                "ylabel": "served rate",
                "baseline_x": baseline_cancel_prob,
                "y_range": (0, 1.05),
            },
        ],
    )

    draw_metric_driver_figure(
        "metric_wait_drivers.png",
        "Mean offered booking delay: Class 1 slices with Class 2 fixed",
        [
            {
                "data": class_1_arrival,
                "x_name": "lambda_per_day",
                "series": wait_series,
                "title": "Class 1 arrival rate changes",
                "xlabel": "Class 1 arrivals per day",
                "ylabel": "days",
                "baseline_x": baseline_class_1_arrival,
            },
            {
                "data": balk_step_class_1,
                "x_name": "class_1_step",
                "series": wait_series,
                "title": "Class 1 balking step changes",
                "xlabel": "Class 1 balking high-delay probability",
                "ylabel": "days",
                "baseline_x": baseline_balk_step,
            },
            {
                "data": balk_threshold_class_1,
                "x_name": "class_1_threshold",
                "series": wait_series,
                "title": "Class 1 balking threshold changes",
                "xlabel": "Class 1 balking threshold (days)",
                "ylabel": "days",
                "baseline_x": baseline_balk_threshold,
            },
            {
                "data": cancel_class_1,
                "x_name": "class_1_cancel_prob",
                "series": wait_series,
                "title": "Class 1 cancellation changes",
                "xlabel": "Class 1 cancellation probability",
                "ylabel": "days",
                "baseline_x": baseline_cancel_prob,
            },
        ],
    )

    draw_metric_driver_figure(
        "metric_class_gap_drivers.png",
        "Served-rate values behind class gaps: Class 1 slices",
        [
            {
                "data": cancel_class_1,
                "x_name": "class_1_cancel_prob",
                "series": access_series,
                "title": "Class 1 cancellation changes",
                "xlabel": "Class 1 cancellation probability",
                "ylabel": "served rate",
                "baseline_x": baseline_cancel_prob,
                "y_range": (0, 1.05),
            },
            {
                "data": balk_step_class_1,
                "x_name": "class_1_step",
                "series": access_series,
                "title": "Class 1 balking step changes",
                "xlabel": "Class 1 balking high-delay probability",
                "ylabel": "served rate",
                "baseline_x": baseline_balk_step,
                "y_range": (0, 1.05),
            },
            {
                "data": balk_threshold_class_1,
                "x_name": "class_1_threshold",
                "series": access_series,
                "title": "Class 1 balking threshold changes",
                "xlabel": "Class 1 balking threshold (days)",
                "ylabel": "served rate",
                "baseline_x": baseline_balk_threshold,
                "y_range": (0, 1.05),
            },
            {
                "data": no_show_step_class_1,
                "x_name": "class_1_step",
                "series": access_series,
                "title": "Class 1 no-show step changes",
                "xlabel": "Class 1 no-show high-delay probability",
                "ylabel": "served rate",
                "baseline_x": baseline_no_show_step,
                "y_range": (0, 1.05),
            },
        ],
    )

    draw_metric_driver_figure(
        "metric_balking_rate_drivers.png",
        "Balking rate: Class 1 slices with Class 2 fixed",
        [
            {
                "data": class_1_arrival,
                "x_name": "lambda_per_day",
                "series": balking_series,
                "title": "Class 1 arrival rate changes",
                "xlabel": "Class 1 arrivals per day",
                "ylabel": "balked / offered",
                "baseline_x": baseline_class_1_arrival,
                "y_range": (0, 1.05),
            },
            {
                "data": balk_step_class_1,
                "x_name": "class_1_step",
                "series": balking_series,
                "title": "Class 1 balking step changes",
                "xlabel": "Class 1 balking high-delay probability",
                "ylabel": "balked / offered",
                "baseline_x": baseline_balk_step,
                "y_range": (0, 1.05),
            },
            {
                "data": balk_threshold_class_1,
                "x_name": "class_1_threshold",
                "series": balking_series,
                "title": "Class 1 balking threshold changes",
                "xlabel": "Class 1 balking threshold (days)",
                "ylabel": "balked / offered",
                "baseline_x": baseline_balk_threshold,
                "y_range": (0, 1.05),
            },
            {
                "data": cancel_class_1,
                "x_name": "class_1_cancel_prob",
                "series": balking_series,
                "title": "Class 1 cancellation changes",
                "xlabel": "Class 1 cancellation probability",
                "ylabel": "balked / offered",
                "baseline_x": baseline_cancel_prob,
                "y_range": (0, 1.05),
            },
        ],
    )


def add_regression_features(df):
    df = df.copy()
    for prefix in ["balk_step", "balk_threshold", "no_show_step", "no_show_threshold", "cancel_prob"]:
        c1 = df[f"class_1_{prefix}"]
        c2 = df[f"class_2_{prefix}"]
        df[f"{prefix}_mean"] = (c1 + c2) / 2.0
        df[f"{prefix}_gap_c1_minus_c2"] = c1 - c2
    return df


def sample_regression_parameters():
    rng = np.random.default_rng(REGRESSION_RANDOM_SEED)
    lambda_total_base = sum(params.lambda_per_day for params in BASE_CONFIG.classes.values())
    rows = []
    for scenario_id in range(REGRESSION_SCENARIOS):
        rows.append(
            {
                "scenario_id": scenario_id,
                "lambda_total": rng.uniform(0.4 * lambda_total_base, 1.7 * lambda_total_base),
                "class_1_share": rng.uniform(0.1, 0.9),
                "class_1_balk_step": rng.uniform(0.0, 1.0),
                "class_2_balk_step": rng.uniform(0.0, 1.0),
                "class_1_balk_threshold": rng.integers(0, BASE_CONFIG.horizon_days),
                "class_2_balk_threshold": rng.integers(0, BASE_CONFIG.horizon_days),
                "class_1_no_show_step": rng.uniform(0.0, 1.0),
                "class_2_no_show_step": rng.uniform(0.0, 1.0),
                "class_1_no_show_threshold": rng.integers(0, BASE_CONFIG.horizon_days),
                "class_2_no_show_threshold": rng.integers(0, BASE_CONFIG.horizon_days),
                "class_1_cancel_prob": rng.uniform(0.0, 0.30),
                "class_2_cancel_prob": rng.uniform(0.0, 0.30),
            }
        )
    return rows


def standardized_ols_coefficients(df, features, target):
    x = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = y.mean()
    y_std = y.std() or 1.0

    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    beta = np.linalg.lstsq(design, y_scaled, rcond=None)[0]
    y_hat = np.sum(design * beta, axis=1)
    ss_res = float(((y_scaled - y_hat) ** 2).sum())
    ss_tot = float(((y_scaled - y_scaled.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return beta[1:], r2


def train_test_r2(df, features, target):
    rng = np.random.default_rng(REGRESSION_RANDOM_SEED + 1)
    indices = rng.permutation(len(df))
    train_count = int(0.8 * len(df))
    train = df.iloc[indices[:train_count]]
    test = df.iloc[indices[train_count:]]

    x_train = train[features].to_numpy(dtype=float)
    y_train = train[target].to_numpy(dtype=float)
    x_test = test[features].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = y_train.mean()
    y_std = y_train.std() or 1.0

    train_design = np.column_stack([np.ones(len(x_train)), (x_train - x_mean) / x_std])
    beta = np.linalg.lstsq(train_design, (y_train - y_mean) / y_std, rcond=None)[0]

    test_design = np.column_stack([np.ones(len(x_test)), (x_test - x_mean) / x_std])
    y_hat = np.sum(test_design * beta, axis=1) * y_std + y_mean
    ss_res = float(((y_test - y_hat) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def plot_regression_coefficients(coef_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Standardized regression coefficients from randomized FCFS simulations", fontsize=14)

    for ax, (target, target_label) in zip(axes.ravel(), REGRESSION_TARGETS.items()):
        data = coef_df[coef_df["target"] == target].copy()
        data = data.reindex(data["coefficient"].abs().sort_values(ascending=False).index).head(8)
        data = data.sort_values("coefficient")
        colors = [
            DRIVER_COLORS.get(driver_from_text(feature), BASELINE_COLOR)
            for feature in data["feature"]
        ]
        ax.barh(data["feature_label"], data["coefficient"], color=colors)
        ax.axvline(0, color=BASELINE_COLOR, linewidth=0.9)
        ax.set_title(target_label)
        ax.set_xlabel("standardized coefficient")

    fig.savefig(OUT_DIR / "regression_standardized_coefficients.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def run_regression_screening():
    rows = []
    for params in sample_regression_parameters():
        config = set_regression_parameters(BASE_CONFIG, params)
        seed_rows = []
        for seed in REGRESSION_SEEDS:
            result = run_result(config, seed)
            seed_rows.append(
                {
                    **result_metrics_from_result(result),
                    **outcome_rates_from_result(result),
                }
            )
        metrics = pd.DataFrame(seed_rows).mean(numeric_only=True).to_dict()
        rows.append({**params, **metrics})

    data = add_regression_features(pd.DataFrame(rows))
    data.to_csv(OUT_DIR / "regression_simulation_data.csv", index=False)

    coefficient_rows = []
    score_rows = []
    for target, target_label in REGRESSION_TARGETS.items():
        coefficients, full_r2 = standardized_ols_coefficients(data, REGRESSION_FEATURES, target)
        test_r2 = train_test_r2(data, REGRESSION_FEATURES, target)
        score_rows.append(
            {
                "target": target,
                "target_label": target_label,
                "full_sample_r2": full_r2,
                "test_r2": test_r2,
            }
        )
        for feature, coefficient in zip(REGRESSION_FEATURES, coefficients):
            coefficient_rows.append(
                {
                    "target": target,
                    "target_label": target_label,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "coefficient": coefficient,
                    "abs_coefficient": abs(coefficient),
                }
            )

    coef_df = pd.DataFrame(coefficient_rows)
    score_df = pd.DataFrame(score_rows)
    coef_df.to_csv(OUT_DIR / "regression_standardized_coefficients.csv", index=False)
    score_df.to_csv(OUT_DIR / "regression_model_scores.csv", index=False)
    plot_regression_coefficients(coef_df)
    return data, coef_df, score_df


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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata():
    def run_git(args):
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
    }


def package_versions():
    versions = {}
    for package in ["numpy", "pandas", "matplotlib", "pyyaml", "jupyter"]:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def serializable_values(values):
    items = []
    for value in list(values):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            value = round(value, 10)
        items.append(value)
    return items


def generated_artifacts(run_started_at):
    artifacts = []
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and path.stat().st_mtime >= run_started_at:
            artifacts.append(path.relative_to(REPO_DIR).as_posix())
    return artifacts


def write_manifest(row_counts, run_started_at):
    manifest = {
        "command": [Path(sys.executable).name, *sys.argv],
        "git": git_metadata(),
        "config_hashes": {
            "configs/baseline.yaml": file_sha256(REPO_DIR / "configs" / "baseline.yaml"),
            "configs/scenario_2.yaml": file_sha256(REPO_DIR / "configs" / "scenario_2.yaml"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "seeds": {
            "fine_seed": FINE_SEED,
            "scenario_seeds": SCENARIO_SEEDS,
            "regression_seeds": REGRESSION_SEEDS,
            "regression_random_seed": REGRESSION_RANDOM_SEED,
        },
        "grids": {
            "step_grid": serializable_values(STEP_GRID),
            "prob_grid": serializable_values(PROB_GRID),
            "threshold_grid": serializable_values(THRESHOLD_GRID),
            "arrival_multipliers": serializable_values(ARRIVAL_MULTIPLIERS),
            "class_1_shares": serializable_values(CLASS_1_SHARES),
            "class_arrival_multipliers": serializable_values(CLASS_ARRIVAL_MULTIPLIERS),
            "fcfs_stress_multipliers": serializable_values(FCFS_STRESS_MULTIPLIERS),
            "regression_scenarios": REGRESSION_SCENARIOS,
        },
        "row_counts": row_counts,
        "generated_artifacts": generated_artifacts(run_started_at),
    }

    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    run_started_at = time.time()
    plt.style.use("default")

    scenario_df = draw_scenario_comparison()
    baseline_balk_step = BASE_CONFIG.classes[1].balk_prob.high - BASE_CONFIG.classes[1].balk_prob.low
    baseline_balk_threshold = BASE_CONFIG.classes[1].balk_prob.threshold
    baseline_no_show_step = BASE_CONFIG.classes[1].no_show_prob.high - BASE_CONFIG.classes[1].no_show_prob.low
    baseline_no_show_threshold = BASE_CONFIG.classes[1].no_show_prob.threshold
    baseline_cancel_prob = BASE_CONFIG.classes[1].cancel_prob

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
        fixed_y=baseline_balk_step,
        subtitle=f"Dashed row: class 2 fixed at baseline step = {baseline_balk_step:.2f}",
    )
    draw_behavior_panel(
        balk_step_df,
        "class_1_step",
        "class_2_step",
        "class 1 balking step",
        "class 2 balking step",
        "balking_step_behavior_panels.png",
        "Balking step by class",
        fixed_y=baseline_balk_step,
        subtitle=f"Dashed row: class 2 fixed at baseline step = {baseline_balk_step:.2f}",
    )
    draw_behavior_slice_figures(
        balk_step_df,
        "class_1_step",
        "class_2_step",
        baseline_balk_step,
        baseline_balk_step,
        "Class 1 balking step",
        "Class 2 balking step",
        "balking_step",
        "Balking step",
    )
    draw_two_metric_heatmap(
        balk_step_df,
        "class_1_step",
        "class_2_step",
        "Class 1 balking step",
        "Class 2 balking step",
        "balking_step_interaction_heatmap.png",
        "Balking step interaction",
        [
            {"metric": "overall_percent_serviced", "title": "Overall served rate", "label": "rate", "cmap": ACCESS_CMAP},
            {"metric": "access_advantage_class_1", "title": "Class gap: served rate", "label": "Class 1 - Class 2", "diverging": True},
        ],
        fixed_y=baseline_balk_step,
        fixed_x=baseline_balk_step,
    )
    draw_two_metric_heatmap(
        balk_step_df,
        "class_1_step",
        "class_2_step",
        "Class 1 balking step",
        "Class 2 balking step",
        "balking_step_balking_rate_heatmap.png",
        "Balking rate under balking-step changes",
        [
            {"metric": "overall_balking_rate", "title": "Overall balking rate", "label": "balked / offered", "cmap": BALKING_CMAP},
            {"metric": "balking_rate_gap_class_1", "title": "Class gap: balking rate", "label": "Class 1 - Class 2", "diverging": True},
        ],
        fixed_y=baseline_balk_step,
        fixed_x=baseline_balk_step,
    )
    draw_balking_class_service_heatmaps(balk_step_df, baseline_balk_step)
    balk_step_slice_df = draw_balking_slice_lines(
        balk_step_df,
        "class_1_step",
        "class_2_step",
        baseline_balk_step,
        baseline_balk_step,
        "class 1 balking step",
        f"class 2 balking step fixed at baseline = {baseline_balk_step:.2f}",
        "balking_class1_step_slice.png",
        "Balking 1D slice: class 1 step varies",
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
        fixed_y=baseline_balk_threshold,
        subtitle=f"Dashed row: class 2 fixed at baseline threshold = {baseline_balk_threshold}",
    )
    draw_behavior_panel(
        balk_threshold_df,
        "class_1_threshold",
        "class_2_threshold",
        "class 1 balking threshold",
        "class 2 balking threshold",
        "balking_threshold_behavior_panels.png",
        "Balking threshold by class",
        fixed_y=baseline_balk_threshold,
        subtitle=f"Dashed row: class 2 fixed at baseline threshold = {baseline_balk_threshold}",
    )
    draw_behavior_slice_figures(
        balk_threshold_df,
        "class_1_threshold",
        "class_2_threshold",
        baseline_balk_threshold,
        baseline_balk_threshold,
        "Class 1 balking threshold",
        "Class 2 balking threshold",
        "balking_threshold",
        "Balking threshold",
    )
    draw_two_metric_heatmap(
        balk_threshold_df,
        "class_1_threshold",
        "class_2_threshold",
        "Class 1 balking threshold",
        "Class 2 balking threshold",
        "balking_threshold_interaction_heatmap.png",
        "Balking threshold interaction",
        [
            {"metric": "mean_offered_booking_delay", "title": "Overall offered wait", "label": "days", "cmap": WAIT_CMAP},
            {"metric": "access_advantage_class_1", "title": "Class gap: served rate", "label": "Class 1 - Class 2", "diverging": True},
        ],
        fixed_y=baseline_balk_threshold,
        fixed_x=baseline_balk_threshold,
    )
    draw_two_metric_heatmap(
        balk_threshold_df,
        "class_1_threshold",
        "class_2_threshold",
        "Class 1 balking threshold",
        "Class 2 balking threshold",
        "balking_threshold_balking_rate_heatmap.png",
        "Balking rate under balking-threshold changes",
        [
            {"metric": "overall_balking_rate", "title": "Overall balking rate", "label": "balked / offered", "cmap": BALKING_CMAP},
            {"metric": "balking_rate_gap_class_1", "title": "Class gap: balking rate", "label": "Class 1 - Class 2", "diverging": True},
        ],
        fixed_y=baseline_balk_threshold,
        fixed_x=baseline_balk_threshold,
    )
    draw_balking_threshold_class_service_heatmaps(balk_threshold_df, baseline_balk_threshold)
    balk_threshold_slice_df = draw_balking_slice_lines(
        balk_threshold_df,
        "class_1_threshold",
        "class_2_threshold",
        baseline_balk_threshold,
        baseline_balk_threshold,
        "class 1 balking threshold",
        f"class 2 balking threshold fixed at baseline = {baseline_balk_threshold}",
        "balking_class1_threshold_slice.png",
        "Balking 1D slice: class 1 threshold varies",
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
    draw_behavior_panel(
        no_show_step_df,
        "class_1_step",
        "class_2_step",
        "class 1 no-show step",
        "class 2 no-show step",
        "no_show_step_behavior_panels.png",
        "No-show step by class",
    )
    draw_behavior_slice_figures(
        no_show_step_df,
        "class_1_step",
        "class_2_step",
        baseline_no_show_step,
        baseline_no_show_step,
        "Class 1 no-show step",
        "Class 2 no-show step",
        "no_show_step",
        "No-show step",
    )
    draw_two_metric_heatmap(
        no_show_step_df,
        "class_1_step",
        "class_2_step",
        "Class 1 no-show step",
        "Class 2 no-show step",
        "no_show_step_interaction_heatmap.png",
        "No-show step interaction",
        [
            {"metric": "average_utilization", "title": "Utilization", "label": "rate", "cmap": UTILIZATION_CMAP},
            {"metric": "access_advantage_class_1", "title": "Class gap: served rate", "label": "Class 1 - Class 2", "diverging": True},
        ],
        fixed_y=baseline_no_show_step,
        fixed_x=baseline_no_show_step,
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
    draw_behavior_panel(
        no_show_threshold_df,
        "class_1_threshold",
        "class_2_threshold",
        "class 1 no-show threshold",
        "class 2 no-show threshold",
        "no_show_threshold_behavior_panels.png",
        "No-show threshold by class",
    )
    draw_behavior_slice_figures(
        no_show_threshold_df,
        "class_1_threshold",
        "class_2_threshold",
        baseline_no_show_threshold,
        baseline_no_show_threshold,
        "Class 1 no-show threshold",
        "Class 2 no-show threshold",
        "no_show_threshold",
        "No-show threshold",
    )
    draw_two_metric_heatmap(
        no_show_threshold_df,
        "class_1_threshold",
        "class_2_threshold",
        "Class 1 no-show threshold",
        "Class 2 no-show threshold",
        "no_show_threshold_interaction_heatmap.png",
        "No-show threshold interaction",
        [
            {"metric": "average_utilization", "title": "Utilization", "label": "rate", "cmap": UTILIZATION_CMAP},
            {"metric": "access_advantage_class_1", "title": "Class gap: served rate", "label": "Class 1 - Class 2", "diverging": True},
        ],
        fixed_y=baseline_no_show_threshold,
        fixed_x=baseline_no_show_threshold,
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
    draw_behavior_panel(
        cancel_df,
        "class_1_cancel_prob",
        "class_2_cancel_prob",
        "class 1 cancellation probability",
        "class 2 cancellation probability",
        "cancellation_behavior_panels.png",
        "Cancellation probability by class",
    )
    draw_behavior_slice_figures(
        cancel_df,
        "class_1_cancel_prob",
        "class_2_cancel_prob",
        baseline_cancel_prob,
        baseline_cancel_prob,
        "Class 1 cancellation probability",
        "Class 2 cancellation probability",
        "cancellation_probability",
        "Cancellation probability",
    )
    draw_two_metric_heatmap(
        cancel_df,
        "class_1_cancel_prob",
        "class_2_cancel_prob",
        "Class 1 cancellation probability",
        "Class 2 cancellation probability",
        "cancellation_probability_interaction_heatmap.png",
        "Cancellation probability interaction",
        [
            {"metric": "overall_percent_serviced", "title": "Overall served rate", "label": "rate", "cmap": ACCESS_CMAP},
            {"metric": "access_advantage_class_1", "title": "Class gap: served rate", "label": "Class 1 - Class 2", "diverging": True},
        ],
        fixed_y=baseline_cancel_prob,
        fixed_x=baseline_cancel_prob,
    )

    arrival_df = draw_arrival_mix()
    class_arrival_df = draw_class_arrival_lines()
    fcfs_stress_df = draw_fcfs_capacity_stress()
    draw_metric_driver_figures(
        class_arrival_df,
        balk_step_df,
        balk_threshold_df,
        no_show_step_df,
        no_show_threshold_df,
        cancel_df,
        baseline_balk_step,
        baseline_balk_threshold,
        baseline_no_show_step,
        baseline_no_show_threshold,
        baseline_cancel_prob,
    )
    regression_data, regression_coef_df, regression_score_df = run_regression_screening()
    write_manifest(
        {
            "scenario_comparison": len(scenario_df),
            "balking_step_grid": len(balk_step_df),
            "balking_threshold_grid": len(balk_threshold_df),
            "balking_threshold_jump_grid": len(balk_threshold_jump_df),
            "no_show_step_grid": len(no_show_step_df),
            "no_show_threshold_grid": len(no_show_threshold_df),
            "no_show_threshold_jump_grid": len(no_show_threshold_jump_df),
            "cancellation_grid": len(cancel_df),
            "arrival_mix_grid": len(arrival_df),
            "class_arrival_grid": len(class_arrival_df),
            "fcfs_capacity_stress": len(fcfs_stress_df),
            "balking_class1_step_slice": len(balk_step_slice_df),
            "balking_class1_threshold_slice": len(balk_threshold_slice_df),
            "regression_simulation_data": len(regression_data),
            "regression_coefficients": len(regression_coef_df),
            "regression_scores": len(regression_score_df),
        },
        run_started_at,
    )

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

    print("\nFCFS capacity stress selected points")
    selected = fcfs_stress_df[fcfs_stress_df["arrival_multiplier"].isin([0.3, 1.0, 1.7])]
    print(
        selected[
            [
                "arrival_multiplier",
                "average_utilization",
                "overall_percent_serviced",
                "mean_offered_booking_delay",
                "served_rate",
                "balked_rate",
                "no_offer_rate",
                "lost_after_booking_rate",
                "access_advantage_class_1",
            ]
        ].to_string(index=False)
    )

    print("\nBalking 1D slices")
    for name, df in [
        ("Class 1 step varies, class 2 fixed at baseline", balk_step_slice_df),
        ("Class 1 threshold varies, class 2 fixed at baseline", balk_threshold_slice_df),
    ]:
        print(name)
        print(
            df[
                [
                    df.columns[0],
                    "average_utilization",
                    "overall_percent_serviced",
                    "mean_offered_booking_delay",
                    "access_advantage_class_1",
                ]
            ].to_string(index=False)
        )

    print("\nRegression screening model scores")
    print(regression_score_df.to_string(index=False))
    print("\nTop regression coefficients by target")
    for target, target_label in REGRESSION_TARGETS.items():
        top = regression_coef_df[regression_coef_df["target"] == target].nlargest(5, "abs_coefficient")
        print(f"\n{target_label}")
        print(top[["feature_label", "coefficient"]].to_string(index=False))


if __name__ == "__main__":
    main()
