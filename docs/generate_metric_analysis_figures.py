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


def aggregate_delay_metrics(result):
    booked = sum(m.booked for m in result.class_metrics.values())
    offered = sum(m.offered for m in result.class_metrics.values())
    accepted_delay = sum(m.total_booking_delay for m in result.class_metrics.values())
    offered_delay = sum(m.total_offered_booking_delay for m in result.class_metrics.values())
    return {
        "mean_accepted_booking_delay": accepted_delay / booked if booked else 0.0,
        "mean_offered_booking_delay": offered_delay / offered if offered else 0.0,
    }


def result_metrics_from_result(result):
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


def result_metrics(config, seed=FINE_SEED):
    return result_metrics_from_result(run_result(config, seed))


def outcome_rates_from_result(result):
    totals = {
        "arrivals": 0,
        "booked": 0,
        "balked": 0,
        "no_offer": 0,
        "canceled": 0,
        "no_show": 0,
        "served": 0,
    }
    for metrics in result.class_metrics.values():
        for key in totals:
            totals[key] += getattr(metrics, key)

    resolved_booked = totals["canceled"] + totals["no_show"] + totals["served"]
    totals["unresolved_booked"] = max(totals["booked"] - resolved_booked, 0)
    arrivals = totals["arrivals"]

    rates = {
        "total_arrivals": arrivals,
        "total_booked": totals["booked"],
    }
    for outcome in ["served", "balked", "no_offer", "canceled", "no_show", "unresolved_booked"]:
        rates[f"{outcome}_rate"] = totals[outcome] / arrivals if arrivals else 0.0
    rates["lost_after_booking_rate"] = (
        (totals["canceled"] + totals["no_show"] + totals["unresolved_booked"]) / arrivals if arrivals else 0.0
    )
    return rates


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


def draw_four_panel(df, x_name, y_name, xlabel, ylabel, filename, title, fixed_y=None, fixed_x=None, subtitle=None):
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
        image = heatmap_panel(ax, table, panel_title, xlabel, ylabel, diverging=diverging)
        mark_heatmap_slice(ax, table, fixed_y=fixed_y, fixed_x=fixed_x)
        fig.colorbar(image, ax=ax, shrink=0.85)

    fig.savefig(OUT_DIR / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)


def draw_balking_slice_lines(df, x_name, y_name, fixed_y, baseline_x, xlabel, fixed_label, filename, title):
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

    for ax, (metric, panel_title, ylabel) in zip(axes.ravel(), specs):
        ax.plot(slice_df[x_name], slice_df[metric], marker="o", linewidth=2)
        ax.axvline(baseline_x, color="black", linestyle="--", linewidth=1.1, label="class 1 baseline")
        if "advantage" in metric:
            ax.axhline(0, color="gray", linewidth=0.9)
        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

    fig.savefig(OUT_DIR / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return slice_df


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
    ax.plot(df.arrival_multiplier, df.average_utilization, marker="o", label="average utilization")
    ax.plot(df.arrival_multiplier, df.overall_percent_serviced, marker="o", label="overall percent serviced")
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Capacity and access")
    ax.set_xlabel("arrival-rate multiplier")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(df.arrival_multiplier, df.mean_accepted_booking_delay, marker="o", label="accepted delay")
    ax.plot(df.arrival_multiplier, df.mean_offered_booking_delay, marker="o", label="offered delay")
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Booking delay")
    ax.set_xlabel("arrival-rate multiplier")
    ax.set_ylabel("days")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(df.arrival_multiplier, df.class_1_percent_serviced, marker="o", label="class 1")
    ax.plot(df.arrival_multiplier, df.class_2_percent_serviced, marker="o", label="class 2")
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Class access under symmetric FCFS")
    ax.set_xlabel("arrival-rate multiplier")
    ax.set_ylabel("percent serviced")
    ax.set_ylim(0, 1.05)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(df.arrival_multiplier, df.balked_rate, marker="o", label="balked")
    ax.plot(df.arrival_multiplier, df.no_offer_rate, marker="o", label="no offer")
    ax.plot(df.arrival_multiplier, df.lost_after_booking_rate, marker="o", label="lost after booking")
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Main loss channels")
    ax.set_xlabel("arrival-rate multiplier")
    ax.set_ylabel("share of arrivals")
    ax.set_ylim(0, 1.05)
    ax.legend()

    fig.savefig(OUT_DIR / "fcfs_capacity_stress_curves.png", dpi=190, bbox_inches="tight")
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
        df.arrival_multiplier,
        *[df[column] for column, _ in outcomes],
        labels=[label for _, label in outcomes],
        alpha=0.9,
    )
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Final outcome decomposition per arrival")
    ax.set_xlabel("arrival-rate multiplier")
    ax.set_ylabel("share of arrivals")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(OUT_DIR / "fcfs_arrival_outcome_decomposition.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    return df


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
        colors = ["#2f6fbb" if value > 0 else "#bb4a4a" for value in data["coefficient"]]
        ax.barh(data["feature_label"], data["coefficient"], color=colors)
        ax.axvline(0, color="black", linewidth=0.9)
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


def main():
    plt.style.use("default")

    scenario_df = draw_scenario_comparison()
    baseline_balk_step = BASE_CONFIG.classes[1].balk_prob.high - BASE_CONFIG.classes[1].balk_prob.low
    baseline_balk_threshold = BASE_CONFIG.classes[1].balk_prob.threshold

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
    fcfs_stress_df = draw_fcfs_capacity_stress()
    regression_data, regression_coef_df, regression_score_df = run_regression_screening()

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
