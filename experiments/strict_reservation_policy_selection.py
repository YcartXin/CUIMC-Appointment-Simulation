"""Strict Class 1 reservation policy-selection experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from analysis.reservation_policy_metrics import score_simulation_row
from analysis.reservation_policy_selection import (
    OBJECTIVE_SPECS,
    SCENARIO_COLUMNS,
    build_selection_summaries,
    pair_with_fcfs,
    summarize_by_q,
)
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import PatientClassParams, SimulationConfig, ThresholdRule


DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "strict_reservation_policy_selection"
DEFAULT_REPORT_DIR = (
    REPO_DIR / "docs" / "reports" / "reservation" / "policy_selection"
)

SLOTS_PER_DAY = 32
HORIZON_DAYS = 14
BURN_IN_DAYS = 30
MEASURE_DAYS = 365
COOLDOWN_DAYS = 14
CANCEL_PROB = 0.10
BALK_LOW = 0.0
NO_SHOW_THRESHOLD = 6
NO_SHOW_LOW = 0.0
NO_SHOW_HIGH = 0.30

THRESHOLD_PAIRS = ((9, 9), (5, 9), (9, 5), (12, 12), (5, 12))
BALK_HIGH_VALUES = (0.3, 0.5, 0.7)
ARRIVAL_RATES = (25, 50)
Q_VALUES = (0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32)
STANDARD_SEEDS = tuple(range(61001, 61021))
WEIGHTS = ((0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (2.0, 1.0))
# Weights only affect post-processing. Keep the historical task identity stable
# so existing simulation shards remain reusable when report weights change.
SIMULATION_TASK_ID_WEIGHTS = ((1.0, 1.0), (2.0, 1.0))
NEAR_TIE_TOLERANCE = 0.01
SHARD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Profile:
    name: str
    threshold_pairs: tuple[tuple[int, int], ...]
    balk_high_values: tuple[float, ...]
    arrival_rates: tuple[int, ...]
    q_values: tuple[int, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class Task:
    profile: str
    scenario_id: str
    arrival_rate_class_1: int
    arrival_rate_class_2: int
    tau_1: int
    tau_2: int
    post_threshold_balking_rate: float
    Q: int
    seed: int
    task_id: str


def profile_grid(name: str) -> Profile:
    if name == "standard":
        return Profile(
            name=name,
            threshold_pairs=THRESHOLD_PAIRS,
            balk_high_values=BALK_HIGH_VALUES,
            arrival_rates=ARRIVAL_RATES,
            q_values=Q_VALUES,
            seeds=STANDARD_SEEDS,
        )
    if name == "smoke":
        return Profile(
            name=name,
            threshold_pairs=((9, 9), (5, 12)),
            balk_high_values=(0.5,),
            arrival_rates=ARRIVAL_RATES,
            q_values=(0, 8, 32),
            seeds=(61001, 61002),
        )
    raise ValueError(f"Unknown profile: {name}")


def expected_cardinality(profile: str | Profile) -> int:
    grid = profile_grid(profile) if isinstance(profile, str) else profile
    return (
        len(grid.threshold_pairs)
        * len(grid.balk_high_values)
        * len(grid.arrival_rates)
        * len(grid.q_values)
        * len(grid.seeds)
    )


def source_fingerprint() -> str:
    files = (
        REPO_DIR / "simulation" / "engine.py",
        REPO_DIR / "simulation" / "model.py",
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def experiment_manifest(
    profile: Profile,
    *,
    include_report_weights: bool = True,
) -> dict[str, Any]:
    manifest = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint(),
        "profile": asdict(profile),
        "weights": SIMULATION_TASK_ID_WEIGHTS,
        "slots_per_day": SLOTS_PER_DAY,
        "horizon_days": HORIZON_DAYS,
        "burn_in_days": BURN_IN_DAYS,
        "measure_days": MEASURE_DAYS,
        "cooldown_days": COOLDOWN_DAYS,
        "cancel_prob": CANCEL_PROB,
        "balk_low": BALK_LOW,
        "no_show_rule": {
            "threshold": NO_SHOW_THRESHOLD,
            "low": NO_SHOW_LOW,
            "high": NO_SHOW_HIGH,
        },
        "near_tie_tolerance": NEAR_TIE_TOLERANCE,
    }
    if include_report_weights:
        manifest["report_weights"] = WEIGHTS
    return json.loads(json.dumps(manifest, sort_keys=True))


def task_identity_manifest(profile: Profile) -> dict[str, Any]:
    return experiment_manifest(profile, include_report_weights=False)


def manifest_task_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(manifest)
    identity.pop("report_weights", None)
    return identity


def build_tasks(profile: str | Profile) -> list[Task]:
    grid = profile_grid(profile) if isinstance(profile, str) else profile
    manifest_hash = hashlib.sha256(
        json.dumps(
            task_identity_manifest(grid),
            sort_keys=True,
            default=list,
        ).encode("utf-8")
    ).hexdigest()[:10]
    tasks = []
    for tau_1, tau_2 in grid.threshold_pairs:
        for balk_high in grid.balk_high_values:
            for arrival_rate in grid.arrival_rates:
                scenario_id = (
                    f"t{tau_1:02d}_{tau_2:02d}_"
                    f"b{int(round(balk_high * 10)):02d}_l{arrival_rate:02d}"
                )
                for q in grid.q_values:
                    for seed in grid.seeds:
                        task_id = (
                            f"{grid.name}__{scenario_id}__q{q:02d}"
                            f"__seed{seed}__{manifest_hash}"
                        )
                        tasks.append(
                            Task(
                                profile=grid.name,
                                scenario_id=scenario_id,
                                arrival_rate_class_1=arrival_rate,
                                arrival_rate_class_2=arrival_rate,
                                tau_1=tau_1,
                                tau_2=tau_2,
                                post_threshold_balking_rate=balk_high,
                                Q=q,
                                seed=seed,
                                task_id=task_id,
                            )
                        )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise AssertionError("Task identifiers are not unique.")
    return tasks


def build_config(task: Task) -> SimulationConfig:
    classes = {
        1: PatientClassParams(
            class_id=1,
            lambda_per_day=task.arrival_rate_class_1,
            balk_prob=ThresholdRule(
                task.tau_1,
                BALK_LOW,
                task.post_threshold_balking_rate,
            ),
            cancel_prob=CANCEL_PROB,
            no_show_prob=ThresholdRule(
                NO_SHOW_THRESHOLD,
                NO_SHOW_LOW,
                NO_SHOW_HIGH,
            ),
        ),
        2: PatientClassParams(
            class_id=2,
            lambda_per_day=task.arrival_rate_class_2,
            balk_prob=ThresholdRule(
                task.tau_2,
                BALK_LOW,
                task.post_threshold_balking_rate,
            ),
            cancel_prob=CANCEL_PROB,
            no_show_prob=ThresholdRule(
                NO_SHOW_THRESHOLD,
                NO_SHOW_LOW,
                NO_SHOW_HIGH,
            ),
        ),
    }
    return SimulationConfig(
        slots_per_day=SLOTS_PER_DAY,
        horizon_days=HORIZON_DAYS,
        burn_in_days=BURN_IN_DAYS,
        measure_days=MEASURE_DAYS,
        cooldown_days=COOLDOWN_DAYS,
        classes=classes,
        seed=task.seed,
        reserved_class_id=1 if task.Q > 0 else None,
        reserved_slots_per_day=task.Q,
    )


def run_task(task: Task) -> dict[str, Any]:
    result = ClinicAppointmentSimulation(build_config(task)).run()
    c1 = result.class_metrics[1]
    c2 = result.class_metrics[2]
    unresolved_1 = c1.booked - c1.canceled - c1.no_show - c1.served
    unresolved_2 = c2.booked - c2.canceled - c2.no_show - c2.served
    accounting_ok = all(
        (
            c1.arrivals == c1.offered + c1.no_offer,
            c2.arrivals == c2.offered + c2.no_offer,
            unresolved_1 == 0,
            unresolved_2 == 0,
            result.slot_metrics.booked_slots
            == result.slot_metrics.served_slots + result.slot_metrics.no_show_slots,
        )
    )
    q32_exclusion_ok = (
        task.Q != SLOTS_PER_DAY
        or (
            c2.offered == 0
            and c2.served == 0
            and c2.no_offer == c2.arrivals
        )
    )
    return {
        "schema_version": SHARD_SCHEMA_VERSION,
        **asdict(task),
        "S": result.total_slots,
        "A1": c1.arrivals,
        "A2": c2.arrivals,
        "Y1": c1.served,
        "Y2": c2.served,
        "offered_1": c1.offered,
        "offered_2": c2.offered,
        "sum_tau_offered_1": c1.total_offered_booking_delay,
        "sum_tau_offered_2": c2.total_offered_booking_delay,
        "no_offer_1": c1.no_offer,
        "no_offer_2": c2.no_offer,
        "booked_1": c1.booked,
        "booked_2": c2.booked,
        "balked_1": c1.balked,
        "balked_2": c2.balked,
        "canceled_1": c1.canceled,
        "canceled_2": c2.canceled,
        "no_show_1": c1.no_show,
        "no_show_2": c2.no_show,
        "unresolved_1": unresolved_1,
        "unresolved_2": unresolved_2,
        "accounting_ok": accounting_ok,
        "q32_exclusion_ok": q32_exclusion_ok,
    }


def shard_path(output_dir: Path, task: Task) -> Path:
    return output_dir / task.profile / "shards" / f"{task.task_id}.csv.gz"


def atomic_write_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_row(path: Path) -> dict[str, str]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Invalid shard row count: {path}")
    return rows[0]


def valid_completed_ids(output_dir: Path, tasks: Sequence[Task]) -> set[str]:
    completed = set()
    for task in tasks:
        path = shard_path(output_dir, task)
        if not path.exists():
            continue
        try:
            row = read_row(path)
        except (OSError, EOFError, csv.Error, ValueError):
            continue
        if (
            row.get("task_id") == task.task_id
            and row.get("schema_version") == str(SHARD_SCHEMA_VERSION)
        ):
            completed.add(task.task_id)
    return completed


def _run_and_write(task: Task, output_dir: str) -> str:
    row = run_task(task)
    atomic_write_row(shard_path(Path(output_dir), task), row)
    return task.task_id


def load_simulation_rows(output_dir: Path, tasks: Sequence[Task]) -> pd.DataFrame:
    rows = [read_row(shard_path(output_dir, task)) for task in tasks]
    frame = pd.DataFrame(rows)
    text_columns = {"profile", "scenario_id", "task_id"}
    bool_columns = {"accounting_ok", "q32_exclusion_ok"}
    for column in frame:
        if column in text_columns:
            continue
        if column in bool_columns:
            frame[column] = frame[column].str.lower().eq("true")
        else:
            frame[column] = pd.to_numeric(frame[column])
    return frame


def score_rows(simulation_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in simulation_rows.to_dict("records"):
        for w1, w2 in WEIGHTS:
            rows.append(score_simulation_row(record, w1=w1, w2=w2))
    scored = pair_with_fcfs(pd.DataFrame(rows))
    if scored.filter(regex=r"^fcfs_").isna().all(axis=1).any():
        raise AssertionError("Missing FCFS values after pairing.")
    return scored


def atomic_write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        frame.to_csv(
            temporary,
            index=False,
            compression="gzip" if path.suffix == ".gz" else None,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def consolidate(
    *,
    profile: Profile,
    output_dir: Path,
    tasks: Sequence[Task],
) -> dict[str, Path]:
    profile_dir = output_dir / profile.name
    simulation_rows = load_simulation_rows(output_dir, tasks)
    run_level = score_rows(simulation_rows)
    q_summary = summarize_by_q(run_level)
    scenario_summary, best_q, near_ranges = build_selection_summaries(
        q_summary,
        tested_q_values=profile.q_values,
        relative_tolerance=NEAR_TIE_TOLERANCE,
    )
    paths = {
        "simulation": profile_dir / "simulation_results.csv.gz",
        "run_level": profile_dir / "run_level_results.csv.gz",
        "q_summary": profile_dir / "q_level_summary.csv",
        "scenario_summary": profile_dir / "scenario_level_summary.csv",
        "best_q": profile_dir / "best_q_summary.csv",
        "near_ranges": profile_dir / "near_tie_q_ranges.csv",
    }
    atomic_write_dataframe(simulation_rows, paths["simulation"])
    atomic_write_dataframe(run_level, paths["run_level"])
    atomic_write_dataframe(q_summary, paths["q_summary"])
    atomic_write_dataframe(scenario_summary, paths["scenario_summary"])
    atomic_write_dataframe(best_q, paths["best_q"])
    atomic_write_dataframe(near_ranges, paths["near_ranges"])
    return paths


def _behavior_label(row: pd.Series) -> str:
    return (
        f"C1={int(row['tau_1'])}d, C2={int(row['tau_2'])}d; "
        f"high={row['post_threshold_balking_rate']:.1f}"
    )


def _context_label(row: pd.Series) -> str:
    return (
        f"{int(row['arrival_rate_class_1'])}/class, "
        f"w1={row['w1']:g}"
    )


def plot_best_q_heatmap(
    best_q: pd.DataFrame,
    *,
    objective: str,
    path: Path,
    title: str,
) -> None:
    data = best_q[best_q["objective_name"] == objective].copy()
    data["behavior"] = data.apply(_behavior_label, axis=1)
    data["context"] = data.apply(_context_label, axis=1)
    behaviors = list(dict.fromkeys(data["behavior"]))
    context_frame = (
        data[["arrival_rate_class_1", "w1", "context"]]
        .drop_duplicates()
        .sort_values(["arrival_rate_class_1", "w1"])
    )
    contexts = context_frame["context"].tolist()
    matrix = np.full((len(behaviors), len(contexts)), np.nan)
    labels = [["" for _ in contexts] for _ in behaviors]
    for _, row in data.iterrows():
        i = behaviors.index(row["behavior"])
        j = contexts.index(row["context"])
        q_values = [int(q) for q in str(row["near_tie_q_values"]).split(",")]
        matrix[i, j] = np.mean(q_values)
        labels[i][j] = str(row["near_tie_q_ranges"]).split(" ", 1)[0]

    fig, ax = plt.subplots(figsize=(14, 7))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=32, aspect="auto")
    arrivals = context_frame["arrival_rate_class_1"].tolist()
    for index in range(1, len(arrivals)):
        if arrivals[index] != arrivals[index - 1]:
            ax.axvline(index - 0.5, color="black", linewidth=1.4)
    ax.set_xticks(range(len(contexts)), contexts, rotation=25, ha="right")
    ax.set_yticks(range(len(behaviors)), behaviors)
    for i in range(len(behaviors)):
        for j in range(len(contexts)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] <= 14 else "black")
    ax.set_title(title)
    ax.set_xlabel("Arrival and weight regime")
    ax.set_ylabel("Behavior regime")
    fig.colorbar(image, ax=ax, label="Mean Q in 1% near-tie set")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _weighted_objective_deltas_for_plot(
    run_level: pd.DataFrame,
    *,
    w1_values: Sequence[float],
) -> pd.DataFrame:
    base = run_level[
        (run_level["tau_1"] == 9)
        & (run_level["tau_2"] == 9)
        & np.isclose(run_level["post_threshold_balking_rate"], 0.5)
        & np.isclose(run_level["w1"], 1.0)
        & np.isclose(run_level["w2"], 1.0)
    ].copy()
    rows = []
    for w1 in w1_values:
        w2 = 1.0
        denominator = w1 + w2
        util = (w1 * base["Y1"] / base["S"] + w2 * base["Y2"] / base["S"]) / denominator
        fcfs_util = (
            w1 * base["fcfs_Y1"] / base["S"]
            + w2 * base["fcfs_Y2"] / base["S"]
        ) / denominator
        service = (
            w1 * _safe_ratio(base["Y1"], base["A1"])
            + w2 * _safe_ratio(base["Y2"], base["A2"])
        ) / denominator
        fcfs_service = (
            w1 * _safe_ratio(base["fcfs_Y1"], base["fcfs_A1"])
            + w2 * _safe_ratio(base["fcfs_Y2"], base["fcfs_A2"])
        ) / denominator
        scored = base[
            ["arrival_rate_class_1", "arrival_rate_class_2", "Q", "seed"]
        ].copy()
        scored["w1_plot"] = w1
        scored["delta_Obj_util_norm"] = util - fcfs_util
        scored["delta_Obj_service_norm"] = service - fcfs_service
        rows.append(scored)
    combined = pd.concat(rows, ignore_index=True)
    return (
        combined.groupby(["arrival_rate_class_1", "w1_plot", "Q"], as_index=False)
        [["delta_Obj_util_norm", "delta_Obj_service_norm"]]
        .mean()
    )


def plot_representative_deltas(run_level: pd.DataFrame, path: Path) -> None:
    data = _weighted_objective_deltas_for_plot(
        run_level,
        w1_values=(0.5, 1.0, 1.5, 2.0),
    )
    colors = {
        0.5: "#6a51a3",
        1.0: "#2b8cbe",
        1.5: "#31a354",
        2.0: "#e6550d",
    }
    markers = {25: "o", 50: "s"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, metric, title in (
        (axes[0], "delta_Obj_util_norm", "Normalized weighted slot utilization"),
        (axes[1], "delta_Obj_service_norm", "Normalized weighted service rate"),
    ):
        for (arrival, w1), group in data.groupby(
            ["arrival_rate_class_1", "w1_plot"],
            sort=True,
        ):
            group = group.sort_values("Q")
            ax.plot(
                group["Q"],
                group[metric],
                marker=markers[int(arrival)],
                color=colors[float(w1)],
                linewidth=1.5,
                markersize=4.5,
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel("Reserved slots Q")
        ax.set_ylabel("Mean delta vs FCFS")
        ax.grid(alpha=0.2)
    weight_handles = [
        Line2D([0], [0], color=color, linewidth=1.8, label=f"w1={w1:g}")
        for w1, color in colors.items()
    ]
    arrival_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="0.35",
            linestyle="None",
            label=f"{arrival}/class",
        )
        for arrival, marker in markers.items()
    ]
    axes[0].legend(handles=weight_handles, title="Class 1 weight", frameon=False, fontsize=8)
    axes[1].legend(handles=arrival_handles, title="Arrival rate", frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_tradeoffs(q_summary: pd.DataFrame, path: Path) -> None:
    data = q_summary[
        (q_summary["tau_1"] == 9)
        & (q_summary["tau_2"] == 9)
        & np.isclose(q_summary["post_threshold_balking_rate"], 0.5)
        & np.isclose(q_summary["w1"], 1)
    ]
    best_data = q_summary[
        (q_summary["tau_1"] == 9)
        & (q_summary["tau_2"] == 9)
        & np.isclose(q_summary["post_threshold_balking_rate"], 0.5)
        & np.isclose(q_summary["w1"], 2)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (arrival, group) in zip(
        axes,
        data.groupby("arrival_rate_class_1", sort=True),
    ):
        group = group.sort_values("Q")
        ax.plot(
            group["class_1_service_rate_mean"],
            group["class_2_service_rate_mean"],
            marker="o",
            color="#3b667f",
            label="Q sweep",
        )
        best_group = best_data[best_data["arrival_rate_class_1"] == arrival]
        best_row = best_group.loc[best_group["Obj_service_norm_mean"].idxmax()]
        x_best = float(best_row["class_1_service_rate_mean"])
        y_best = float(best_row["class_2_service_rate_mean"])
        target_raw = 2.0 * x_best + y_best
        xs = np.linspace(
            max(0.0, float(group["class_1_service_rate_mean"].min()) - 0.02),
            min(1.0, float(group["class_1_service_rate_mean"].max()) + 0.02),
            100,
        )
        ys = target_raw - 2.0 * xs
        mask = (ys >= 0) & (ys <= 1)
        ax.plot(
            xs[mask],
            ys[mask],
            linestyle="--",
            linewidth=1.0,
            color="0.35",
            label="same service objective, w1=2",
        )
        ax.scatter(
            [x_best],
            [y_best],
            marker="*",
            s=150,
            color="#d62728",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label=f"highest service objective: Q={int(best_row['Q'])}",
        )
        other_objectives = [
            (
                "Obj_util_norm_mean",
                "max",
                "highest utilization objective",
                "D",
                "#31a354",
            ),
            (
                "T_wait_offered_mean",
                "min",
                "lowest offered-wait objective",
                "X",
                "#9467bd",
            ),
        ]
        for metric, direction, label, marker, color in other_objectives:
            if direction == "max":
                row = best_group.loc[best_group[metric].idxmax()]
            else:
                row = best_group.loc[best_group[metric].idxmin()]
            ax.scatter(
                [float(row["class_1_service_rate_mean"])],
                [float(row["class_2_service_rate_mean"])],
                marker=marker,
                s=95,
                color=color,
                alpha=0.42,
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
                label=f"{label}: Q={int(row['Q'])}",
            )
        for _, row in group.iterrows():
            ax.annotate(
                str(int(row["Q"])),
                (row["class_1_service_rate_mean"], row["class_2_service_rate_mean"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_title(f"{int(arrival)} arrivals per class")
        ax.set_xlabel("Class 1 service rate")
        ax.set_ylabel("Class 2 service rate")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Service redistribution as Q increases")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_wait_no_offer(q_summary: pd.DataFrame, path: Path) -> None:
    data = q_summary[
        (q_summary["tau_1"] == 9)
        & (q_summary["tau_2"] == 9)
        & np.isclose(q_summary["post_threshold_balking_rate"], 0.5)
        & np.isclose(q_summary["w1"], 2)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for column, (arrival, group) in enumerate(
        data.groupby("arrival_rate_class_1", sort=True)
    ):
        group = group.sort_values("Q")
        axes[0, column].plot(
            group["Q"],
            group["T_wait_offered_mean"],
            marker="o",
            color="0.2",
            label="Weighted offered wait",
        )
        axes[0, column].plot(
            group["Q"],
            group["class_1_avg_offered_wait_mean"],
            marker="s",
            color="#1f77b4",
            label="Class 1",
        )
        axes[0, column].plot(
            group["Q"],
            group["class_2_avg_offered_wait_mean"],
            marker="^",
            color="#ff7f0e",
            label="Class 2",
        )
        axes[1, column].plot(
            group["Q"],
            group["class_1_no_offer_rate_mean"],
            marker="s",
            color="#1f77b4",
            label="Class 1",
        )
        axes[1, column].plot(
            group["Q"],
            group["class_2_no_offer_rate_mean"],
            marker="^",
            color="#ff7f0e",
            label="Class 2",
        )
        axes[0, column].set_title(f"{int(arrival)} arrivals per class")
        axes[0, column].set_ylabel("Offered wait (days)")
        axes[1, column].set_ylabel("No-offer rate")
        axes[1, column].set_xlabel("Reserved slots Q")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
        axes[0, column].annotate(
            "Q=32: Class 2 receives no offers",
            xy=(32, group.loc[group["Q"] == 32, "T_wait_offered_mean"].iloc[0]),
            xytext=(-8, 28),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            arrowprops={"arrowstyle": "->", "linewidth": 0.7},
        )
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Offered wait must be read with no-offer rates "
        "(thresholds 9,9; balk-high 0.5; w1=2)"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(str(value).replace("|", "\\|") for value in row)
            + " |"
        )
    return "\n".join(lines)


def _arrival_utilization_table(
    scenario_summary: pd.DataFrame,
    *,
    arrival_rate: int,
) -> pd.DataFrame:
    data = scenario_summary[
        scenario_summary["arrival_rate_class_1"] == arrival_rate
    ].copy()
    data["behavior"] = data.apply(_behavior_label, axis=1)
    data["weight"] = data["w1"].map(lambda value: f"w1={value:g}")
    table = data.pivot(
        index="behavior",
        columns="weight",
        values="util_near_tie_q_ranges",
    )
    ordered_columns = [f"w1={w1:g}" for w1, _ in WEIGHTS]
    table = table.reindex(columns=ordered_columns)
    return table.reset_index()


def write_report(
    *,
    q_summary: pd.DataFrame,
    run_level: pd.DataFrame,
    best_q: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    near_ranges: pd.DataFrame,
    report_dir: Path,
) -> Path:
    figures = report_dir / "figures"
    util_heatmap = figures / "best_q_obj_util_norm.png"
    service_heatmap = figures / "best_q_obj_service_norm.png"
    wait_heatmap = figures / "best_q_t_wait_offered.png"
    delta_plot = figures / "representative_delta_vs_fcfs.png"
    tradeoff_plot = figures / "class_service_tradeoff.png"
    wait_plot = figures / "offered_wait_no_offer_diagnostic.png"
    plot_best_q_heatmap(
        best_q,
        objective="Obj_util_norm",
        path=util_heatmap,
        title="1% near-tie Q ranges for normalized weighted slot utilization",
    )
    plot_best_q_heatmap(
        best_q,
        objective="Obj_service_norm",
        path=service_heatmap,
        title="1% near-tie Q ranges for normalized weighted service rate",
    )
    plot_best_q_heatmap(
        best_q,
        objective="T_wait_offered",
        path=wait_heatmap,
        title="1% near-tie Q ranges for lowest weighted offered wait",
    )
    plot_representative_deltas(run_level, delta_plot)
    plot_class_tradeoffs(q_summary, tradeoff_plot)
    plot_wait_no_offer(q_summary, wait_plot)

    util = best_q[best_q["objective_name"] == "Obj_util_norm"].copy()
    service = best_q[best_q["objective_name"] == "Obj_service_norm"].copy()
    wait = best_q[best_q["objective_name"] == "T_wait_offered"].copy()
    tables = report_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    atomic_write_dataframe(
        scenario_summary,
        tables / "scenario_level_summary.csv",
    )
    atomic_write_dataframe(best_q, tables / "best_q_summary.csv")
    atomic_write_dataframe(near_ranges, tables / "near_tie_q_ranges.csv")

    def findings(frame: pd.DataFrame) -> tuple[int, int, int]:
        improved = int((frame["best_strict_delta_vs_fcfs"] > 1e-12).sum())
        tied_or_better = int(
            frame["near_tie_q_values"].astype(str).str.split(",").apply(
                lambda values: any(int(value) > 0 for value in values)
            ).sum()
        )
        multiple = int(
            frame["near_tie_q_values"].astype(str).str.contains(",").sum()
        )
        return improved, tied_or_better, multiple

    util_improved, util_candidates, util_multiple = findings(util)
    service_improved, service_candidates, service_multiple = findings(service)

    def scalar_best(frame: pd.DataFrame) -> pd.Series:
        return frame["exact_best_q_values"].astype(str).str.split(",").apply(
            lambda values: np.mean([int(value) for value in values])
        )

    util = util.assign(best_q_scalar=scalar_best(util))
    service = service.assign(best_q_scalar=scalar_best(service))

    util_table_25 = _arrival_utilization_table(
        scenario_summary,
        arrival_rate=25,
    )
    util_table_50 = _arrival_utilization_table(
        scenario_summary,
        arrival_rate=50,
    )

    composition_count = int(wait["composition_effect_likely"].sum())
    total_scenarios = len(util)
    wait_q32 = q_summary[
        (q_summary["Q"] == 32)
        & (q_summary["class_2_no_offer_rate_mean"] >= 1 - 1e-12)
    ]
    wait_exclusion_count = int(
        wait.merge(
            wait_q32[SCENARIO_COLUMNS],
            on=SCENARIO_COLUMNS,
            how="inner",
        ).shape[0]
    )
    util_median_delta = float(util["best_strict_delta_vs_fcfs"].median())
    service_median_delta = float(service["best_strict_delta_vs_fcfs"].median())
    lines = [
        "# Strict Class 1 Reservation: Policy-Selection Results",
        "",
        "> **Conclusion.** Strict reservation raises both normalized weighted "
        "objectives in many tested cells, but the gains are produced by moving "
        "service toward Class 1. The expanded weight sweep shows why the "
        "chosen Class 1 weight matters: low `w1` keeps low-Q ranges competitive, "
        "while `w1 >= 1` pushes high-demand settings toward `Q=32`. The lowest "
        "offered waiting time is not an overall access improvement.",
        "",
        "## 1. Purpose",
        "",
        "This report identifies reservation quantities that perform similarly "
        "under two primary normalized objectives. It compares every Q with "
        "pooled FCFS (`Q=0`) and does not treat offered waiting time as an "
        "access objective by itself.",
        "",
        "## 2. Experiment Grid",
        "",
        "- 5 balking-threshold pairs and 3 common post-threshold balking rates.",
        "- Equal Class 1 and Class 2 arrival rates of 25 or 50 per day.",
        "- `Q = [0,2,4,6,8,10,12,16,20,24,28,32]`.",
        "- Class 1 weights `w1 = [0.5,1,1.5,2]` with `w2 = 1`; 20 seeds per simulation cell.",
        "- Capacity 32/day, horizon 14, burn-in 30, measurement 365, cooldown 14.",
        "",
        "## 3. Objective Definitions",
        "",
        "With `S = 32 × 365` measured slots:",
        "",
        "$$Obj_{util,raw}=w_1\\frac{Y_1}{S}+w_2\\frac{Y_2}{S},\\qquad "
        "Obj_{util,norm}=\\frac{Obj_{util,raw}}{w_1+w_2}.$$",
        "",
        "$$Obj_{service,raw}=w_1\\frac{Y_1}{A_1}+w_2\\frac{Y_2}{A_2},\\qquad "
        "Obj_{service,norm}=\\frac{Obj_{service,raw}}{w_1+w_2}.$$",
        "",
        "$$T_{wait,offered}=\\frac{w_1\\sum\\tau_{offered,1}+"
        "w_2\\sum\\tau_{offered,2}}{w_1\\,\\mathrm{offered}_1+"
        "w_2\\,\\mathrm{offered}_2}.$$",
        "",
        "The normalized utilization and service values are the primary "
        "comparison objectives. Raw values are retained. Offered waiting time "
        "is secondary and conditional on receiving an offer.",
        "",
        "## 4. Main Findings",
        "",
        f"- A positive-Q policy exceeds FCFS in mean normalized utilization in "
        f"{util_improved} of {total_scenarios} scenario-weight cells; a positive "
        f"Q appears in the 1% near-tie set in {util_candidates} cells. The "
        f"median best-strict delta is {util_median_delta:+.3f}.",
        f"- A positive-Q policy exceeds FCFS in mean normalized service rate in "
        f"{service_improved} of {total_scenarios} cells; a positive Q appears "
        f"in the 1% near-tie set in {service_candidates} cells. The median "
        f"best-strict delta is {service_median_delta:+.3f}.",
        f"- Several Q values are effectively equivalent in {util_multiple} "
        f"utilization cells and {service_multiple} service-rate cells.",
        "- At 25 arrivals per class, the utilization ranges are sensitive to "
        "`w1`: lower Class 1 weights keep more low-Q and FCFS-equivalent "
        "ranges, while larger weights move the range upward.",
        "- At 50 arrivals per class, the utilization ranges concentrate at high "
        "`Q`. These cells must be read with Class 2 access because high `Q` "
        "approaches full Class 1 protection.",
        "",
        "## 5. Utilization Best-Q Ranges By Regime",
        "",
        "The heatmap below shows 1% near-tie ranges for normalized weighted "
        "slot utilization, not forced unique optima. These are mathematical "
        "candidates, not access-constrained policy recommendations. The "
        "detailed tables are moved to the appendix.",
        "`C1` and `C2` are the class-specific balking thresholds in days; "
        "`high` is the post-threshold balking probability. The solid vertical "
        "line separates 25 arrivals per class from 50 arrivals per class.",
        "",
        "![Utilization best-Q ranges](figures/best_q_obj_util_norm.png)",
        "",
        "The utilization map shows how weighting Class 1 and increasing demand "
        "move the practically equivalent reservation region.",
        "",
        "## 6. FCFS Comparison",
        "",
        "![Representative deltas](figures/representative_delta_vs_fcfs.png)",
        "",
        "Positive values indicate improvement over matched-seed FCFS. The "
        "representative symmetric regime sweeps `w1` from 0.5 to 2.0 in steps "
        "of 0.5. Color shows the Class 1 weight; marker shape shows the "
        "arrival rate. The improvement is weighted: it does not mean both "
        "classes improve.",
        "",
        "## 7. Class Tradeoffs",
        "",
        "![Class service tradeoff](figures/class_service_tradeoff.png)",
        "",
        "Increasing Q generally moves service toward Class 1 and away from "
        "Class 2. The star marks the highest service objective in this "
        "representative slice for `w1=2`, and the dashed line shows an example "
        "iso-objective line: points on it have the same weighted service value. "
        "The lighter diamond and X show where the utilization and offered-wait "
        "objectives point on the same service-rate tradeoff curve. "
        "An objective improvement should therefore be read as a weighted "
        "tradeoff, not as a simultaneous improvement for both classes.",
        "",
        "## 8. Offered Waiting Time And No-Offer Composition Effects",
        "",
        "![Offered-wait best-Q ranges](figures/best_q_t_wait_offered.png)",
        "",
        "The offered-wait heatmap shows the Q ranges that minimize "
        "`T_wait_offered`. This heatmap must be read with the no-offer "
        "diagnostic below, because lower offered wait can come from excluding "
        "patients from offers.",
        "",
        "![Wait and no-offer diagnostic](figures/offered_wait_no_offer_diagnostic.png)",
        "",
        f"Under the pre-specified strict flag, both class no-offer rates rise in "
        f"{composition_count} of {total_scenarios} minimum-wait cells. However, "
        f"all {wait_exclusion_count} minimum-wait cells select `Q=32`, where "
        "Class 2 receives no offers and its waiting-time line is undefined. "
        "The lower weighted offered wait is therefore an access-composition "
        "effect, not a true overall waiting-time improvement.",
        "",
        "## 9. Limitations",
        "",
        "- Same seeds provide matched labels, but policy-dependent RNG use means "
        "they are not exact common-random-number experiments.",
        "- The utilization objective follows measured-arrival cohorts through "
        "cooldown; it is not identical to measured-service-day utilization.",
        "- Results use a 1% practical-equivalence rule and 20 seeds.",
        "- No reservation cost or external access constraint is imposed.",
        "- The reported best Q values maximize the stated weighted objectives; "
        "they should not be adopted without a Class 2 access requirement.",
        "",
        "## 10. Next Steps",
        "",
        "Next steps are tracked in the shared reservation note: "
        "[Reservation Analysis Next Steps](../next_steps.md).",
        "",
        "## Appendix: Detailed Utilization Tables",
        "",
        "These tables give the full 1% near-tie Q ranges behind the utilization "
        "heatmap. Bracketed lists show the tested Q values represented by each "
        "range.",
        "",
        "### Lambda = 25 arrivals per class",
        "",
        _markdown_table(util_table_25),
        "",
        "### Lambda = 50 arrivals per class",
        "",
        _markdown_table(util_table_50),
        "",
        "## Appendix: Service-Rate Heatmap",
        "",
        "The service-rate objective is retained as a secondary primary objective, "
        "but its heatmap is placed here so the main body focuses on the "
        "utilization view requested for visual selection.",
        "",
        "![Service-rate best-Q ranges](figures/best_q_obj_service_norm.png)",
        "",
        "The service-rate map separates access performance from capacity-based "
        "performance; the two objectives need not choose the same range.",
        "",
        "Data tables: `tables/scenario_level_summary.csv` and "
        "`tables/best_q_summary.csv`; explicit near-tie members are in "
        "`tables/near_tie_q_ranges.csv`. Full run-level outputs remain under "
        "`outputs/strict_reservation_policy_selection/standard/`.",
        "",
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "strict_reservation_policy_selection.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_sweep(
    *,
    profile_name: str,
    output_dir: Path,
    workers: int,
    resume: bool,
    generate_report: bool,
    report_dir: Path,
) -> dict[str, Path]:
    profile = profile_grid(profile_name)
    tasks = build_tasks(profile)
    profile_dir = output_dir / profile.name
    manifest_path = profile_dir / "manifest.json"
    manifest = experiment_manifest(profile)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_task_identity(existing) != task_identity_manifest(profile):
            raise RuntimeError(
                f"Manifest mismatch in {profile_dir}; use a clean output directory."
            )
    else:
        write_manifest(manifest_path, manifest)

    completed = valid_completed_ids(output_dir, tasks) if resume else set()
    pending = [task for task in tasks if task.task_id not in completed]
    print(
        f"{profile.name}: {len(tasks):,} simulations; "
        f"{len(completed):,} resumable; {len(pending):,} pending."
    )
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_and_write, task, str(output_dir)): task
                for task in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                future.result()
                if index == 1 or index % max(1, len(pending) // 10) == 0:
                    print(f"Completed {index:,}/{len(pending):,} pending simulations.")

    completed = valid_completed_ids(output_dir, tasks)
    if len(completed) != len(tasks):
        raise RuntimeError(
            f"Only {len(completed):,} of {len(tasks):,} shards are valid."
        )
    paths = consolidate(profile=profile, output_dir=output_dir, tasks=tasks)
    if generate_report:
        q_summary = pd.read_csv(paths["q_summary"])
        run_level = pd.read_csv(paths["run_level"])
        best_q = pd.read_csv(paths["best_q"])
        scenario_summary = pd.read_csv(paths["scenario_summary"])
        near_ranges = pd.read_csv(paths["near_ranges"])
        paths["report"] = write_report(
            q_summary=q_summary,
            run_level=run_level,
            best_q=best_q,
            scenario_summary=scenario_summary,
            near_ranges=near_ranges,
            report_dir=report_dir,
        )
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate the tracked Markdown report and five figures.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive.")
    profile = profile_grid(args.profile)
    cardinality = expected_cardinality(profile)
    print(
        f"{args.profile} cardinality: {cardinality:,} simulations and "
        f"{cardinality * len(WEIGHTS):,} weighted run-level rows."
    )
    if args.profile == "standard" and cardinality != 7200:
        raise AssertionError("Standard grid must contain 7,200 simulations.")
    if args.dry_run:
        return 0
    paths = run_sweep(
        profile_name=args.profile,
        output_dir=args.output_dir.resolve(),
        workers=args.workers,
        resume=args.resume,
        generate_report=args.generate_report,
        report_dir=args.report_dir.resolve(),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
