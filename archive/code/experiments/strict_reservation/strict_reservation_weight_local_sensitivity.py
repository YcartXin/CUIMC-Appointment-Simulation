"""Local weight and behavior sensitivity report for strict reservation."""

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
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from analysis.reservation_policy_metrics import score_simulation_row
from analysis.reservation_policy_selection import (
    bootstrap_mean_ci,
    contiguous_q_ranges,
    near_tie_values,
)
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import PatientClassParams, SimulationConfig, ThresholdRule


DEFAULT_OUTPUT_DIR = (
    REPO_DIR / "outputs" / "strict_reservation_weight_local_sensitivity"
)
DEFAULT_REPORT_DIR = (
    REPO_DIR / "docs" / "reports" / "reservation" / "weight_local_sensitivity"
)

SLOTS_PER_DAY = 32
HORIZON_DAYS = 14
BURN_IN_DAYS = 30
MEASURE_DAYS = 365
COOLDOWN_DAYS = 14
DEFAULT_CANCEL_PROB = 0.10
DEFAULT_BALK_THRESHOLD = 9
DEFAULT_BALK_HIGH = 0.50
BALK_LOW = 0.0
DEFAULT_NO_SHOW_THRESHOLD = 6
NO_SHOW_LOW = 0.0
DEFAULT_NO_SHOW_HIGH = 0.30
ARRIVAL_RATE = 25

BALK_HIGH_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
THRESHOLD_GRID = (5, 7, 9, 11, 12)
CANCEL_PROB_GRID = (0.0, 0.05, 0.10, 0.15, 0.20)
NO_SHOW_HIGH_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)
NO_SHOW_THRESHOLD_GRID = (3, 5, 6, 8, 10)
Q_VALUES = tuple(range(33))
STANDARD_SEEDS = tuple(range(62001, 62021))
WEIGHTS = ((0.9, 1.0), (1.0, 1.0), (1.1, 1.0))
NEAR_TIE_TOLERANCE = 0.01
SHARD_SCHEMA_VERSION = 1

OBJECTIVES = {
    "Obj_util_norm": "maximize",
    "Obj_service_norm": "maximize",
}

SCENARIO_COLUMNS = [
    "analysis_family",
    "scenario_id",
    "arrival_rate_class_1",
    "arrival_rate_class_2",
    "tau_1",
    "tau_2",
    "balk_high_class_1",
    "balk_high_class_2",
    "cancel_prob_class_1",
    "cancel_prob_class_2",
    "no_show_threshold_1",
    "no_show_threshold_2",
    "no_show_high_class_1",
    "no_show_high_class_2",
    "w1",
    "w2",
]

METRIC_COLUMNS = [
    "Obj_util_norm",
    "Obj_service_norm",
    "T_wait_offered",
    "class_1_service_rate",
    "class_2_service_rate",
    "class_1_no_offer_rate",
    "class_2_no_offer_rate",
]


@dataclass(frozen=True)
class Profile:
    name: str
    balk_high_values: tuple[float, ...]
    threshold_values: tuple[int, ...]
    cancel_prob_values: tuple[float, ...]
    no_show_high_values: tuple[float, ...]
    no_show_threshold_values: tuple[int, ...]
    q_values: tuple[int, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class Task:
    profile: str
    analysis_family: str
    scenario_id: str
    arrival_rate_class_1: int
    arrival_rate_class_2: int
    tau_1: int
    tau_2: int
    balk_high_class_1: float
    balk_high_class_2: float
    cancel_prob_class_1: float
    cancel_prob_class_2: float
    no_show_threshold_1: int
    no_show_threshold_2: int
    no_show_high_class_1: float
    no_show_high_class_2: float
    Q: int
    seed: int
    task_id: str


def profile_grid(name: str) -> Profile:
    if name == "standard":
        return Profile(
            name=name,
            balk_high_values=BALK_HIGH_GRID,
            threshold_values=THRESHOLD_GRID,
            cancel_prob_values=CANCEL_PROB_GRID,
            no_show_high_values=NO_SHOW_HIGH_GRID,
            no_show_threshold_values=NO_SHOW_THRESHOLD_GRID,
            q_values=Q_VALUES,
            seeds=STANDARD_SEEDS,
        )
    if name == "smoke":
        return Profile(
            name=name,
            balk_high_values=(0.3, 0.7),
            threshold_values=(5, 9),
            cancel_prob_values=(0.0, 0.20),
            no_show_high_values=(0.1, 0.5),
            no_show_threshold_values=(3, 6),
            q_values=Q_VALUES,
            seeds=(62001, 62002),
        )
    raise ValueError(f"Unknown profile: {name}")


def _format_prob(value: float) -> str:
    return str(int(round(value * 100))).zfill(2)


def _scenario_id(spec: Mapping[str, Any]) -> str:
    return (
        f"{spec['analysis_family']}__bt{spec['tau_1']:02d}_{spec['tau_2']:02d}"
        f"__bh{_format_prob(spec['balk_high_class_1'])}_"
        f"{_format_prob(spec['balk_high_class_2'])}"
        f"__c{_format_prob(spec['cancel_prob_class_1'])}_"
        f"{_format_prob(spec['cancel_prob_class_2'])}"
        f"__nt{spec['no_show_threshold_1']:02d}_"
        f"{spec['no_show_threshold_2']:02d}"
        f"__nh{_format_prob(spec['no_show_high_class_1'])}_"
        f"{_format_prob(spec['no_show_high_class_2'])}"
    )


def _base_behavior(analysis_family: str) -> dict[str, Any]:
    return {
        "analysis_family": analysis_family,
        "arrival_rate_class_1": ARRIVAL_RATE,
        "arrival_rate_class_2": ARRIVAL_RATE,
        "tau_1": DEFAULT_BALK_THRESHOLD,
        "tau_2": DEFAULT_BALK_THRESHOLD,
        "balk_high_class_1": DEFAULT_BALK_HIGH,
        "balk_high_class_2": DEFAULT_BALK_HIGH,
        "cancel_prob_class_1": DEFAULT_CANCEL_PROB,
        "cancel_prob_class_2": DEFAULT_CANCEL_PROB,
        "no_show_threshold_1": DEFAULT_NO_SHOW_THRESHOLD,
        "no_show_threshold_2": DEFAULT_NO_SHOW_THRESHOLD,
        "no_show_high_class_1": DEFAULT_NO_SHOW_HIGH,
        "no_show_high_class_2": DEFAULT_NO_SHOW_HIGH,
    }


def _scenario_specs(profile: Profile) -> Iterable[dict[str, Any]]:
    for b1 in profile.balk_high_values:
        for b2 in profile.balk_high_values:
            yield {
                **_base_behavior("balk_probability_grid"),
                "balk_high_class_1": b1,
                "balk_high_class_2": b2,
            }
    for tau_1 in profile.threshold_values:
        for tau_2 in profile.threshold_values:
            yield {
                **_base_behavior("balk_threshold_grid"),
                "tau_1": tau_1,
                "tau_2": tau_2,
            }
    for tau_1 in profile.threshold_values:
        for b1 in profile.balk_high_values:
            yield {
                **_base_behavior("class1_balk_surface"),
                "tau_1": tau_1,
                "balk_high_class_1": b1,
            }
    for tau_2 in profile.threshold_values:
        for b2 in profile.balk_high_values:
            yield {
                **_base_behavior("class2_balk_surface"),
                "tau_2": tau_2,
                "balk_high_class_2": b2,
            }
    for c1 in profile.cancel_prob_values:
        for c2 in profile.cancel_prob_values:
            yield {
                **_base_behavior("cancellation_probability_grid"),
                "cancel_prob_class_1": c1,
                "cancel_prob_class_2": c2,
            }
    for h1 in profile.no_show_high_values:
        for h2 in profile.no_show_high_values:
            yield {
                **_base_behavior("no_show_probability_grid"),
                "no_show_high_class_1": h1,
                "no_show_high_class_2": h2,
            }
    for t1 in profile.no_show_threshold_values:
        for t2 in profile.no_show_threshold_values:
            yield {
                **_base_behavior("no_show_threshold_grid"),
                "no_show_threshold_1": t1,
                "no_show_threshold_2": t2,
            }
    for t1 in profile.no_show_threshold_values:
        for h1 in profile.no_show_high_values:
            yield {
                **_base_behavior("class1_no_show_surface"),
                "no_show_threshold_1": t1,
                "no_show_high_class_1": h1,
            }
    for t2 in profile.no_show_threshold_values:
        for h2 in profile.no_show_high_values:
            yield {
                **_base_behavior("class2_no_show_surface"),
                "no_show_threshold_2": t2,
                "no_show_high_class_2": h2,
            }


def experiment_manifest(profile: Profile) -> dict[str, Any]:
    manifest = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "profile": asdict(profile),
        "weights": WEIGHTS,
        "arrival_rate": ARRIVAL_RATE,
        "slots_per_day": SLOTS_PER_DAY,
        "horizon_days": HORIZON_DAYS,
        "burn_in_days": BURN_IN_DAYS,
        "measure_days": MEASURE_DAYS,
        "cooldown_days": COOLDOWN_DAYS,
        "baseline_behavior": {
            "balk_threshold": DEFAULT_BALK_THRESHOLD,
            "balk_low": BALK_LOW,
            "balk_high": DEFAULT_BALK_HIGH,
            "cancel_prob": DEFAULT_CANCEL_PROB,
            "no_show_threshold": DEFAULT_NO_SHOW_THRESHOLD,
            "no_show_low": NO_SHOW_LOW,
            "no_show_high": DEFAULT_NO_SHOW_HIGH,
        },
        "behavior_grids": {
            "balk_high_values": profile.balk_high_values,
            "balk_threshold_values": profile.threshold_values,
            "cancel_prob_values": profile.cancel_prob_values,
            "no_show_high_values": profile.no_show_high_values,
            "no_show_threshold_values": profile.no_show_threshold_values,
        },
        "near_tie_tolerance": NEAR_TIE_TOLERANCE,
    }
    return json.loads(json.dumps(manifest, sort_keys=True))


def experiment_manifest_hash(profile: Profile) -> str:
    return hashlib.sha256(
        json.dumps(experiment_manifest(profile), sort_keys=True, default=list).encode(
            "utf-8"
        )
    ).hexdigest()[:10]


def expected_cardinality(profile: str | Profile) -> int:
    grid = profile_grid(profile) if isinstance(profile, str) else profile
    return (
        len(tuple(_scenario_specs(grid)))
        * len(grid.q_values)
        * len(grid.seeds)
    )


def build_tasks(profile: str | Profile) -> list[Task]:
    grid = profile_grid(profile) if isinstance(profile, str) else profile
    manifest_hash = experiment_manifest_hash(grid)
    tasks: list[Task] = []
    for spec in _scenario_specs(grid):
        scenario_id = _scenario_id(spec)
        for q in grid.q_values:
            for seed in grid.seeds:
                task_id = (
                    f"{grid.name}__{scenario_id}__q{q:02d}"
                    f"__seed{seed}__{manifest_hash}"
                )
                tasks.append(
                    Task(
                        profile=grid.name,
                        analysis_family=spec["analysis_family"],
                        scenario_id=scenario_id,
                        arrival_rate_class_1=spec["arrival_rate_class_1"],
                        arrival_rate_class_2=spec["arrival_rate_class_2"],
                        tau_1=spec["tau_1"],
                        tau_2=spec["tau_2"],
                        balk_high_class_1=spec["balk_high_class_1"],
                        balk_high_class_2=spec["balk_high_class_2"],
                        cancel_prob_class_1=spec["cancel_prob_class_1"],
                        cancel_prob_class_2=spec["cancel_prob_class_2"],
                        no_show_threshold_1=spec["no_show_threshold_1"],
                        no_show_threshold_2=spec["no_show_threshold_2"],
                        no_show_high_class_1=spec["no_show_high_class_1"],
                        no_show_high_class_2=spec["no_show_high_class_2"],
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
            balk_prob=ThresholdRule(task.tau_1, BALK_LOW, task.balk_high_class_1),
            cancel_prob=task.cancel_prob_class_1,
            no_show_prob=ThresholdRule(
                task.no_show_threshold_1,
                NO_SHOW_LOW,
                task.no_show_high_class_1,
            ),
        ),
        2: PatientClassParams(
            class_id=2,
            lambda_per_day=task.arrival_rate_class_2,
            balk_prob=ThresholdRule(task.tau_2, BALK_LOW, task.balk_high_class_2),
            cancel_prob=task.cancel_prob_class_2,
            no_show_prob=ThresholdRule(
                task.no_show_threshold_2,
                NO_SHOW_LOW,
                task.no_show_high_class_2,
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
        "accounting_ok": (
            c1.arrivals == c1.offered + c1.no_offer
            and c2.arrivals == c2.offered + c2.no_offer
            and unresolved_1 == 0
            and unresolved_2 == 0
        ),
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
    completed: set[str] = set()
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
    text_columns = {"profile", "analysis_family", "scenario_id", "task_id"}
    bool_columns = {"accounting_ok"}
    for column in frame:
        if column in text_columns:
            continue
        if column in bool_columns:
            frame[column] = frame[column].str.lower().eq("true")
        else:
            frame[column] = pd.to_numeric(frame[column])
    return frame


def pair_with_fcfs(run_level: pd.DataFrame) -> pd.DataFrame:
    pair_keys = [*SCENARIO_COLUMNS, "seed"]
    paired_columns = [
        "A1",
        "A2",
        "Y1",
        "Y2",
        "offered_1",
        "offered_2",
        "sum_tau_offered_1",
        "sum_tau_offered_2",
        *METRIC_COLUMNS,
    ]
    fcfs = run_level.loc[run_level["Q"] == 0, pair_keys + paired_columns].copy()
    if fcfs.duplicated(pair_keys).any():
        raise ValueError("Duplicate FCFS rows found.")
    fcfs = fcfs.rename(
        columns={column: f"fcfs_{column}" for column in paired_columns}
    )
    paired = run_level.merge(fcfs, on=pair_keys, how="left", validate="many_to_one")
    for column in paired_columns:
        paired[f"delta_{column}"] = paired[column] - paired[f"fcfs_{column}"]
    if paired.filter(regex=r"^fcfs_").isna().all(axis=1).any():
        raise AssertionError("Missing FCFS values after pairing.")
    return paired


def score_rows(simulation_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in simulation_rows.to_dict("records"):
        for w1, w2 in WEIGHTS:
            rows.append(score_simulation_row(record, w1=w1, w2=w2))
    return pair_with_fcfs(pd.DataFrame(rows))


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


def q_level_summary(run_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in run_level.groupby([*SCENARIO_COLUMNS, "Q"], sort=True):
        row = dict(zip([*SCENARIO_COLUMNS, "Q"], keys))
        for metric in METRIC_COLUMNS:
            mean, low, high, n = bootstrap_mean_ci(group[metric].to_numpy())
            delta_mean, delta_low, delta_high, delta_n = bootstrap_mean_ci(
                group[f"delta_{metric}"].to_numpy()
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
            row[f"{metric}_n"] = n
            row[f"delta_{metric}_mean"] = delta_mean
            row[f"delta_{metric}_ci95_low"] = delta_low
            row[f"delta_{metric}_ci95_high"] = delta_high
            row[f"delta_{metric}_n"] = delta_n
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_best_q(q_summary: pd.DataFrame, q_values: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_keys, group in q_summary.groupby(SCENARIO_COLUMNS, sort=True):
        scenario = dict(zip(SCENARIO_COLUMNS, scenario_keys))
        by_q = group.set_index("Q").sort_index()
        fcfs = by_q.loc[0]
        for objective, direction in OBJECTIVES.items():
            exact, near, best_value = near_tie_values(
                by_q[f"{objective}_mean"],
                direction=direction,
                relative_tolerance=NEAR_TIE_TOLERANCE,
            )
            ranges = contiguous_q_ranges(near, tested_q_values=q_values)
            rows.append(
                {
                    **scenario,
                    "objective_name": objective,
                    "direction": direction,
                    "fcfs_mean": fcfs[f"{objective}_mean"],
                    "exact_best_q_values": ",".join(map(str, exact)),
                    "exact_best_q_scalar": float(np.mean(exact)) if exact else np.nan,
                    "exact_best_value": best_value,
                    "near_tie_q_values": ",".join(map(str, near)),
                    "near_tie_q_ranges": "; ".join(
                        item["range_label"] for item in ranges
                    ),
                    "best_delta_vs_fcfs": best_value - fcfs[f"{objective}_mean"],
                }
            )
    return pd.DataFrame(rows)


def consolidate(
    *,
    profile: Profile,
    output_dir: Path,
    tasks: Sequence[Task],
) -> dict[str, Path]:
    profile_dir = output_dir / profile.name
    simulation_rows = load_simulation_rows(output_dir, tasks)
    run_level = score_rows(simulation_rows)
    q_summary = q_level_summary(run_level)
    best_q = summarize_best_q(q_summary, profile.q_values)
    paths = {
        "simulation": profile_dir / "simulation_results.csv.gz",
        "run_level": profile_dir / "run_level_results.csv.gz",
        "q_summary": profile_dir / "q_level_summary.csv",
        "best_q": profile_dir / "best_q_summary.csv",
    }
    atomic_write_dataframe(simulation_rows, paths["simulation"])
    atomic_write_dataframe(run_level, paths["run_level"])
    atomic_write_dataframe(q_summary, paths["q_summary"])
    atomic_write_dataframe(best_q, paths["best_q"])
    return paths


def _cell_label(row: pd.Series) -> str:
    values = [int(value) for value in str(row["exact_best_q_values"]).split(",") if value]
    if len(values) <= 3:
        return ",".join(map(str, values))
    ranges = contiguous_q_ranges(values, tested_q_values=Q_VALUES)
    return "; ".join(item["range_label"].split(" ", 1)[0] for item in ranges)


def plot_best_q_surface(
    best_q: pd.DataFrame,
    *,
    family: str,
    objective: str,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    path: Path,
) -> None:
    data = best_q[
        (best_q["analysis_family"] == family)
        & (best_q["objective_name"] == objective)
    ].copy()
    weights = sorted(data["w1"].dropna().unique())
    x_values = sorted(data[x_column].dropna().unique())
    y_values = sorted(data[y_column].dropna().unique())
    fig, axes = plt.subplots(1, len(weights), figsize=(5.1 * len(weights), 4.6))
    axes = np.atleast_1d(axes)
    for ax, w1 in zip(axes, weights):
        subset = data[np.isclose(data["w1"], w1)]
        matrix = np.full((len(y_values), len(x_values)), np.nan)
        labels = [["" for _ in x_values] for _ in y_values]
        for _, row in subset.iterrows():
            i = y_values.index(row[y_column])
            j = x_values.index(row[x_column])
            matrix[i, j] = row["exact_best_q_scalar"]
            labels[i][j] = _cell_label(row)
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=32, aspect="auto")
        ax.set_xticks(range(len(x_values)), [f"{value:g}" for value in x_values])
        ax.set_yticks(range(len(y_values)), [f"{value:g}" for value in y_values])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"w1={w1:g}, w2=1")
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                if not np.isfinite(matrix[i, j]):
                    continue
                ax.text(
                    j,
                    i,
                    labels[i][j],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if matrix[i, j] < 16 else "black",
                )
    fig.suptitle(title)
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.15, top=0.82, wspace=0.30)
    cax = fig.add_axes([0.92, 0.18, 0.014, 0.60])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Best Q by mean Obj_util_norm")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
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


def compact_summary(best_q: pd.DataFrame) -> pd.DataFrame:
    util = best_q[best_q["objective_name"] == "Obj_util_norm"].copy()
    return (
        util.groupby(["analysis_family", "w1"], as_index=False)
        .agg(
            median_best_q=("exact_best_q_scalar", "median"),
            low_best_q=("exact_best_q_scalar", "min"),
            high_best_q=("exact_best_q_scalar", "max"),
            median_delta_vs_fcfs=("best_delta_vs_fcfs", "median"),
        )
        .round(4)
    )


def write_report(
    *,
    profile: Profile,
    best_q: pd.DataFrame,
    report_dir: Path,
) -> Path:
    figures = report_dir / "figures"
    tables = report_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    best_q_path = tables / "best_q_summary.csv"
    summary_path = tables / "family_weight_summary.csv"
    atomic_write_dataframe(best_q, best_q_path)
    family_summary = compact_summary(best_q)
    atomic_write_dataframe(family_summary, summary_path)

    probability_plot = figures / "best_q_balk_probability_grid.png"
    threshold_plot = figures / "best_q_threshold_grid.png"
    class1_plot = figures / "best_q_class1_threshold_probability_surface.png"
    class2_plot = figures / "best_q_class2_threshold_probability_surface.png"
    cancellation_plot = figures / "best_q_cancellation_probability_grid.png"
    no_show_probability_plot = figures / "best_q_no_show_probability_grid.png"
    no_show_threshold_plot = figures / "best_q_no_show_threshold_grid.png"
    class1_no_show_plot = (
        figures / "best_q_class1_no_show_threshold_probability_surface.png"
    )
    class2_no_show_plot = (
        figures / "best_q_class2_no_show_threshold_probability_surface.png"
    )
    plot_best_q_surface(
        best_q,
        family="balk_probability_grid",
        objective="Obj_util_norm",
        x_column="balk_high_class_2",
        y_column="balk_high_class_1",
        x_label="Class 2 post-threshold balk probability",
        y_label="Class 1 post-threshold balk probability",
        title="Best Q as class-specific balk probabilities change",
        path=probability_plot,
    )
    plot_best_q_surface(
        best_q,
        family="balk_threshold_grid",
        objective="Obj_util_norm",
        x_column="tau_2",
        y_column="tau_1",
        x_label="Class 2 balking threshold (days)",
        y_label="Class 1 balking threshold (days)",
        title="Best Q as class-specific balking thresholds change",
        path=threshold_plot,
    )
    plot_best_q_surface(
        best_q,
        family="class1_balk_surface",
        objective="Obj_util_norm",
        x_column="balk_high_class_1",
        y_column="tau_1",
        x_label="Class 1 post-threshold balk probability",
        y_label="Class 1 balking threshold (days)",
        title="Best Q when Class 2 balking is fixed at threshold 9, high 0.5",
        path=class1_plot,
    )
    plot_best_q_surface(
        best_q,
        family="class2_balk_surface",
        objective="Obj_util_norm",
        x_column="balk_high_class_2",
        y_column="tau_2",
        x_label="Class 2 post-threshold balk probability",
        y_label="Class 2 balking threshold (days)",
        title="Best Q when Class 1 balking is fixed at threshold 9, high 0.5",
        path=class2_plot,
    )
    plot_best_q_surface(
        best_q,
        family="cancellation_probability_grid",
        objective="Obj_util_norm",
        x_column="cancel_prob_class_2",
        y_column="cancel_prob_class_1",
        x_label="Class 2 cancellation probability",
        y_label="Class 1 cancellation probability",
        title="Best Q as class-specific cancellation probabilities change",
        path=cancellation_plot,
    )
    plot_best_q_surface(
        best_q,
        family="no_show_probability_grid",
        objective="Obj_util_norm",
        x_column="no_show_high_class_2",
        y_column="no_show_high_class_1",
        x_label="Class 2 post-threshold no-show probability",
        y_label="Class 1 post-threshold no-show probability",
        title="Best Q as class-specific no-show probabilities change",
        path=no_show_probability_plot,
    )
    plot_best_q_surface(
        best_q,
        family="no_show_threshold_grid",
        objective="Obj_util_norm",
        x_column="no_show_threshold_2",
        y_column="no_show_threshold_1",
        x_label="Class 2 no-show threshold (days)",
        y_label="Class 1 no-show threshold (days)",
        title="Best Q as class-specific no-show thresholds change",
        path=no_show_threshold_plot,
    )
    plot_best_q_surface(
        best_q,
        family="class1_no_show_surface",
        objective="Obj_util_norm",
        x_column="no_show_high_class_1",
        y_column="no_show_threshold_1",
        x_label="Class 1 post-threshold no-show probability",
        y_label="Class 1 no-show threshold (days)",
        title="Best Q when Class 2 no-show is fixed at threshold 6, high 0.3",
        path=class1_no_show_plot,
    )
    plot_best_q_surface(
        best_q,
        family="class2_no_show_surface",
        objective="Obj_util_norm",
        x_column="no_show_high_class_2",
        y_column="no_show_threshold_2",
        x_label="Class 2 post-threshold no-show probability",
        y_label="Class 2 no-show threshold (days)",
        title="Best Q when Class 1 no-show is fixed at threshold 6, high 0.3",
        path=class2_no_show_plot,
    )

    lines = [
        "# Local Weight And Behavior Sensitivity For Strict Reservation",
        "",
        "> **Conclusion.** This report checks whether small changes around "
        "equal class weights change the reservation quantity selected by the "
        "normalized utilization objective. It varies `w1 = [0.9, 1.0, 1.1]` "
        "with `w2 = 1.0` under equal demand of 25 arrivals per class, then "
        "varies balking, cancellation, and no-show behavior one block at a time.",
        "",
        "## 1. Purpose",
        "",
        "The goal is to see how sensitive the best reservation quantity `Q` is "
        "to small changes in Class 1 weight and to class-specific patient "
        "behavior. This is exploratory and does not recommend a policy.",
        "",
        "## 2. Experiment Grid",
        "",
        "- Strict Class 1 reservation only.",
        "- Equal demand: `lambda_1 = lambda_2 = 25` per day.",
        "- `Q` uses every integer from 0 through 32.",
        "- Weights: `w1 = [0.9,1.0,1.1]`, `w2 = 1.0`.",
        f"- Balk probabilities: `[{','.join(f'{value:g}' for value in profile.balk_high_values)}]`.",
        f"- Balking thresholds: `[{','.join(map(str, profile.threshold_values))}]` days.",
        f"- Cancellation probabilities: `[{','.join(f'{value:g}' for value in profile.cancel_prob_values)}]`.",
        f"- No-show probabilities: `[{','.join(f'{value:g}' for value in profile.no_show_high_values)}]`.",
        f"- No-show thresholds: `[{','.join(map(str, profile.no_show_threshold_values))}]` days.",
        f"- {len(profile.seeds)} seeds, capacity 32/day, horizon 14, burn-in 30, measurement 365, cooldown 14.",
        "- In each behavior block, the other two behaviors are fixed at "
        "baseline: cancellation 0.10, balking threshold 9/high 0.50, "
        "and no-show threshold 6/high 0.30.",
        "",
        "## 3. Objective Used In The Plots",
        "",
        "The plots use normalized weighted slot utilization:",
        "",
        "$$Obj_{util,norm}=\\frac{w_1\\frac{Y_1}{S}+w_2\\frac{Y_2}{S}}{w_1+w_2}.$$",
        "",
        "Color and cell labels show the best `Q` by mean objective value "
        "across seeds. `Q=0` is pooled FCFS. Near-tie ranges are saved in "
        "`tables/best_q_summary.csv`.",
        "",
        "## 4. Summary By Analysis Family",
        "",
        markdown_table(family_summary),
        "",
        "## 5. Balking Sensitivity",
        "",
        "This grid varies the post-threshold balking probability for both "
        "classes while keeping both thresholds at 9 days. Cancellation and "
        "no-show behavior are fixed at baseline.",
        "",
        "![Best Q by balk probability](figures/best_q_balk_probability_grid.png)",
        "",
        "This grid varies the balking threshold for both classes while keeping "
        "both post-threshold balking probabilities at 0.5.",
        "",
        "![Best Q by threshold days](figures/best_q_threshold_grid.png)",
        "",
        "Class 2 is fixed at threshold 9 days and post-threshold balking "
        "probability 0.5. Class 1 threshold and post-threshold balking "
        "probability vary.",
        "",
        "![Best Q by Class 1 threshold and probability](figures/best_q_class1_threshold_probability_surface.png)",
        "",
        "Class 1 is fixed at threshold 9 days and post-threshold balking "
        "probability 0.5. Class 2 threshold and post-threshold balking "
        "probability vary.",
        "",
        "![Best Q by Class 2 threshold and probability](figures/best_q_class2_threshold_probability_surface.png)",
        "",
        "## 6. Cancellation Sensitivity",
        "",
        "This grid varies the cancellation probability for both classes. "
        "Balking and no-show behavior are fixed at baseline.",
        "",
        "![Best Q by cancellation probability](figures/best_q_cancellation_probability_grid.png)",
        "",
        "## 7. No-Show Sensitivity",
        "",
        "This grid varies the post-threshold no-show probability for both "
        "classes while keeping both no-show thresholds at 6 days. Balking "
        "and cancellation behavior are fixed at baseline.",
        "",
        "![Best Q by no-show probability](figures/best_q_no_show_probability_grid.png)",
        "",
        "This grid varies the no-show threshold for both classes while keeping "
        "both post-threshold no-show probabilities at 0.3.",
        "",
        "![Best Q by no-show threshold days](figures/best_q_no_show_threshold_grid.png)",
        "",
        "Class 2 no-show behavior is fixed at threshold 6 days and "
        "post-threshold probability 0.3. Class 1 no-show threshold and "
        "post-threshold probability vary.",
        "",
        "![Best Q by Class 1 no-show threshold and probability](figures/best_q_class1_no_show_threshold_probability_surface.png)",
        "",
        "Class 1 no-show behavior is fixed at threshold 6 days and "
        "post-threshold probability 0.3. Class 2 no-show threshold and "
        "post-threshold probability vary.",
        "",
        "![Best Q by Class 2 no-show threshold and probability](figures/best_q_class2_no_show_threshold_probability_surface.png)",
        "",
        "## 8. Interpretation Notes",
        "",
        "- These plots show mathematical best Q values for `Obj_util_norm`; they "
        "are not access-constrained recommendations.",
        "- Small changes from `w1=0.9` to `w1=1.1` are useful for detecting "
        "whether the equal-weight case is fragile.",
        "- Service-rate objective results are saved in `tables/best_q_summary.csv` "
        "but not plotted here to keep the report compact.",
        "- Offered wait is not used as a selection objective in this report.",
        "",
        "## Files",
        "",
        f"- Best-Q table: `tables/{best_q_path.name}`",
        f"- Compact family summary: `tables/{summary_path.name}`",
        "",
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "strict_reservation_weight_local_sensitivity.md"
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
    manifest_path = profile_dir / f"manifest_{experiment_manifest_hash(profile)}.json"
    manifest = experiment_manifest(profile)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                f"Manifest mismatch for {profile_dir}. Use a different output dir."
            )
    else:
        profile_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    completed = valid_completed_ids(output_dir, tasks) if resume else set()
    pending = [task for task in tasks if task.task_id not in completed]
    print(
        f"Profile {profile.name}: {len(tasks):,} tasks; "
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
                if index % 500 == 0 or index == len(pending):
                    print(f"Completed {index:,}/{len(pending):,} pending tasks.")

    paths = consolidate(profile=profile, output_dir=output_dir, tasks=tasks)
    if generate_report:
        best_q = pd.read_csv(paths["best_q"])
        paths["report"] = write_report(
            profile=profile,
            best_q=best_q,
            report_dir=report_dir,
        )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate-report", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = profile_grid(args.profile)
    print(
        f"{profile.name} cardinality: {expected_cardinality(profile):,} tasks "
        f"({len(tuple(_scenario_specs(profile)))} behavior cells x "
        f"{len(profile.q_values)} Q x {len(profile.seeds)} seeds)."
    )
    if args.dry_run:
        return
    paths = run_sweep(
        profile_name=args.profile,
        output_dir=args.output_dir,
        workers=args.workers,
        resume=args.resume,
        generate_report=args.generate_report,
        report_dir=args.report_dir,
    )
    if "report" in paths:
        print(f"Report: {paths['report']}")


if __name__ == "__main__":
    main()
