"""Postprocess H1 raw shards into patient-group and objective-tradeoff outputs.

This script does not run new simulations. It re-selects each policy optimum from
existing H1 raw shards under both objectives:

* average_utilization
* weighted_utilization (Class 1 served rate has twice Class 2's weight)

For each strict/release variant it writes:

* selected_policy_seed_outcomes.csv
    One row per background x selection objective x policy x seed.
* selected_policy_outcomes.csv
    One row per background x selection objective x policy, with seed means.
* pairwise_group_deltas.csv
    All six policy comparisons, including paired deltas for Class 1 and Class 2
    served rates and a patient-group tradeoff category.
* objective_switch_deltas.csv
    Weighted-optimal minus average-optimal outcomes for each policy.
* selection_validation.csv
    Optional comparison against existing condition_optima.csv summary files.
* postprocess_summary.md
    Basic counts and validation notes.

Run from the repository root, normally on the CBS Grid where the raw shards
remain available. See H1_POLICY_POSTPROCESS_README.md for commands.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


POLICIES = ("baseline", "horizon_only", "reservation_only", "both_flexible")
OBJECTIVES = ("average_utilization", "weighted_utilization")

COMPARISONS = (
    ("horizon_only_vs_baseline", "horizon_only", "baseline"),
    ("reservation_only_vs_baseline", "reservation_only", "baseline"),
    ("both_flexible_vs_baseline", "both_flexible", "baseline"),
    ("both_flexible_vs_horizon_only", "both_flexible", "horizon_only"),
    ("both_flexible_vs_reservation_only", "both_flexible", "reservation_only"),
    ("reservation_only_vs_horizon_only", "reservation_only", "horizon_only"),
)

CORE_METRICS = (
    "average_utilization",
    "weighted_utilization",
    "class_1_percent_serviced",
    "class_2_percent_serviced",
    "class_1_slot_utilization",
    "class_2_slot_utilization",
    "access_advantage_class_1",
    "class_1_balking_rate",
    "class_2_balking_rate",
    "class_1_no_show_rate",
    "class_2_no_show_rate",
    "class_1_no_offer_rate",
    "class_2_no_offer_rate",
    "class_1_mean_offered_booking_delay",
    "class_2_mean_offered_booking_delay",
)

DELTA_METRICS = (
    "average_utilization",
    "weighted_utilization",
    "class_1_percent_serviced",
    "class_2_percent_serviced",
    "class_1_slot_utilization",
    "class_2_slot_utilization",
)

REQUIRED_RAW_COLUMNS = {
    "stage",
    "source_background_id",
    "seed",
    "horizon_days",
    "Q",
    "window",
    *OBJECTIVES,
    "class_1_percent_serviced",
    "class_2_percent_serviced",
}

DEFAULT_TOLERANCE = 0.005
DEFAULT_DRAWS = 2000


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _validate_columns(frame: pd.DataFrame, required: Iterable[str], *, label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _dedupe_policy_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Keep one deterministic row per policy cell and seed.

    A coarse and fine phase can contain the same (H, Q, window, seed) cell.
    Keeping one row prevents duplicated phases from changing the objective mean
    or seed-paired comparisons.
    """
    if cells.empty:
        return cells.copy()
    sort_cols = [column for column in ("horizon_days", "Q", "window", "seed", "arm") if column in cells]
    ordered = cells.sort_values(sort_cols, kind="stable") if sort_cols else cells
    return ordered.drop_duplicates(
        subset=["horizon_days", "Q", "window", "seed"],
        keep="first",
    )


def _condition_optimum(
    shard: pd.DataFrame,
    stage: str,
    objective: str,
    *,
    extra_zero_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return seed-level rows for the best evaluated policy cell."""
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")

    cells = shard[shard["stage"] == stage].copy()
    if extra_zero_rows is not None and not extra_zero_rows.empty:
        cells = pd.concat([cells, extra_zero_rows], ignore_index=True)
    cells = _dedupe_policy_cells(cells)
    if cells.empty:
        raise ValueError(f"No candidate rows found for stage={stage!r}")

    group_cols = ["horizon_days", "Q", "window"]
    means = cells.groupby(group_cols, as_index=False, sort=True)[objective].mean()
    best = means.loc[means[objective].idxmax()]
    mask = (
        (cells["horizon_days"] == best["horizon_days"])
        & (cells["Q"] == best["Q"])
        & (cells["window"] == best["window"])
    )
    selected = cells.loc[mask].copy()
    selected = selected.groupby("seed", as_index=False, sort=True).first()
    return selected


def select_policies(shard: pd.DataFrame, objective: str) -> dict[str, pd.DataFrame]:
    """Replicate H1's four-regime selection logic for one objective."""
    _validate_columns(shard, REQUIRED_RAW_COLUMNS, label="raw shard")

    baseline = _dedupe_policy_cells(shard[shard["stage"] == "baseline"].copy())
    if baseline.empty:
        raise ValueError("No baseline rows found")
    baseline = baseline.groupby("seed", as_index=False, sort=True).first()

    horizon_only = _condition_optimum(shard, "horizon_only", objective)
    reservation_only = _condition_optimum(
        shard,
        "reservation_only",
        objective,
        extra_zero_rows=baseline,
    )
    both_flexible = _condition_optimum(
        shard,
        "both_flexible",
        objective,
        extra_zero_rows=horizon_only,
    )

    return {
        "baseline": baseline,
        "horizon_only": horizon_only,
        "reservation_only": reservation_only,
        "both_flexible": both_flexible,
    }


def _paired_delta_ci(
    first: pd.DataFrame,
    second: pd.DataFrame,
    metric: str,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float, int]:
    """Paired bootstrap CI for first minus second, aligned by seed."""
    a = first.groupby("seed", sort=True)[metric].mean()
    b = second.groupby("seed", sort=True)[metric].mean()
    paired = sorted(set(a.index) & set(b.index))
    if not paired:
        return float("nan"), float("nan"), float("nan"), 0

    deltas = a.loc[paired].to_numpy(float) - b.loc[paired].to_numpy(float)
    mean = float(deltas.mean())
    if draws <= 0 or len(deltas) == 1:
        return mean, float("nan"), float("nan"), len(deltas)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(draws, len(deltas)))
    boot_means = deltas[indices].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return mean, float(low), float(high), len(deltas)


def _change_status(mean: float, low: float, high: float, tolerance: float) -> str:
    if np.isnan(mean):
        return "missing"
    if abs(mean) < tolerance:
        return "practical_tie"
    if not np.isnan(low) and not np.isnan(high):
        if mean >= tolerance and low > 0:
            return "supported_increase"
        if mean <= -tolerance and high < 0:
            return "supported_decrease"
    return "uncertain_increase" if mean > 0 else "uncertain_decrease"


def _direction(mean: float, tolerance: float) -> str:
    if np.isnan(mean):
        return "missing"
    if mean >= tolerance:
        return "increase"
    if mean <= -tolerance:
        return "decrease"
    return "tie"


def _tradeoff_category(c1_direction: str, c2_direction: str) -> str:
    if "uncertain" in (c1_direction, c2_direction):
        return "uncertain"
    mapping = {
        ("increase", "increase"): "both_groups_gain",
        ("increase", "decrease"): "priority_gains_general_loses",
        ("decrease", "increase"): "priority_loses_general_gains",
        ("decrease", "decrease"): "both_groups_lose",
        ("tie", "tie"): "both_groups_tied",
        ("increase", "tie"): "priority_gains_general_tied",
        ("tie", "increase"): "priority_tied_general_gains",
        ("decrease", "tie"): "priority_loses_general_tied",
        ("tie", "decrease"): "priority_tied_general_loses",
    }
    return mapping.get((c1_direction, c2_direction), "missing")


def _supported_direction(status: str) -> str:
    return {
        "supported_increase": "increase",
        "supported_decrease": "decrease",
        "practical_tie": "tie",
        "missing": "missing",
    }.get(status, "uncertain")


def _selected_source_stage(policy: str, cells: pd.DataFrame) -> str:
    stages = sorted(set(cells["stage"].astype(str)))
    if len(stages) == 1:
        return stages[0]
    return "+".join(stages)


def _seed_rows_for_selection(
    *,
    background_id: str,
    variant: str,
    objective: str,
    policy: str,
    cells: pd.DataFrame,
) -> pd.DataFrame:
    rows = cells.copy()
    rename = {}
    if "background_id" in rows.columns:
        rename["background_id"] = "raw_policy_cell_id"
    if "variant" in rows.columns:
        rename["variant"] = "raw_variant"
    if rename:
        rows = rows.rename(columns=rename)
    rows.insert(0, "policy", policy)
    rows.insert(0, "selection_objective", objective)
    rows.insert(0, "variant", variant)
    rows.insert(0, "background_id", background_id)
    rows["selected_source_stage"] = _selected_source_stage(policy, cells)
    rows["selected_horizon_days"] = int(cells["horizon_days"].iloc[0])
    rows["selected_Q"] = int(cells["Q"].iloc[0])
    rows["selected_window"] = int(cells["window"].iloc[0])
    return rows


def _mean_row(
    *,
    background_id: str,
    variant: str,
    objective: str,
    policy: str,
    cells: pd.DataFrame,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "background_id": background_id,
        "variant": variant,
        "selection_objective": objective,
        "policy": policy,
        "selected_source_stage": _selected_source_stage(policy, cells),
        "selected_horizon_days": int(cells["horizon_days"].iloc[0]),
        "selected_Q": int(cells["Q"].iloc[0]),
        "selected_window": int(cells["window"].iloc[0]),
        "n_seeds": int(cells["seed"].nunique()),
    }
    for metric in CORE_METRICS:
        if metric in cells.columns:
            row[metric] = float(cells[metric].mean())
            row[f"{metric}_sd"] = float(cells[metric].std(ddof=1)) if len(cells) > 1 else 0.0
    return row


def _comparison_row(
    *,
    background_id: str,
    variant: str,
    objective: str,
    comparison: str,
    first_policy: str,
    second_policy: str,
    first: pd.DataFrame,
    second: pd.DataFrame,
    draws: int,
    tolerance: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "background_id": background_id,
        "variant": variant,
        "selection_objective": objective,
        "comparison": comparison,
        "first_policy": first_policy,
        "second_policy": second_policy,
    }
    n_paired: int | None = None
    for metric in DELTA_METRICS:
        if metric not in first.columns or metric not in second.columns:
            continue
        mean, low, high, n = _paired_delta_ci(
            first,
            second,
            metric,
            draws=draws,
            seed=_stable_seed(background_id, variant, objective, comparison, metric),
        )
        n_paired = n if n_paired is None else min(n_paired, n)
        row[f"delta_{metric}"] = mean
        row[f"delta_{metric}_ci_low"] = low
        row[f"delta_{metric}_ci_high"] = high
        row[f"{metric}_change_status"] = _change_status(mean, low, high, tolerance)

    row["n_paired_seeds"] = int(n_paired or 0)
    c1 = row.get("delta_class_1_percent_serviced", float("nan"))
    c2 = row.get("delta_class_2_percent_serviced", float("nan"))
    c1_direction = _direction(c1, tolerance)
    c2_direction = _direction(c2, tolerance)
    row["class_1_served_rate_direction"] = c1_direction
    row["class_2_served_rate_direction"] = c2_direction
    row["patient_group_tradeoff"] = _tradeoff_category(c1_direction, c2_direction)
    c1_supported = _supported_direction(
        row.get("class_1_percent_serviced_change_status", "missing")
    )
    c2_supported = _supported_direction(
        row.get("class_2_percent_serviced_change_status", "missing")
    )
    row["patient_group_supported_tradeoff"] = _tradeoff_category(
        c1_supported, c2_supported
    )
    return row


def _objective_switch_row(
    *,
    background_id: str,
    variant: str,
    policy: str,
    average_cells: pd.DataFrame,
    weighted_cells: pd.DataFrame,
    draws: int,
    tolerance: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "background_id": background_id,
        "variant": variant,
        "policy": policy,
        "comparison": "weighted_optimal_minus_average_optimal",
        "average_optimal_horizon_days": int(average_cells["horizon_days"].iloc[0]),
        "average_optimal_Q": int(average_cells["Q"].iloc[0]),
        "average_optimal_window": int(average_cells["window"].iloc[0]),
        "weighted_optimal_horizon_days": int(weighted_cells["horizon_days"].iloc[0]),
        "weighted_optimal_Q": int(weighted_cells["Q"].iloc[0]),
        "weighted_optimal_window": int(weighted_cells["window"].iloc[0]),
    }
    row["same_horizon"] = row["average_optimal_horizon_days"] == row["weighted_optimal_horizon_days"]
    row["same_Q"] = row["average_optimal_Q"] == row["weighted_optimal_Q"]
    row["same_window"] = row["average_optimal_window"] == row["weighted_optimal_window"]
    row["same_policy_cell"] = bool(row["same_horizon"] and row["same_Q"] and row["same_window"])

    n_paired: int | None = None
    for metric in DELTA_METRICS:
        if metric not in weighted_cells.columns or metric not in average_cells.columns:
            continue
        mean, low, high, n = _paired_delta_ci(
            weighted_cells,
            average_cells,
            metric,
            draws=draws,
            seed=_stable_seed(background_id, variant, policy, "objective_switch", metric),
        )
        n_paired = n if n_paired is None else min(n_paired, n)
        row[f"delta_{metric}"] = mean
        row[f"delta_{metric}_ci_low"] = low
        row[f"delta_{metric}_ci_high"] = high
        row[f"{metric}_change_status"] = _change_status(mean, low, high, tolerance)

    row["n_paired_seeds"] = int(n_paired or 0)
    row["capacity_tradeoff"] = _direction(row.get("delta_average_utilization", float("nan")), tolerance)
    row["priority_weighted_access_tradeoff"] = _direction(
        row.get("delta_weighted_utilization", float("nan")), tolerance
    )
    c1_direction = _direction(row.get("delta_class_1_percent_serviced", float("nan")), tolerance)
    c2_direction = _direction(row.get("delta_class_2_percent_serviced", float("nan")), tolerance)
    row["class_1_served_rate_direction"] = c1_direction
    row["class_2_served_rate_direction"] = c2_direction
    row["patient_group_tradeoff"] = _tradeoff_category(c1_direction, c2_direction)
    c1_supported = _supported_direction(
        row.get("class_1_percent_serviced_change_status", "missing")
    )
    c2_supported = _supported_direction(
        row.get("class_2_percent_serviced_change_status", "missing")
    )
    row["patient_group_supported_tradeoff"] = _tradeoff_category(
        c1_supported, c2_supported
    )
    return row


def _summary_optima_path(summary_root: Path, variant: str, objective: str) -> Path:
    folder = "h1_10seed_average" if objective == "average_utilization" else "h1_10seed"
    return summary_root / folder / variant / "condition_optima.csv"


def _validation_rows(
    selected_means: pd.DataFrame,
    *,
    summary_root: Path | None,
    variant: str,
) -> pd.DataFrame:
    columns = [
        "background_id",
        "variant",
        "selection_objective",
        "policy",
        "summary_file_found",
        "raw_horizon_days",
        "summary_horizon_days",
        "raw_Q",
        "summary_Q",
        "raw_window",
        "summary_window",
        "same_policy_cell",
    ]
    if summary_root is None:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        path = _summary_optima_path(summary_root, variant, objective)
        subset = selected_means[selected_means["selection_objective"] == objective]
        if not path.exists():
            for record in subset.itertuples(index=False):
                rows.append(
                    {
                        "background_id": record.background_id,
                        "variant": variant,
                        "selection_objective": objective,
                        "policy": record.policy,
                        "summary_file_found": False,
                        "raw_horizon_days": record.selected_horizon_days,
                        "summary_horizon_days": np.nan,
                        "raw_Q": record.selected_Q,
                        "summary_Q": np.nan,
                        "raw_window": record.selected_window,
                        "summary_window": np.nan,
                        "same_policy_cell": False,
                    }
                )
            continue

        summary = pd.read_csv(path).set_index("background_id")
        for record in subset.itertuples(index=False):
            policy = record.policy
            summary_row = summary.loc[record.background_id]
            summary_h = int(summary_row[f"{policy}_horizon_days"])
            summary_q = int(summary_row[f"{policy}_Q"])
            summary_w = int(summary_row[f"{policy}_window"])
            rows.append(
                {
                    "background_id": record.background_id,
                    "variant": variant,
                    "selection_objective": objective,
                    "policy": policy,
                    "summary_file_found": True,
                    "raw_horizon_days": int(record.selected_horizon_days),
                    "summary_horizon_days": summary_h,
                    "raw_Q": int(record.selected_Q),
                    "summary_Q": summary_q,
                    "raw_window": int(record.selected_window),
                    "summary_window": summary_w,
                    "same_policy_cell": bool(
                        int(record.selected_horizon_days) == summary_h
                        and int(record.selected_Q) == summary_q
                        and int(record.selected_window) == summary_w
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _merge_bank(frame: pd.DataFrame, bank: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    bank_columns = [column for column in bank.columns if column != "background_id"]
    collisions = [column for column in bank_columns if column in frame.columns]
    if collisions:
        bank = bank.rename(columns={column: f"scenario_{column}" for column in collisions})
    return frame.merge(bank, on="background_id", how="left", validate="many_to_one")


def process_variant(
    *,
    variant: str,
    raw_root: Path,
    bank: pd.DataFrame,
    output_root: Path,
    summary_root: Path | None,
    draws: int,
    tolerance: float,
    limit_backgrounds: int | None = None,
) -> dict[str, pd.DataFrame]:
    raw_dir = raw_root / variant / "raw"
    shards = sorted(raw_dir.glob("*.csv"))
    if limit_backgrounds is not None:
        shards = shards[:limit_backgrounds]
    if not shards:
        raise FileNotFoundError(f"No raw shard CSVs found under {raw_dir}")

    seed_frames: list[pd.DataFrame] = []
    mean_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []

    for index, shard_path in enumerate(shards, start=1):
        shard = pd.read_csv(shard_path)
        _validate_columns(shard, REQUIRED_RAW_COLUMNS, label=str(shard_path))
        if shard.empty:
            continue
        background_id = str(shard["source_background_id"].iloc[0])

        selections = {
            objective: select_policies(shard, objective)
            for objective in OBJECTIVES
        }

        for objective, policies in selections.items():
            for policy, cells in policies.items():
                seed_frames.append(
                    _seed_rows_for_selection(
                        background_id=background_id,
                        variant=variant,
                        objective=objective,
                        policy=policy,
                        cells=cells,
                    )
                )
                mean_rows.append(
                    _mean_row(
                        background_id=background_id,
                        variant=variant,
                        objective=objective,
                        policy=policy,
                        cells=cells,
                    )
                )

            for comparison, first_policy, second_policy in COMPARISONS:
                delta_rows.append(
                    _comparison_row(
                        background_id=background_id,
                        variant=variant,
                        objective=objective,
                        comparison=comparison,
                        first_policy=first_policy,
                        second_policy=second_policy,
                        first=policies[first_policy],
                        second=policies[second_policy],
                        draws=draws,
                        tolerance=tolerance,
                    )
                )

        for policy in POLICIES:
            switch_rows.append(
                _objective_switch_row(
                    background_id=background_id,
                    variant=variant,
                    policy=policy,
                    average_cells=selections["average_utilization"][policy],
                    weighted_cells=selections["weighted_utilization"][policy],
                    draws=draws,
                    tolerance=tolerance,
                )
            )

        if index % 100 == 0 or index == len(shards):
            print(f"[{variant}] processed {index:,}/{len(shards):,} backgrounds")

    seed_table = pd.concat(seed_frames, ignore_index=True)
    mean_table = pd.DataFrame(mean_rows)
    delta_table = pd.DataFrame(delta_rows)
    switch_table = pd.DataFrame(switch_rows)
    validation_table = _validation_rows(
        mean_table,
        summary_root=summary_root,
        variant=variant,
    )

    seed_table = _merge_bank(seed_table, bank)
    mean_table = _merge_bank(mean_table, bank)
    delta_table = _merge_bank(delta_table, bank)
    switch_table = _merge_bank(switch_table, bank)
    validation_table = _merge_bank(validation_table, bank)

    variant_dir = output_root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    seed_table.to_csv(variant_dir / "selected_policy_seed_outcomes.csv", index=False)
    mean_table.to_csv(variant_dir / "selected_policy_outcomes.csv", index=False)
    delta_table.to_csv(variant_dir / "pairwise_group_deltas.csv", index=False)
    switch_table.to_csv(variant_dir / "objective_switch_deltas.csv", index=False)
    validation_table.to_csv(variant_dir / "selection_validation.csv", index=False)

    validation_found = validation_table["summary_file_found"] if not validation_table.empty else pd.Series(dtype=bool)
    validation_matched = validation_table["same_policy_cell"] if not validation_table.empty else pd.Series(dtype=bool)
    lines = [
        f"# H1 policy-outcome postprocessing: {variant}",
        "",
        f"Backgrounds processed: {mean_table['background_id'].nunique():,}",
        f"Bootstrap draws per paired delta: {draws:,}",
        f"Practical-change tolerance: {tolerance}",
        "",
        "## Output row counts",
        "",
        f"- selected_policy_seed_outcomes.csv: {len(seed_table):,}",
        f"- selected_policy_outcomes.csv: {len(mean_table):,}",
        f"- pairwise_group_deltas.csv: {len(delta_table):,}",
        f"- objective_switch_deltas.csv: {len(switch_table):,}",
        f"- selection_validation.csv: {len(validation_table):,}",
        "",
        "## Interpretation",
        "",
        "- Pairwise deltas are first policy minus second policy.",
        "- Class 1 is the priority group only when Q > 0; otherwise both classes use the general pool.",
        "- weighted_utilization gives Class 1 served rate twice Class 2's policy weight.",
        "- objective_switch_deltas are weighted-optimal minus average-optimal.",
        "",
        "## Selection validation",
        "",
    ]
    if validation_table.empty:
        lines.append("Existing condition_optima.csv files were not checked.")
    else:
        lines.extend(
            [
                f"Summary rows found: {int(validation_found.sum()):,}/{len(validation_table):,}",
                f"Exact policy-cell matches: {int(validation_matched.sum()):,}/{len(validation_table):,}",
                "Weighted-objective mismatches can occur if the supplemental average-objective refinement added new cells that were not present when the older weighted summary was classified.",
            ]
        )
    (variant_dir / "postprocess_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "seed": seed_table,
        "means": mean_table,
        "deltas": delta_table,
        "switch": switch_table,
        "validation": validation_table,
    }


def build_parser() -> argparse.ArgumentParser:
    default_user = os.environ.get("USER", "")
    default_raw = Path("/scratch") / default_user / "h1_short_horizon_reservation_10seed_v2"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=default_raw,
        help="Root containing strict/raw and release/raw shard directories.",
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path("outputs/hypotheses/background_scenarios.csv"),
        help="H1 background scenario bank.",
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("full_run_summaries"),
        help="Root containing h1_10seed and h1_10seed_average; used only for validation.",
    )
    parser.add_argument(
        "--no-summary-validation",
        action="store_true",
        help="Do not compare raw-selected policy cells with existing condition_optima.csv files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("full_run_summaries/h1_policy_outcomes"),
        help="Destination for postprocessed CSVs.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("strict", "release"),
        default=("strict", "release"),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--limit-backgrounds",
        type=int,
        default=None,
        help="Testing aid: process only the first N shards per variant.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_draws < 0:
        raise ValueError("--bootstrap-draws must be >= 0")
    if args.tolerance < 0:
        raise ValueError("--tolerance must be >= 0")

    bank = pd.read_csv(args.bank)
    _validate_columns(bank, {"background_id"}, label=str(args.bank))

    args.output_root.mkdir(parents=True, exist_ok=True)
    combined: dict[str, list[pd.DataFrame]] = {
        "means": [],
        "deltas": [],
        "switch": [],
        "validation": [],
    }
    summary_root = None if args.no_summary_validation else args.summary_root

    for variant in args.variants:
        outputs = process_variant(
            variant=variant,
            raw_root=args.raw_root,
            bank=bank,
            output_root=args.output_root,
            summary_root=summary_root,
            draws=args.bootstrap_draws,
            tolerance=args.tolerance,
            limit_backgrounds=args.limit_backgrounds,
        )
        for key in combined:
            combined[key].append(outputs[key])

    file_names = {
        "means": "combined_selected_policy_outcomes.csv",
        "deltas": "combined_pairwise_group_deltas.csv",
        "switch": "combined_objective_switch_deltas.csv",
        "validation": "combined_selection_validation.csv",
    }
    for key, frames in combined.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                args.output_root / file_names[key],
                index=False,
            )

    print(f"Postprocessed outputs written to: {args.output_root}")


if __name__ == "__main__":
    main()
