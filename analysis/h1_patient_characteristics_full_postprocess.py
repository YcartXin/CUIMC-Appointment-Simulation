"""Postprocess independent evaluation runs for the confirmatory H1 design."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

POLICIES = ("baseline", "horizon_only", "reservation_only", "both_flexible")
OBJECTIVES = ("average_utilization", "priority_weighted_utilization")
COMPARISONS = (
    ("horizon_only_vs_baseline", "horizon_only", "baseline"),
    ("reservation_only_vs_baseline", "reservation_only", "baseline"),
    ("both_flexible_vs_baseline", "both_flexible", "baseline"),
    ("both_flexible_vs_horizon_only", "both_flexible", "horizon_only"),
    ("both_flexible_vs_reservation_only", "both_flexible", "reservation_only"),
    ("reservation_only_vs_horizon_only", "reservation_only", "horizon_only"),
)
PRIMARY_HETEROGENEITY_COMPARISONS = {
    "horizon_only_vs_baseline",
    "reservation_only_vs_baseline",
    "both_flexible_vs_horizon_only",
    "both_flexible_vs_reservation_only",
}

CORE_METRICS = (
    "average_utilization",
    "priority_weighted_utilization",
    "class_1_percent_serviced",
    "class_2_percent_serviced",
    "class_1_slot_utilization",
    "class_2_slot_utilization",
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
    "priority_weighted_utilization",
    "class_1_percent_serviced",
    "class_2_percent_serviced",
    "class_1_slot_utilization",
    "class_2_slot_utilization",
)
HETEROGENEITY_METRICS = (
    "average_utilization",
    "priority_weighted_utilization",
    "class_1_percent_serviced",
    "class_2_percent_serviced",
)


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _bootstrap_matrix(
    values: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, lower and upper CIs for an n-seed by p-metric matrix."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    means = np.nanmean(values, axis=0)
    if draws <= 0 or len(values) <= 1:
        nan = np.full(values.shape[1], np.nan)
        return means, nan, nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    boot = np.nanmean(values[indices, :], axis=1)
    low, high = np.nanquantile(boot, [0.025, 0.975], axis=0)
    return means, low, high


def _effect_status(mean: float, low: float, high: float, tolerance: float) -> str:
    if any(math.isnan(value) for value in (mean, low, high)):
        return "uncertain"
    if mean >= tolerance and low > 0:
        return "meaningful_gain"
    if mean <= -tolerance and high < 0:
        return "meaningful_harm"
    if low >= -tolerance and high <= tolerance:
        return "supported_neutral"
    return "uncertain"


def _reservation_outcome(c1_status: str, c2_status: str) -> str:
    if c1_status == "meaningful_gain" and c2_status == "meaningful_gain":
        return "win_win"
    if c1_status == "meaningful_gain" and c2_status == "supported_neutral":
        return "priority_win_neutral"
    if c1_status == "meaningful_gain" and c2_status == "meaningful_harm":
        return "priority_tradeoff"
    if c1_status in {"meaningful_harm", "supported_neutral"}:
        return "no_demonstrated_priority_benefit"
    return "uncertain"


def _load_eval_cell(frame: pd.DataFrame, horizon: int, q: int, window: int) -> pd.DataFrame:
    selected = frame[
        (frame["horizon_days"] == int(horizon))
        & (frame["Q"] == int(q))
        & (frame["window"] == int(window))
    ].copy()
    if selected.empty:
        raise ValueError(f"Evaluation cell not found: H={horizon}, Q={q}, W={window}")
    return selected.groupby("seed", as_index=False, sort=True).first()


def _selection_seed_rows(
    *,
    background_id: str,
    selection: pd.Series,
    eval_frame: pd.DataFrame,
) -> pd.DataFrame:
    cells = _load_eval_cell(
        eval_frame,
        int(selection["selected_horizon_days"]),
        int(selection["selected_Q"]),
        int(selection["selected_window"]),
    )
    if "background_id" in cells.columns:
        cells = cells.rename(columns={"background_id": "raw_policy_cell_id"})
    cells.insert(0, "policy", selection["policy"])
    cells.insert(0, "selection_objective", selection["selection_objective"])
    cells.insert(0, "background_id", background_id)
    cells["selected_source_stage"] = selection["selected_source_stage"]
    cells["selected_horizon_days"] = int(selection["selected_horizon_days"])
    cells["selected_Q"] = int(selection["selected_Q"])
    cells["selected_window"] = int(selection["selected_window"])
    return cells


def _mean_row(seed_rows: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {
        "background_id": seed_rows["background_id"].iloc[0],
        "selection_objective": seed_rows["selection_objective"].iloc[0],
        "policy": seed_rows["policy"].iloc[0],
        "selected_source_stage": seed_rows["selected_source_stage"].iloc[0],
        "selected_horizon_days": int(seed_rows["selected_horizon_days"].iloc[0]),
        "selected_Q": int(seed_rows["selected_Q"].iloc[0]),
        "selected_window": int(seed_rows["selected_window"].iloc[0]),
        "n_seeds": int(seed_rows["seed"].nunique()),
    }
    for metric in CORE_METRICS:
        if metric in seed_rows:
            row[metric] = float(seed_rows[metric].mean())
            row[f"{metric}_sd"] = float(seed_rows[metric].std(ddof=1))
    return row


def _comparison_outputs(
    *,
    background_id: str,
    objective: str,
    comparison: str,
    first_policy: str,
    second_policy: str,
    first: pd.DataFrame,
    second: pd.DataFrame,
    draws: int,
    tolerance: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    first_idx = first.set_index("seed")
    second_idx = second.set_index("seed")
    seeds = sorted(set(first_idx.index) & set(second_idx.index))
    if not seeds:
        raise ValueError(f"No paired evaluation seeds for {background_id}: {comparison}")
    delta = first_idx.loc[seeds, list(DELTA_METRICS)].to_numpy(float) - second_idx.loc[
        seeds, list(DELTA_METRICS)
    ].to_numpy(float)
    mean, low, high = _bootstrap_matrix(
        delta,
        draws=draws,
        seed=_stable_seed(background_id, objective, comparison),
    )
    row: dict[str, Any] = {
        "background_id": background_id,
        "selection_objective": objective,
        "comparison": comparison,
        "first_policy": first_policy,
        "second_policy": second_policy,
        "n_paired_seeds": len(seeds),
    }
    for index, metric in enumerate(DELTA_METRICS):
        row[f"delta_{metric}"] = float(mean[index])
        row[f"delta_{metric}_ci_low"] = float(low[index])
        row[f"delta_{metric}_ci_high"] = float(high[index])
        row[f"{metric}_status"] = _effect_status(
            float(mean[index]), float(low[index]), float(high[index]), tolerance
        )

    c1_status = row["class_1_percent_serviced_status"]
    c2_status = row["class_2_percent_serviced_status"]
    row["reservation_outcome"] = _reservation_outcome(c1_status, c2_status)
    target_delta = row[f"delta_{objective}"]
    row["target_objective_practical_tie"] = abs(float(target_delta)) < tolerance
    row["baseline_equivalent"] = bool(
        second_policy == "baseline" and float(target_delta) < tolerance
    )

    seed_delta = pd.DataFrame(delta, columns=[f"delta_{metric}" for metric in DELTA_METRICS])
    seed_delta.insert(0, "seed", seeds)
    seed_delta.insert(0, "comparison", comparison)
    seed_delta.insert(0, "selection_objective", objective)
    seed_delta.insert(0, "background_id", background_id)
    return row, seed_delta


def _merge_bank(frame: pd.DataFrame, bank: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    metadata = bank.copy()
    if "horizon_days" in metadata.columns:
        metadata = metadata.rename(columns={"horizon_days": "native_horizon_days"})
    overlaps = (set(frame.columns) & set(metadata.columns)) - {"background_id"}
    if overlaps:
        metadata = metadata.rename(columns={column: f"scenario_{column}" for column in overlaps})
    return frame.merge(metadata, on="background_id", how="left", validate="many_to_one")


def _heterogeneity_outputs(
    pairwise_seed: pd.DataFrame,
    bank: pd.DataFrame,
    *,
    draws: int,
    tolerance: float,
) -> pd.DataFrame:
    metadata = bank[
        [
            "background_id", "patient_characteristic", "class2_reference",
            "contrast_level", "clinic_context_id",
        ]
    ]
    data = pairwise_seed.merge(metadata, on="background_id", how="left", validate="many_to_one")
    data = data[data["comparison"].isin(PRIMARY_HETEROGENEITY_COMPARISONS)].copy()
    join_cols = [
        "patient_characteristic", "class2_reference", "clinic_context_id",
        "selection_objective", "comparison", "seed",
    ]
    controls = data[data["contrast_level"] == "same"].copy()
    control_columns = {f"delta_{metric}": f"same_delta_{metric}" for metric in HETEROGENEITY_METRICS}
    controls = controls[join_cols + list(control_columns)].rename(columns=control_columns)
    treatments = data[data["contrast_level"].isin(["mild", "strong"])].copy()
    matched = treatments.merge(controls, on=join_cols, how="inner", validate="many_to_one")
    for metric in HETEROGENEITY_METRICS:
        matched[f"increment_{metric}"] = (
            matched[f"delta_{metric}"] - matched[f"same_delta_{metric}"]
        )

    group_cols = ["background_id", "selection_objective", "comparison"]
    rows: list[dict[str, Any]] = []
    increment_cols = [f"increment_{metric}" for metric in HETEROGENEITY_METRICS]
    for keys, group in matched.groupby(group_cols, sort=False):
        values = group[increment_cols].to_numpy(float)
        mean, low, high = _bootstrap_matrix(
            values,
            draws=draws,
            seed=_stable_seed(*keys, "heterogeneity"),
        )
        row: dict[str, Any] = dict(zip(group_cols, keys))
        row["n_paired_seeds"] = len(group)
        for index, metric in enumerate(HETEROGENEITY_METRICS):
            row[f"increment_{metric}"] = float(mean[index])
            row[f"increment_{metric}_ci_low"] = float(low[index])
            row[f"increment_{metric}_ci_high"] = float(high[index])
            row[f"increment_{metric}_status"] = _effect_status(
                float(mean[index]), float(low[index]), float(high[index]), tolerance
            )
        rows.append(row)
    return _merge_bank(pd.DataFrame(rows), bank)


def process(
    *,
    raw_root: Path,
    bank_path: Path,
    output_root: Path,
    draws: int,
    tolerance: float,
    class2_tolerance: float,
    expected_evaluation_seeds: int,
) -> None:
    bank = pd.read_csv(bank_path)
    selection_dir = raw_root / "selection"
    selections = pd.read_csv(selection_dir / "selected_cells.csv")
    constrained = pd.read_csv(selection_dir / "constrained_priority_cells.csv")
    eval_dir = raw_root / "evaluation" / "raw"
    shards = sorted(eval_dir.glob("*.csv"))
    if not shards:
        raise FileNotFoundError(f"No evaluation shards found under {eval_dir}")

    selected_seed_frames: list[pd.DataFrame] = []
    selected_mean_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    pairwise_seed_frames: list[pd.DataFrame] = []
    switch_rows: list[dict[str, Any]] = []
    constrained_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for number, path in enumerate(shards, start=1):
        evaluation = pd.read_csv(path)
        background_id = str(evaluation["source_background_id"].iloc[0])
        background_selections = selections[selections["background_id"] == background_id]
        if len(background_selections) != len(OBJECTIVES) * len(POLICIES):
            raise ValueError(f"Incomplete selections for {background_id}")

        policy_seed: dict[tuple[str, str], pd.DataFrame] = {}
        for _, selection in background_selections.iterrows():
            seed_rows = _selection_seed_rows(
                background_id=background_id,
                selection=selection,
                eval_frame=evaluation,
            )
            key = (str(selection["selection_objective"]), str(selection["policy"]))
            found_seeds = int(seed_rows["seed"].nunique())
            if found_seeds != expected_evaluation_seeds:
                raise ValueError(
                    f"Incomplete evaluation for {background_id}, {key}: "
                    f"expected {expected_evaluation_seeds} seeds, found {found_seeds}"
                )
            policy_seed[key] = seed_rows
            selected_seed_frames.append(seed_rows)
            selected_mean_rows.append(_mean_row(seed_rows))
            validation_rows.append(
                {
                    "background_id": background_id,
                    "selection_objective": key[0],
                    "policy": key[1],
                    "selected_horizon_days": int(selection["selected_horizon_days"]),
                    "selected_Q": int(selection["selected_Q"]),
                    "selected_window": int(selection["selected_window"]),
                    "evaluation_seed_count": int(seed_rows["seed"].nunique()),
                    "evaluation_cell_found": True,
                }
            )

        for objective in OBJECTIVES:
            for comparison, first_policy, second_policy in COMPARISONS:
                summary, seed_delta = _comparison_outputs(
                    background_id=background_id,
                    objective=objective,
                    comparison=comparison,
                    first_policy=first_policy,
                    second_policy=second_policy,
                    first=policy_seed[(objective, first_policy)],
                    second=policy_seed[(objective, second_policy)],
                    draws=draws,
                    tolerance=tolerance,
                )
                pairwise_rows.append(summary)
                pairwise_seed_frames.append(seed_delta)

        for policy in POLICIES:
            priority = policy_seed[("priority_weighted_utilization", policy)]
            average = policy_seed[("average_utilization", policy)]
            summary, _ = _comparison_outputs(
                background_id=background_id,
                objective="priority_weighted_utilization",
                comparison="priority_optimal_minus_average_optimal",
                first_policy=policy,
                second_policy=policy,
                first=priority,
                second=average,
                draws=draws,
                tolerance=tolerance,
            )
            summary["policy"] = policy
            summary["average_optimal_horizon_days"] = int(average["selected_horizon_days"].iloc[0])
            summary["average_optimal_Q"] = int(average["selected_Q"].iloc[0])
            summary["average_optimal_window"] = int(average["selected_window"].iloc[0])
            summary["priority_optimal_horizon_days"] = int(priority["selected_horizon_days"].iloc[0])
            summary["priority_optimal_Q"] = int(priority["selected_Q"].iloc[0])
            summary["priority_optimal_window"] = int(priority["selected_window"].iloc[0])
            summary["same_policy_cell"] = bool(
                summary["average_optimal_horizon_days"] == summary["priority_optimal_horizon_days"]
                and summary["average_optimal_Q"] == summary["priority_optimal_Q"]
                and summary["average_optimal_window"] == summary["priority_optimal_window"]
            )
            switch_rows.append(summary)

        crow = constrained[constrained["background_id"] == background_id]
        if len(crow) != 1:
            raise ValueError(f"Missing constrained selection for {background_id}")
        crow = crow.iloc[0]
        constrained_seed = _load_eval_cell(
            evaluation,
            int(crow["selected_horizon_days"]),
            int(crow["selected_Q"]),
            int(crow["selected_window"]),
        )
        baseline_seed = policy_seed[("average_utilization", "baseline")]
        avg_optimal_seed = policy_seed[("average_utilization", "both_flexible")]
        constrained_mean = constrained_seed[list(CORE_METRICS)].mean(numeric_only=True)
        baseline_mean = baseline_seed[list(CORE_METRICS)].mean(numeric_only=True)
        avg_optimal_mean = avg_optimal_seed[list(CORE_METRICS)].mean(numeric_only=True)
        constrained_rows.append(
            {
                "background_id": background_id,
                "selected_policy": crow["selected_policy"],
                "search_constraints_feasible": bool(crow["search_constraints_feasible"]),
                "search_failure_reason": crow.get("search_failure_reason", ""),
                "selected_horizon_days": int(crow["selected_horizon_days"]),
                "selected_Q": int(crow["selected_Q"]),
                "selected_window": int(crow["selected_window"]),
                "n_seeds": int(constrained_seed["seed"].nunique()),
                "average_utilization": float(constrained_mean["average_utilization"]),
                "priority_weighted_utilization": float(constrained_mean["priority_weighted_utilization"]),
                "class_1_percent_serviced": float(constrained_mean["class_1_percent_serviced"]),
                "class_2_percent_serviced": float(constrained_mean["class_2_percent_serviced"]),
                "delta_average_utilization_vs_average_optimum": float(
                    constrained_mean["average_utilization"] - avg_optimal_mean["average_utilization"]
                ),
                "delta_class_2_percent_serviced_vs_baseline": float(
                    constrained_mean["class_2_percent_serviced"] - baseline_mean["class_2_percent_serviced"]
                ),
                "delta_priority_weighted_utilization_vs_baseline": float(
                    constrained_mean["priority_weighted_utilization"]
                    - baseline_mean["priority_weighted_utilization"]
                ),
                "evaluation_average_constraint_met": bool(
                    constrained_mean["average_utilization"]
                    >= avg_optimal_mean["average_utilization"] - tolerance
                ),
                "evaluation_class_2_constraint_met": bool(
                    constrained_mean["class_2_percent_serviced"]
                    >= baseline_mean["class_2_percent_serviced"] - class2_tolerance
                ),
                "baseline_equivalent": bool(
                    constrained_mean["priority_weighted_utilization"]
                    - baseline_mean["priority_weighted_utilization"]
                    < tolerance
                ),
            }
        )

        if number % 100 == 0 or number == len(shards):
            print(f"Processed {number:,}/{len(shards):,} evaluation shards")

    selected_seed = pd.concat(selected_seed_frames, ignore_index=True)
    selected_means = pd.DataFrame(selected_mean_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise_seed = pd.concat(pairwise_seed_frames, ignore_index=True)
    switch = pd.DataFrame(switch_rows)
    constrained_output = pd.DataFrame(constrained_rows)
    validation = pd.DataFrame(validation_rows)

    heterogeneity = _heterogeneity_outputs(
        pairwise_seed,
        bank,
        draws=draws,
        tolerance=tolerance,
    )

    selected_seed = _merge_bank(selected_seed, bank)
    selected_means = _merge_bank(selected_means, bank)
    pairwise = _merge_bank(pairwise, bank)
    switch = _merge_bank(switch, bank)
    constrained_output = _merge_bank(constrained_output, bank)
    validation = _merge_bank(validation, bank)

    output_root.mkdir(parents=True, exist_ok=True)
    selected_seed.to_csv(output_root / "selected_policy_seed_outcomes.csv", index=False)
    selected_means.to_csv(output_root / "selected_policy_outcomes.csv", index=False)
    pairwise.to_csv(output_root / "pairwise_group_deltas.csv", index=False)
    switch.to_csv(output_root / "objective_switch_deltas.csv", index=False)
    constrained_output.to_csv(output_root / "constrained_priority_outcomes.csv", index=False)
    heterogeneity.to_csv(output_root / "heterogeneity_increment_deltas.csv", index=False)
    validation.to_csv(output_root / "selection_validation.csv", index=False)

    summary = [
        "# Controlled horizon-reservation postprocessing",
        "",
        f"Backgrounds: {selected_means['background_id'].nunique():,}",
        f"Bootstrap draws: {draws:,}",
        f"Practical-effect threshold: {tolerance}",
        f"Class 2 constrained-priority tolerance: {class2_tolerance}",
        "",
        "## Output rows",
        "",
        f"- selected_policy_seed_outcomes.csv: {len(selected_seed):,}",
        f"- selected_policy_outcomes.csv: {len(selected_means):,}",
        f"- pairwise_group_deltas.csv: {len(pairwise):,}",
        f"- objective_switch_deltas.csv: {len(switch):,}",
        f"- constrained_priority_outcomes.csv: {len(constrained_output):,}",
        f"- heterogeneity_increment_deltas.csv: {len(heterogeneity):,}",
        f"- selection_validation.csv: {len(validation):,}",
        "",
        "Neutral means the entire 95% confidence interval lies within the practical-equivalence band.",
        "Search seeds selected policy cells; independent evaluation seeds produced all reported effects.",
    ]
    (output_root / "postprocess_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Final outputs: {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--class2-tolerance", type=float, default=0.01)
    parser.add_argument("--expected-evaluation-seeds", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    process(
        raw_root=args.raw_root,
        bank_path=args.bank,
        output_root=args.output_root,
        draws=args.bootstrap_draws,
        tolerance=args.tolerance,
        class2_tolerance=args.class2_tolerance,
        expected_evaluation_seeds=args.expected_evaluation_seeds,
    )


if __name__ == "__main__":
    main()
