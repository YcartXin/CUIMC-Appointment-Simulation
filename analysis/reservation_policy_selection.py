from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


SCENARIO_COLUMNS = [
    "scenario_id",
    "arrival_rate_class_1",
    "arrival_rate_class_2",
    "tau_1",
    "tau_2",
    "post_threshold_balking_rate",
    "w1",
    "w2",
]

OBJECTIVE_SPECS = {
    "Obj_util_norm": "maximize",
    "Obj_service_norm": "maximize",
    "T_wait_offered": "minimize",
}

DIAGNOSTIC_COLUMNS = [
    "class_1_service_rate",
    "class_2_service_rate",
    "class_1_no_offer_rate",
    "class_2_no_offer_rate",
    "class_1_avg_offered_wait",
    "class_2_avg_offered_wait",
]

PAIRED_COLUMNS = [
    "A1",
    "A2",
    "Y1",
    "Y2",
    "offered_1",
    "offered_2",
    "sum_tau_offered_1",
    "sum_tau_offered_2",
    "Obj_util_raw",
    "Obj_util_norm",
    "Obj_service_raw",
    "Obj_service_norm",
    "T_wait_offered",
    *DIAGNOSTIC_COLUMNS,
]


def pair_with_fcfs(run_level: pd.DataFrame) -> pd.DataFrame:
    """Attach the unique same-scenario, same-seed, same-weight Q=0 row."""
    pair_keys = [*SCENARIO_COLUMNS, "seed"]
    fcfs = run_level.loc[run_level["Q"] == 0, pair_keys + PAIRED_COLUMNS].copy()
    if fcfs.duplicated(pair_keys).any():
        raise ValueError("Duplicate FCFS rows found for at least one pairing key.")
    expected = run_level[pair_keys].drop_duplicates()
    available = fcfs[pair_keys].drop_duplicates()
    missing = expected.merge(available, on=pair_keys, how="left", indicator=True)
    if (missing["_merge"] != "both").any():
        raise ValueError("At least one run has no matching FCFS row.")

    fcfs = fcfs.rename(
        columns={column: f"fcfs_{column}" for column in PAIRED_COLUMNS}
    )
    paired = run_level.merge(fcfs, on=pair_keys, how="left", validate="many_to_one")
    for column in PAIRED_COLUMNS:
        paired[f"delta_{column}"] = paired[column] - paired[f"fcfs_{column}"]
    return paired


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    draws: int = 4000,
    seed: int = 20260623,
) -> tuple[float, float, float, int]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(data.mean())
    if data.size == 1:
        return mean, mean, mean, 1
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(draws, data.size))
    means = data[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high), int(data.size)


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(
        hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest(),
        "big",
    )


def summarize_by_q(paired: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Obj_util_raw",
        "Obj_util_norm",
        "Obj_service_raw",
        "Obj_service_norm",
        "T_wait_offered",
        *DIAGNOSTIC_COLUMNS,
    ]
    rows: list[dict[str, object]] = []
    for keys, group in paired.groupby([*SCENARIO_COLUMNS, "Q"], sort=True):
        row = dict(zip([*SCENARIO_COLUMNS, "Q"], keys))
        for metric in metrics:
            mean, low, high, n = bootstrap_mean_ci(
                group[metric].to_numpy(),
                seed=_stable_seed(*keys, metric, "level"),
            )
            delta_mean, delta_low, delta_high, delta_n = bootstrap_mean_ci(
                group[f"delta_{metric}"].to_numpy(),
                seed=_stable_seed(*keys, metric, "delta"),
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


def near_tie_values(
    values_by_q: pd.Series,
    *,
    direction: str,
    relative_tolerance: float = 0.01,
) -> tuple[list[int], list[int], float]:
    finite = values_by_q[np.isfinite(values_by_q)]
    if finite.empty:
        return [], [], math.nan
    if direction == "maximize":
        best = float(finite.max())
        exact = finite[np.isclose(finite, best, rtol=0, atol=1e-12)].index
        threshold = best - relative_tolerance * abs(best)
        near = finite[finite >= threshold - 1e-12].index
    elif direction == "minimize":
        best = float(finite.min())
        exact = finite[np.isclose(finite, best, rtol=0, atol=1e-12)].index
        threshold = best + relative_tolerance * abs(best)
        near = finite[finite <= threshold + 1e-12].index
    else:
        raise ValueError("direction must be 'maximize' or 'minimize'.")
    return sorted(map(int, exact)), sorted(map(int, near)), best


def contiguous_q_ranges(
    q_values: Iterable[int],
    *,
    tested_q_values: Sequence[int],
) -> list[dict[str, object]]:
    selected = set(map(int, q_values))
    tested = list(map(int, tested_q_values))
    ranges: list[list[int]] = []
    current: list[int] = []
    for q in tested:
        if q in selected:
            current.append(q)
        elif current:
            ranges.append(current)
            current = []
    if current:
        ranges.append(current)
    return [
        {
            "range_start": values[0],
            "range_end": values[-1],
            "q_values": ",".join(map(str, values)),
            "range_label": (
                str(values[0])
                if len(values) == 1
                else f"{values[0]}-{values[-1]} [{','.join(map(str, values))}]"
            ),
        }
        for values in ranges
    ]


def build_selection_summaries(
    q_summary: pd.DataFrame,
    *,
    tested_q_values: Sequence[int],
    relative_tolerance: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best_rows: list[dict[str, object]] = []
    range_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []

    for scenario_keys, group in q_summary.groupby(SCENARIO_COLUMNS, sort=True):
        scenario = dict(zip(SCENARIO_COLUMNS, scenario_keys))
        by_q = group.set_index("Q").sort_index()
        scenario_row: dict[str, object] = {**scenario}
        fcfs = by_q.loc[0]

        for objective, direction in OBJECTIVE_SPECS.items():
            exact, near, best_value = near_tie_values(
                by_q[f"{objective}_mean"],
                direction=direction,
                relative_tolerance=relative_tolerance,
            )
            strict = by_q.loc[by_q.index > 0, f"{objective}_mean"]
            strict_exact, strict_near, strict_best = near_tie_values(
                strict,
                direction=direction,
                relative_tolerance=relative_tolerance,
            )
            strict_ranges = contiguous_q_ranges(
                strict_near,
                tested_q_values=tested_q_values,
            )
            all_ranges = contiguous_q_ranges(
                near,
                tested_q_values=tested_q_values,
            )

            composition_flags = []
            for q in near:
                row = by_q.loc[q]
                composition_flags.append(
                    bool(
                        row["T_wait_offered_mean"]
                        < fcfs["T_wait_offered_mean"]
                        and row["class_1_no_offer_rate_mean"]
                        > fcfs["class_1_no_offer_rate_mean"]
                        and row["class_2_no_offer_rate_mean"]
                        > fcfs["class_2_no_offer_rate_mean"]
                    )
                )

            prefix = {
                "Obj_util_norm": "util",
                "Obj_service_norm": "service",
                "T_wait_offered": "wait",
            }[objective]
            scenario_row[f"{prefix}_exact_best_q_values"] = ",".join(map(str, exact))
            scenario_row[f"{prefix}_near_tie_q_values"] = ",".join(map(str, near))
            scenario_row[f"{prefix}_near_tie_q_ranges"] = "; ".join(
                item["range_label"] for item in all_ranges
            )
            scenario_row[f"{prefix}_best_value"] = best_value
            scenario_row[f"{prefix}_fcfs_value"] = fcfs[f"{objective}_mean"]
            scenario_row[f"{prefix}_best_strict_value"] = strict_best
            scenario_row[f"{prefix}_best_strict_delta_vs_fcfs"] = (
                strict_best - fcfs[f"{objective}_mean"]
            )
            scenario_row[f"{prefix}_composition_effect_likely"] = any(
                composition_flags
            )

            best_row = {
                **scenario,
                "objective_name": objective,
                "direction": direction,
                "near_tie_tolerance": relative_tolerance,
                "fcfs_mean": fcfs[f"{objective}_mean"],
                "exact_best_q_values": ",".join(map(str, exact)),
                "exact_best_value": best_value,
                "near_tie_q_values": ",".join(map(str, near)),
                "near_tie_q_ranges": "; ".join(
                    item["range_label"] for item in all_ranges
                ),
                "best_strict_q_values": ",".join(map(str, strict_exact)),
                "best_strict_near_tie_q_values": ",".join(map(str, strict_near)),
                "best_strict_q_ranges": "; ".join(
                    item["range_label"] for item in strict_ranges
                ),
                "best_strict_value": strict_best,
                "best_strict_delta_vs_fcfs": (
                    strict_best - fcfs[f"{objective}_mean"]
                ),
                "composition_effect_likely": any(composition_flags),
            }

            tradeoff_qs = strict_near or near
            tradeoff = by_q.loc[tradeoff_qs]
            for metric in (
                "class_1_service_rate",
                "class_2_service_rate",
                "class_1_no_offer_rate",
                "class_2_no_offer_rate",
            ):
                deltas = tradeoff[f"delta_{metric}_mean"]
                best_row[f"{metric}_delta_min"] = float(deltas.min())
                best_row[f"{metric}_delta_max"] = float(deltas.max())
            best_rows.append(best_row)

            for index, item in enumerate(all_ranges, start=1):
                range_rows.append(
                    {
                        **scenario,
                        "objective_name": objective,
                        "direction": direction,
                        "near_tie_tolerance": relative_tolerance,
                        "range_index": index,
                        **item,
                    }
                )

        scenario_rows.append(scenario_row)

    return (
        pd.DataFrame(scenario_rows),
        pd.DataFrame(best_rows),
        pd.DataFrame(range_rows),
    )
