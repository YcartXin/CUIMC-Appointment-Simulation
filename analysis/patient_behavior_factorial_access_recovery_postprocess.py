#!/usr/bin/env python3
"""Post-process the frozen 3x3 access-recovery experiment on 10 evaluation seeds.

Important: this script performs NO re-selection and NO re-optimization.
All policy cells are frozen from the 5-search-seed stage. The independent
evaluation seeds are used only to estimate out-of-sample effects and uncertainty.

Comparisons:
1. Each policy's original average-utilization optimum vs matched no-policy baseline.
2. Each frozen candidate vs matched no-policy baseline.
3. Each frozen candidate vs that policy's original average-utilization optimum.

Inference:
- 10 paired independent evaluation seeds (default).
- 2,000 paired percentile-bootstrap resamples (default).
- Practical band = +/-0.005.
- Win-win: both served-rate means >= +0.005 and both 95% CIs > 0.
- C1 win / C2 neutral: C1 mean >= +0.005 with CI > 0, and the entire
  C2 95% CI lies inside [-0.005, +0.005].

Outputs are written under:
    <output-dir>/access_recovery_optimization/evaluation_release/
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.patient_behavior_factorial import EVALUATION_SEED_POOL


POLICIES = ("horizon_only", "reservation_only", "both_flexible")
CANDIDATE_TYPES = (
    "access_recovery",
    "best_win_win",
    "c1_win_c2_neutral_if_better",
)

METRICS = (
    "average_utilization",
    "priority_weighted_utilization",
    "class_1_percent_serviced",
    "class_2_percent_serviced",
)

CELL_COLS = ("horizon_days", "Q", "window")
BAND = 0.005

LEVEL_MAP = {
    "LOW": "low",
    "MEDIUM": "medium",
    "HIGH": "high",
}

BG_RE = re.compile(
    r"^PBF_NS_(LOW|MEDIUM|HIGH)_BK_(LOW|MEDIUM|HIGH)_C\d+$"
)


def _stable_seed(*parts: object, base: int) -> int:
    text = "|".join([str(base), *map(str, parts)])
    digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("Bootstrap values must be a non-empty 1D array")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot = x[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def _parse_background(background_id: str) -> tuple[str, str]:
    m = BG_RE.match(background_id)
    if not m:
        raise ValueError(
            f"Could not parse no-show/balking levels from {background_id}"
        )
    return LEVEL_MAP[m.group(1)], LEVEL_MAP[m.group(2)]


def _context_frame(bank: pd.DataFrame) -> pd.DataFrame:
    x = bank.copy()
    lam = x["lambda_1"].astype(float) + x["lambda_2"].astype(float)

    x["rho"] = lam / x["slots_per_day"].astype(float)
    x["class1_share"] = np.where(
        lam > 0,
        x["lambda_1"].astype(float) / lam,
        np.nan,
    )
    x["capacity"] = x["slots_per_day"].astype(int)

    parsed = x["background_id"].map(_parse_background)
    x["noshow_level"] = parsed.map(lambda z: z[0])
    x["balk_level"] = parsed.map(lambda z: z[1])

    return x[
        [
            "background_id",
            "rho",
            "class1_share",
            "capacity",
            "noshow_level",
            "balk_level",
        ]
    ].copy()


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def _cell_from_row(
    row: pd.Series,
    *,
    h: str,
    q: str,
    w: str,
) -> tuple[int, int, int]:
    return int(row[h]), int(row[q]), int(row[w])


def _dedupe_eval(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [*CELL_COLS, "seed"]
    required = set(keys) | set(METRICS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Evaluation shard missing columns: {missing}")

    x = frame.copy()

    # Repeated cleanup jobs should normally be prevented by _filter_pending.
    # If exact duplicate keys nevertheless exist, require identical metrics.
    dup = x[x.duplicated(keys, keep=False)].copy()
    if not dup.empty:
        for _, g in dup.groupby(keys, dropna=False):
            for metric in METRICS:
                vals = g[metric].dropna().to_numpy(dtype=float)
                if len(vals) and not np.allclose(
                    vals,
                    vals[0],
                    rtol=0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        "Conflicting duplicate evaluation rows for "
                        f"{tuple(g.iloc[0][keys])}"
                    )

    return (
        x.sort_values(keys, kind="stable")
        .drop_duplicates(keys, keep="last")
        .copy()
    )


def _cell_rows(
    evaluation: pd.DataFrame,
    cell: tuple[int, int, int],
    seeds: tuple[int, ...],
    *,
    label: str,
) -> pd.DataFrame:
    h, q, w = cell

    x = evaluation[
        (evaluation["horizon_days"] == h)
        & (evaluation["Q"] == q)
        & (evaluation["window"] == w)
        & (evaluation["seed"].isin(seeds))
    ].copy()

    if x["seed"].nunique() != len(seeds):
        observed = sorted(x["seed"].unique().tolist())
        missing = sorted(set(seeds) - set(observed))
        raise ValueError(
            f"{label}: expected {len(seeds)} evaluation seeds for "
            f"cell {(h, q, w)}, found {len(observed)}; missing={missing}"
        )

    return (
        x.sort_values("seed", kind="stable")
        .drop_duplicates(["seed"], keep="last")
        [["seed", *METRICS]]
        .reset_index(drop=True)
    )


def _point_summary(
    rows: pd.DataFrame,
    *,
    background_id: str,
    policy: str,
    point_type: str,
    cell: tuple[int, int, int],
) -> dict[str, object]:
    out: dict[str, object] = {
        "background_id": background_id,
        "policy": policy,
        "point_type": point_type,
        "horizon_days": cell[0],
        "Q": cell[1],
        "window": cell[2],
        "n_eval_seeds": int(rows["seed"].nunique()),
    }
    for metric in METRICS:
        out[metric] = float(rows[metric].mean())
    return out


def _paired_record(
    *,
    lhs: pd.DataFrame,
    rhs: pd.DataFrame,
    background_id: str,
    policy: str,
    point_type: str,
    comparison: str,
    lhs_cell: tuple[int, int, int],
    rhs_cell: tuple[int, int, int],
    n_boot: int,
    bootstrap_seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    merged = lhs.merge(
        rhs,
        on="seed",
        suffixes=("_lhs", "_rhs"),
        validate="one_to_one",
    )

    rec: dict[str, object] = {
        "background_id": background_id,
        "policy": policy,
        "point_type": point_type,
        "comparison": comparison,
        "lhs_horizon_days": lhs_cell[0],
        "lhs_Q": lhs_cell[1],
        "lhs_window": lhs_cell[2],
        "rhs_horizon_days": rhs_cell[0],
        "rhs_Q": rhs_cell[1],
        "rhs_window": rhs_cell[2],
        "n_eval_seeds": int(len(merged)),
    }

    seed_rows: list[dict[str, object]] = []

    for metric in METRICS:
        delta = (
            merged[f"{metric}_lhs"].to_numpy(dtype=float)
            - merged[f"{metric}_rhs"].to_numpy(dtype=float)
        )

        rec[f"delta_{metric}"] = float(delta.mean())

        lo, hi = _bootstrap_mean_ci(
            delta,
            n_boot=n_boot,
            seed=_stable_seed(
                background_id,
                policy,
                point_type,
                comparison,
                metric,
                base=bootstrap_seed,
            ),
        )
        rec[f"delta_{metric}_ci_low"] = lo
        rec[f"delta_{metric}_ci_high"] = hi

        for seed, value in zip(merged["seed"], delta):
            seed_rows.append(
                {
                    "background_id": background_id,
                    "policy": policy,
                    "point_type": point_type,
                    "comparison": comparison,
                    "seed": int(seed),
                    "metric": metric,
                    "delta": float(value),
                }
            )

    _add_statuses(rec)
    return rec, seed_rows


def _add_statuses(rec: dict[str, object]) -> None:
    u = float(rec["delta_average_utilization"])
    ulo = float(rec["delta_average_utilization_ci_low"])
    uhi = float(rec["delta_average_utilization_ci_high"])

    if u >= BAND and ulo > 0:
        util_status = "Meaningful increase"
    elif u <= -BAND and uhi < 0:
        util_status = "Meaningful decrease"
    elif ulo >= -BAND and uhi <= BAND:
        util_status = "Supported neutral"
    else:
        util_status = "Uncertain"

    c1 = float(rec["delta_class_1_percent_serviced"])
    c1lo = float(rec["delta_class_1_percent_serviced_ci_low"])
    c1hi = float(rec["delta_class_1_percent_serviced_ci_high"])

    c2 = float(rec["delta_class_2_percent_serviced"])
    c2lo = float(rec["delta_class_2_percent_serviced_ci_low"])
    c2hi = float(rec["delta_class_2_percent_serviced_ci_high"])

    c1gain = c1 >= BAND and c1lo > 0
    c2gain = c2 >= BAND and c2lo > 0
    c1harm = c1 <= -BAND and c1hi < 0
    c2harm = c2 <= -BAND and c2hi < 0
    c1neutral = c1lo >= -BAND and c1hi <= BAND
    c2neutral = c2lo >= -BAND and c2hi <= BAND

    if c1gain and c2gain:
        access_status = "Win-win"
    elif c1gain and c2neutral:
        access_status = "C1 win / C2 neutral"
    elif c2gain and c1neutral:
        access_status = "C2 win / C1 neutral"
    elif c1gain and c2harm:
        access_status = "C1 gain / C2 harm"
    elif c2gain and c1harm:
        access_status = "C2 gain / C1 harm"
    elif c1neutral and c2neutral:
        access_status = "Both neutral"
    else:
        access_status = "Uncertain / mixed"

    rec["utilization_status"] = util_status
    rec["access_status"] = access_status

    rec["c1_supported_gain"] = bool(c1gain)
    rec["c2_supported_gain"] = bool(c2gain)
    rec["c1_supported_neutral"] = bool(c1neutral)
    rec["c2_supported_neutral"] = bool(c2neutral)

    # Relevant for candidate-vs-average-optimum comparisons.
    rec["utilization_within_0_5pp_budget_mean"] = bool(
        u >= -BAND - 1e-12
    )
    rec["utilization_budget_supported_by_ci"] = bool(
        ulo >= -BAND - 1e-12
    )


def _ranges(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "full", frame
    yield "headline_rho_le_2_5", frame[frame["rho"] <= 2.5].copy()
    yield "stress_rho_3", frame[np.isclose(frame["rho"], 3.0)].copy()


def _access_summary(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for range_name, subset in _ranges(frame):
        if subset.empty:
            continue

        for keys, g in subset.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)

            row = dict(zip(group_cols, keys))
            row["analysis_range"] = range_name
            row["backgrounds"] = int(len(g))

            for metric in (
                "average_utilization",
                "class_1_percent_serviced",
                "class_2_percent_serviced",
            ):
                col = f"delta_{metric}"
                row[f"median_{col}"] = float(g[col].median())
                row[f"mean_{col}"] = float(g[col].mean())

            row["share_c1_positive"] = float(
                (g["delta_class_1_percent_serviced"] > 0).mean()
            )
            row["share_c1_ge_0_5pp"] = float(
                (
                    g["delta_class_1_percent_serviced"]
                    >= BAND
                ).mean()
            )
            row["share_c1_supported_gain"] = float(
                g["c1_supported_gain"].mean()
            )
            row["share_util_within_0_5pp_budget_mean"] = float(
                g["utilization_within_0_5pp_budget_mean"].mean()
            )
            row["share_util_budget_supported_by_ci"] = float(
                g["utilization_budget_supported_by_ci"].mean()
            )

            rows.append(row)

    return pd.DataFrame(rows)


def _favorable_validation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for range_name, subset in _ranges(frame):
        if subset.empty:
            continue

        for keys, g in subset.groupby(
            ["policy", "point_type"],
            dropna=False,
        ):
            policy, point_type = keys

            target = (
                "Win-win"
                if point_type == "best_win_win"
                else "C1 win / C2 neutral"
            )

            strict = g["access_status"].eq(target)
            favorable = g["access_status"].isin(
                ["Win-win", "C1 win / C2 neutral"]
            )

            rows.append(
                {
                    "analysis_range": range_name,
                    "policy": policy,
                    "point_type": point_type,
                    "frozen_candidates": int(len(g)),
                    "strict_target_validated": int(strict.sum()),
                    "strict_target_share": float(strict.mean()),
                    "favorable_or_better": int(favorable.sum()),
                    "favorable_or_better_share": float(
                        favorable.mean()
                    ),
                    "final_win_win": int(
                        g["access_status"].eq("Win-win").sum()
                    ),
                    "final_c1_win_c2_neutral": int(
                        g["access_status"]
                        .eq("C1 win / C2 neutral")
                        .sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260828,
    )
    args = parser.parse_args()

    if not 1 <= args.n_seeds <= len(EVALUATION_SEED_POOL):
        raise ValueError("Invalid --n-seeds")
    if args.bootstrap < 100:
        raise ValueError("--bootstrap should be at least 100")

    seeds = tuple(EVALUATION_SEED_POOL[: args.n_seeds])

    root = args.output_dir.resolve()
    bank = pd.read_csv(args.bank)
    context = _context_frame(bank)

    selected_path = root / "selection" / "selected_cells.csv"
    candidates_path = (
        root
        / "access_recovery_optimization"
        / "access_recovery_candidates.csv"
    )
    raw_dir = root / "evaluation" / "raw"

    for path in (selected_path, candidates_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)

    selected = pd.read_csv(selected_path)
    selected = selected[
        selected["selection_objective"].eq("average_utilization")
    ].copy()

    candidates = pd.read_csv(candidates_path)
    candidates["candidate_exists"] = _bool_series(
        candidates["candidate_exists"]
    )

    selected_idx = selected.set_index(
        ["background_id", "policy"],
        drop=False,
    )

    points: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    seed_deltas: list[dict[str, object]] = []

    for i, ctx in context.iterrows():
        bg = str(ctx["background_id"])
        path = raw_dir / f"{bg}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing evaluation shard: {path}"
            )

        evaluation = _dedupe_eval(pd.read_csv(path))

        # Baseline reference.
        bkey = (bg, "baseline")
        if bkey not in selected_idx.index:
            raise KeyError(f"Missing baseline selection for {bg}")

        brow = selected_idx.loc[bkey]
        if isinstance(brow, pd.DataFrame):
            if len(brow) != 1:
                raise ValueError(
                    f"Duplicate baseline selections for {bg}"
                )
            brow = brow.iloc[0]

        baseline_cell = _cell_from_row(
            brow,
            h="selected_horizon_days",
            q="selected_Q",
            w="selected_window",
        )
        baseline_rows = _cell_rows(
            evaluation,
            baseline_cell,
            seeds,
            label=f"{bg} baseline",
        )

        points.append(
            _point_summary(
                baseline_rows,
                background_id=bg,
                policy="baseline",
                point_type="baseline",
                cell=baseline_cell,
            )
        )

        for policy in POLICIES:
            pkey = (bg, policy)
            if pkey not in selected_idx.index:
                raise KeyError(
                    f"Missing average-utilization selection for {pkey}"
                )

            prow = selected_idx.loc[pkey]
            if isinstance(prow, pd.DataFrame):
                if len(prow) != 1:
                    raise ValueError(
                        f"Duplicate selections for {pkey}"
                    )
                prow = prow.iloc[0]

            avg_cell = _cell_from_row(
                prow,
                h="selected_horizon_days",
                q="selected_Q",
                w="selected_window",
            )
            avg_rows = _cell_rows(
                evaluation,
                avg_cell,
                seeds,
                label=f"{bg} {policy} average optimum",
            )

            points.append(
                _point_summary(
                    avg_rows,
                    background_id=bg,
                    policy=policy,
                    point_type="avg_util_optimum",
                    cell=avg_cell,
                )
            )

            rec, srows = _paired_record(
                lhs=avg_rows,
                rhs=baseline_rows,
                background_id=bg,
                policy=policy,
                point_type="avg_util_optimum",
                comparison="vs_baseline",
                lhs_cell=avg_cell,
                rhs_cell=baseline_cell,
                n_boot=args.bootstrap,
                bootstrap_seed=args.bootstrap_seed,
            )
            paired.append(rec)
            seed_deltas.extend(srows)

            cand = candidates[
                (candidates["background_id"].eq(bg))
                & (candidates["policy"].eq(policy))
                & (candidates["candidate_exists"])
                & (
                    candidates["candidate_type"]
                    .isin(CANDIDATE_TYPES)
                )
            ].copy()

            if cand["candidate_type"].duplicated().any():
                raise ValueError(
                    f"Duplicate candidate types for {(bg, policy)}"
                )

            for _, crow in cand.iterrows():
                point_type = str(crow["candidate_type"])

                cell = _cell_from_row(
                    crow,
                    h="selected_horizon_days",
                    q="selected_Q",
                    w="selected_window",
                )
                crows = _cell_rows(
                    evaluation,
                    cell,
                    seeds,
                    label=f"{bg} {policy} {point_type}",
                )

                points.append(
                    _point_summary(
                        crows,
                        background_id=bg,
                        policy=policy,
                        point_type=point_type,
                        cell=cell,
                    )
                )

                for comparison, rhs, rhs_cell in (
                    (
                        "vs_avg_util_optimum",
                        avg_rows,
                        avg_cell,
                    ),
                    (
                        "vs_baseline",
                        baseline_rows,
                        baseline_cell,
                    ),
                ):
                    rec, srows = _paired_record(
                        lhs=crows,
                        rhs=rhs,
                        background_id=bg,
                        policy=policy,
                        point_type=point_type,
                        comparison=comparison,
                        lhs_cell=cell,
                        rhs_cell=rhs_cell,
                        n_boot=args.bootstrap,
                        bootstrap_seed=args.bootstrap_seed,
                    )
                    paired.append(rec)
                    seed_deltas.extend(srows)

        if (i + 1) % 50 == 0 or (i + 1) == len(context):
            print(
                f"Processed {i + 1}/{len(context)} backgrounds"
            )

    point_df = pd.DataFrame(points).merge(
        context,
        on="background_id",
        how="left",
        validate="many_to_one",
    )
    paired_df = pd.DataFrame(paired).merge(
        context,
        on="background_id",
        how="left",
        validate="many_to_one",
    )
    seed_df = pd.DataFrame(seed_deltas).merge(
        context,
        on="background_id",
        how="left",
        validate="many_to_one",
    )

    # Primary access-recovery comparison: frozen access point vs
    # same policy's frozen average-utilization optimum.
    access = paired_df[
        paired_df["point_type"].eq("access_recovery")
        & paired_df["comparison"].eq(
            "vs_avg_util_optimum"
        )
    ].copy()

    overall_access = _access_summary(
        access,
        group_cols=["policy"],
    )
    grid_access = _access_summary(
        access,
        group_cols=[
            "policy",
            "noshow_level",
            "balk_level",
        ],
    )

    # Secondary baseline comparison of frozen favorable candidates.
    favorable = paired_df[
        paired_df["point_type"].isin(
            [
                "best_win_win",
                "c1_win_c2_neutral_if_better",
            ]
        )
        & paired_df["comparison"].eq("vs_baseline")
    ].copy()

    favorable["strict_target_validated"] = np.where(
        favorable["point_type"].eq("best_win_win"),
        favorable["access_status"].eq("Win-win"),
        favorable["access_status"].eq(
            "C1 win / C2 neutral"
        ),
    )

    favorable["favorable_or_better"] = (
        favorable["access_status"].isin(
            ["Win-win", "C1 win / C2 neutral"]
        )
    )

    favorable_summary = _favorable_validation_summary(
        favorable
    )

    favorable_grid_parts: list[pd.DataFrame] = []
    for range_name, range_df in _ranges(favorable):
        if range_df.empty:
            continue
        g = (
            range_df.assign(
                final_win_win=range_df["access_status"].eq(
                    "Win-win"
                ),
                final_c1_win_c2_neutral=range_df[
                    "access_status"
                ].eq("C1 win / C2 neutral"),
            )
            .groupby(
                [
                    "policy",
                    "point_type",
                    "noshow_level",
                    "balk_level",
                ],
                as_index=False,
            )
            .agg(
                frozen_candidates=("background_id", "size"),
                strict_target_validated=(
                    "strict_target_validated",
                    "sum",
                ),
                favorable_or_better=(
                    "favorable_or_better",
                    "sum",
                ),
                final_win_win=("final_win_win", "sum"),
                final_c1_win_c2_neutral=(
                    "final_c1_win_c2_neutral",
                    "sum",
                ),
            )
        )
        g.insert(0, "analysis_range", range_name)
        favorable_grid_parts.append(g)

    favorable_grid = pd.concat(
        favorable_grid_parts,
        ignore_index=True,
    )

    # Background-level existence flags across the pre-frozen favorable
    # candidates. This does not re-select a policy cell using evaluation data.
    favorable_background = (
        favorable.assign(
            final_win_win=favorable["access_status"].eq(
                "Win-win"
            ),
            final_c1_win_c2_neutral=favorable[
                "access_status"
            ].eq("C1 win / C2 neutral"),
        )
        .groupby(
            [
                "background_id",
                "policy",
                "rho",
                "class1_share",
                "capacity",
                "noshow_level",
                "balk_level",
            ],
            as_index=False,
        )
        .agg(
            frozen_favorable_candidates=("point_type", "size"),
            any_final_win_win=("final_win_win", "max"),
            any_final_c1_win_c2_neutral=(
                "final_c1_win_c2_neutral",
                "max",
            ),
            any_final_favorable=(
                "favorable_or_better",
                "max",
            ),
        )
    )

    # Average-utilization-optimum baseline effects for context.
    avgopt = paired_df[
        paired_df["point_type"].eq("avg_util_optimum")
        & paired_df["comparison"].eq("vs_baseline")
    ].copy()

    avgopt_summary = _access_summary(
        avgopt,
        group_cols=["policy"],
    )
    avgopt_grid = _access_summary(
        avgopt,
        group_cols=[
            "policy",
            "noshow_level",
            "balk_level",
        ],
    )

    release = (
        root
        / "access_recovery_optimization"
        / "evaluation_release"
    )
    release.mkdir(parents=True, exist_ok=True)

    outputs = {
        "evaluation_points.csv": point_df,
        "paired_deltas_background.csv": paired_df,
        "paired_deltas_seed.csv": seed_df,
        "access_recovery_overall_summary.csv": overall_access,
        "access_recovery_3x3_summary.csv": grid_access,
        "favorable_validation_summary.csv": favorable_summary,
        "favorable_3x3_counts.csv": favorable_grid,
        "favorable_validated_details.csv": favorable,
        "favorable_background_validation.csv": favorable_background,
        "avg_opt_vs_baseline_summary.csv": avgopt_summary,
        "avg_opt_vs_baseline_3x3_summary.csv": avgopt_grid,
    }

    for name, frame in outputs.items():
        frame.to_csv(release / name, index=False)
        print(f"Wrote: {release / name}")

    print("\nFINAL INDEPENDENT-EVALUATION DIAGNOSTIC")
    print(
        f"Evaluation seeds: {len(seeds)} "
        f"({min(seeds)}..{max(seeds)})"
    )
    print(f"Paired bootstrap resamples: {args.bootstrap}")

    print("\nAccess recovery vs same-policy average-util optimum")
    show = overall_access[
        overall_access["analysis_range"].eq(
            "headline_rho_le_2_5"
        )
    ][
        [
            "policy",
            "backgrounds",
            "median_delta_average_utilization",
            "median_delta_class_1_percent_serviced",
            "median_delta_class_2_percent_serviced",
            "share_c1_supported_gain",
            "share_util_within_0_5pp_budget_mean",
        ]
    ]
    print(show.to_string(index=False))

    print("\nFrozen favorable candidates vs baseline")
    show2 = favorable_summary[
        favorable_summary["analysis_range"].eq(
            "headline_rho_le_2_5"
        )
    ]
    print(show2.to_string(index=False))


if __name__ == "__main__":
    main()
