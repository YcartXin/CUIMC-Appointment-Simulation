"""Postprocess the expanded horizon-reservation experiment.

Reads the frozen expanded selections and 10-seed independent evaluation shards,
validates every selected cell, and writes compact release tables for:

* average-utilization selections;
* revenue-weighted service selections R_alpha = alpha*Y1 + Y2 for alpha 1.3/1.6/2.0;
* constrained-revenue selections at 0.5pp and 1.0pp no-harm tolerances; and
* dedicated max-min win-win selections at 0.5pp and 1.0pp utilization floors.

All reported policy effects are paired across independent evaluation seeds. The
primary practical-effect band is +/-0.5 percentage points (0.005); a +/-1.0pp
sensitivity classification is also written. Bootstrap confidence intervals use
paired seed resampling and are deterministic by background.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

POLICIES = ("baseline", "horizon_only", "reservation_only", "both_flexible")
COMPARISONS = (
    ("horizon_only_vs_baseline", "horizon_only", "baseline"),
    ("reservation_only_vs_baseline", "reservation_only", "baseline"),
    ("both_flexible_vs_baseline", "both_flexible", "baseline"),
    ("both_flexible_vs_horizon_only", "both_flexible", "horizon_only"),
    ("both_flexible_vs_reservation_only", "both_flexible", "reservation_only"),
    ("reservation_only_vs_horizon_only", "reservation_only", "horizon_only"),
)
ALPHAS = (1.3, 1.6, 2.0)
PRIMARY_TOL = 0.005
SENSITIVITY_TOL = 0.010

# Metrics for paired bootstrap. Revenue metrics are derived from the same Y1/Y2
# draws so their intervals preserve the within-seed covariance exactly.
BOOT_METRICS = (
    "average_utilization",
    "class_1_percent_serviced",
    "class_2_percent_serviced",
    "class_1_served",
    "class_2_served",
)

MEAN_METRICS = (
    "average_utilization",
    "booked_slot_utilization",
    "overall_percent_serviced",
    "mean_accepted_booking_delay",
    "mean_offered_booking_delay",
    "class_1_percent_serviced",
    "class_2_percent_serviced",
    "overall_balking_rate",
    "class_1_balking_rate",
    "class_2_balking_rate",
    "class_1_slot_utilization",
    "class_2_slot_utilization",
    "class_1_booked_slot_utilization",
    "class_2_booked_slot_utilization",
    "class_1_mean_offered_booking_delay",
    "class_2_mean_offered_booking_delay",
    "access_advantage_class_1",
    "balking_rate_gap_class_1",
    "delay_advantage_class_1",
    "priority_weighted_utilization",
    "class_1_arrivals",
    "class_2_arrivals",
    "class_1_served",
    "class_2_served",
    "class_1_no_show_rate",
    "class_2_no_show_rate",
    "class_1_no_offer_rate",
    "class_2_no_offer_rate",
    "reserved_slot_fill_rate",
)

KEY_METADATA = (
    "bank_segment",
    "patient_characteristic",
    "class2_reference",
    "contrast_level",
    "profile_id",
    "clinic_context_id",
    "rho",
    "class1_share",
    "slots_per_day",
    "native_horizon_days",
)


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _fmt_float_token(value: float) -> str:
    return ("%g" % value).replace(".", "p")


def _selection_id(family: str, alpha: float | None, tolerance: float | None) -> str:
    if family == "average":
        return "average"
    if family == "revenue":
        if alpha is None or math.isnan(alpha):
            raise ValueError("Revenue selection missing alpha")
        a = _fmt_float_token(float(alpha))
        if tolerance is None or math.isnan(tolerance):
            return f"revenue_a{a}_unconstrained"
        return f"revenue_a{a}_tol{int(round(float(tolerance) * 1000)):03d}"
    if family == "winwin":
        if tolerance is None or math.isnan(tolerance):
            raise ValueError("Win-win selection missing loss tolerance")
        return f"winwin_tol{int(round(float(tolerance) * 1000)):03d}"
    raise ValueError(f"Unknown selection family: {family}")


def _effect_status(mean: float, low: float, high: float, tolerance: float) -> str:
    if any(math.isnan(v) for v in (mean, low, high)):
        return "uncertain"
    if mean >= tolerance and low > 0:
        return "meaningful_gain"
    if mean <= -tolerance and high < 0:
        return "meaningful_harm"
    if low >= -tolerance and high <= tolerance:
        return "supported_neutral"
    return "uncertain"


def _access_outcome(c1_status: str, c2_status: str) -> str:
    if c1_status == "meaningful_gain" and c2_status == "meaningful_gain":
        return "win_win"
    if c1_status == "meaningful_gain" and c2_status == "supported_neutral":
        return "class1_win_class2_neutral"
    if c1_status == "meaningful_gain" and c2_status == "meaningful_harm":
        return "class1_win_class2_harm"
    if c1_status == "supported_neutral" and c2_status == "meaningful_gain":
        return "class1_neutral_class2_win"
    if c1_status == "meaningful_harm" and c2_status == "meaningful_gain":
        return "class1_harm_class2_win"
    if c1_status == "meaningful_harm" and c2_status == "meaningful_harm":
        return "lose_lose"
    if c1_status == "supported_neutral" and c2_status == "supported_neutral":
        return "both_supported_neutral"
    return "uncertain_or_mixed"


def _as_optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _augment_eval(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("class_1_served", "class_2_served"):
        if col not in out.columns:
            raise ValueError(f"Evaluation data missing required column: {col}")
    for alpha in ALPHAS:
        token = str(alpha).replace(".", "_")
        out[f"revenue_alpha_{token}"] = alpha * out["class_1_served"] + out["class_2_served"]
    return out


def _load_eval_cell(frame: pd.DataFrame, h: int, q: int, w: int, expected_seeds: int) -> pd.DataFrame:
    cell = frame[
        (frame["horizon_days"] == int(h))
        & (frame["Q"] == int(q))
        & (frame["window"] == int(w))
    ].copy()
    if cell.empty:
        raise ValueError(f"Evaluation cell not found: H={h}, Q={q}, W={w}")
    if cell.duplicated(["seed"]).any():
        # Evaluation rows should be unique by selected cell and seed.
        cell = cell.groupby("seed", as_index=False, sort=True).first()
    else:
        cell = cell.sort_values("seed").reset_index(drop=True)
    if cell["seed"].nunique() != expected_seeds or len(cell) != expected_seeds:
        raise ValueError(
            f"Evaluation cell H={h},Q={q},W={w} has {len(cell)} rows / "
            f"{cell['seed'].nunique()} seeds; expected {expected_seeds}"
        )
    return cell


def _selection_rows_for_background(selection: pd.DataFrame, background_id: str) -> pd.DataFrame:
    rows = selection[selection["background_id"].astype(str) == background_id].copy()
    if len(rows) != 48:
        raise ValueError(f"{background_id}: expected 48 selection rows, found {len(rows)}")
    rows["selection_id"] = [
        _selection_id(
            str(r.selection_family),
            _as_optional_float(r.alpha),
            _as_optional_float(r.loss_tolerance),
        )
        for r in rows.itertuples(index=False)
    ]
    counts = rows.groupby("selection_id")["policy"].nunique()
    if len(counts) != 12 or not (counts == 4).all():
        raise ValueError(
            f"{background_id}: expected 12 selection specs x 4 policies; got {counts.to_dict()}"
        )
    return rows


def _mean_metrics(cell: pd.DataFrame) -> dict[str, float]:
    row: dict[str, float] = {}
    for metric in MEAN_METRICS:
        if metric in cell.columns:
            row[metric] = float(cell[metric].mean())
            row[f"{metric}_sd"] = float(cell[metric].std(ddof=1))
    for alpha in ALPHAS:
        token = str(alpha).replace(".", "_")
        col = f"revenue_alpha_{token}"
        row[col] = float(cell[col].mean())
        row[f"{col}_sd"] = float(cell[col].std(ddof=1))
    return row


def _key_metadata(bank_row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in KEY_METADATA:
        if col == "native_horizon_days":
            out[col] = int(bank_row["horizon_days"])
        elif col in bank_row.index:
            out[col] = bank_row[col]
    out["headline_eligible"] = str(bank_row.get("contrast_level", "")) != "same"
    out["demand_regime"] = "stress" if float(bank_row.get("rho", math.nan)) >= 3.0 else "main"
    return out


def _full_bank_metadata(bank_row: pd.Series) -> dict[str, Any]:
    out = bank_row.to_dict()
    if "horizon_days" in out:
        out["native_horizon_days"] = out.pop("horizon_days")
    out["headline_eligible"] = str(bank_row.get("contrast_level", "")) != "same"
    out["demand_regime"] = "stress" if float(bank_row.get("rho", math.nan)) >= 3.0 else "main"
    return out


def _constraint_flags(
    *,
    family: str,
    tolerance: float | None,
    policy: str,
    delta_u: float,
    delta_c1: float,
    delta_c2: float,
) -> dict[str, Any]:
    if policy == "baseline":
        if tolerance is None:
            return {"evaluation_constraint_applicable": False, "evaluation_constraint_met": True}
        return {"evaluation_constraint_applicable": True, "evaluation_constraint_met": True}
    if tolerance is None:
        return {"evaluation_constraint_applicable": False, "evaluation_constraint_met": math.nan}
    if family == "revenue":
        met = delta_u >= -tolerance and delta_c1 >= -tolerance and delta_c2 >= -tolerance
    elif family == "winwin":
        met = delta_u >= -tolerance
    else:
        return {"evaluation_constraint_applicable": False, "evaluation_constraint_met": math.nan}
    return {"evaluation_constraint_applicable": True, "evaluation_constraint_met": bool(met)}


def _writer(path: Path, fieldnames: list[str]) -> tuple[Any, csv.DictWriter]:
    fh = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    return fh, writer


def process(
    *,
    raw_root: Path,
    bank_path: Path,
    output_root: Path,
    draws: int,
    expected_evaluation_seeds: int,
) -> None:
    bank = pd.read_csv(bank_path)
    if bank["background_id"].duplicated().any():
        raise ValueError("Expanded bank has duplicate background_id values")
    bank["background_id"] = bank["background_id"].astype(str)
    bank_idx = bank.set_index("background_id", drop=False)

    selection_path = raw_root / "selection" / "selected_cells.csv"
    eval_dir = raw_root / "evaluation" / "raw"
    selection = pd.read_csv(selection_path)
    selection["background_id"] = selection["background_id"].astype(str)

    expected_ids = set(bank["background_id"])
    found_files = sorted(eval_dir.glob("*.csv"))
    found_ids = {p.stem for p in found_files}
    if found_ids != expected_ids:
        missing = sorted(expected_ids - found_ids)
        extra = sorted(found_ids - expected_ids)
        raise RuntimeError(
            f"Evaluation coverage mismatch: missing={len(missing)}, extra={len(extra)}; "
            f"examples missing={missing[:5]}, extra={extra[:5]}"
        )
    if selection["background_id"].nunique() != len(bank):
        raise RuntimeError(
            f"Selection background count {selection['background_id'].nunique()} != bank count {len(bank)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    selected_tmp = output_root / "selected_policy_outcomes.csv.tmp"
    pairwise_tmp = output_root / "pairwise_group_deltas.csv.tmp"
    constraints_tmp = output_root / "constraint_validation.csv.tmp"
    validation_tmp = output_root / "selection_validation.csv.tmp"

    # Stable field order. Full bank metadata is appended to selected outcomes; pairwise
    # tables intentionally carry only report-critical metadata to keep release size sane.
    selection_fields = [
        "background_id", "selection_id", "selection_family", "policy", "alpha", "loss_tolerance",
        "selected_horizon_days", "selected_Q", "selected_window", "selected_source_stage",
        "selected_canonical_policy", "legacy_reused", "boundary_extended_both",
        "n_seeds", "delta_average_utilization_vs_baseline", "delta_class_1_percent_serviced_vs_baseline",
        "delta_class_2_percent_serviced_vs_baseline", "min_class_served_rate_delta_vs_baseline",
        "evaluation_constraint_applicable", "evaluation_constraint_met",
    ]
    selection_fields += [c for c in MEAN_METRICS if c not in selection_fields]
    selection_fields += [f"{c}_sd" for c in MEAN_METRICS]
    for alpha in ALPHAS:
        tok = str(alpha).replace(".", "_")
        selection_fields += [f"revenue_alpha_{tok}", f"revenue_alpha_{tok}_sd", f"revenue_alpha_{tok}_per_capacity"]
    selection_fields += [c for c in selection.columns if c not in selection_fields and c != "background_id"]
    bank_output_cols = ["native_horizon_days" if c == "horizon_days" else c for c in bank.columns if c != "background_id"]
    selection_fields += [c for c in bank_output_cols if c not in selection_fields]
    selection_fields += ["headline_eligible", "demand_regime"]
    selection_fields = list(dict.fromkeys(selection_fields))

    pairwise_fields = [
        "background_id", "selection_id", "selection_family", "alpha", "loss_tolerance",
        "comparison", "first_policy", "second_policy", "n_paired_seeds",
    ] + list(KEY_METADATA) + ["headline_eligible", "demand_regime"]
    for metric in ("average_utilization", "class_1_percent_serviced", "class_2_percent_serviced", "class_1_served", "class_2_served"):
        pairwise_fields += [f"delta_{metric}", f"delta_{metric}_ci_low", f"delta_{metric}_ci_high"]
    for metric in ("average_utilization", "class_1_percent_serviced", "class_2_percent_serviced"):
        pairwise_fields += [f"{metric}_status", f"{metric}_status_1pp"]
    for alpha in ALPHAS:
        tok = str(alpha).replace(".", "_")
        pairwise_fields += [
            f"delta_revenue_alpha_{tok}", f"delta_revenue_alpha_{tok}_ci_low", f"delta_revenue_alpha_{tok}_ci_high",
            f"delta_revenue_alpha_{tok}_per_capacity", f"delta_revenue_alpha_{tok}_per_capacity_ci_low", f"delta_revenue_alpha_{tok}_per_capacity_ci_high",
            f"revenue_alpha_{tok}_percent_change_vs_second",
        ]
    pairwise_fields += [
        "min_class_served_rate_delta", "min_class_served_rate_delta_ci_low", "min_class_served_rate_delta_ci_high",
        "access_outcome", "access_outcome_1pp", "target_delta", "target_delta_ci_low", "target_delta_ci_high",
    ]
    pairwise_fields = list(dict.fromkeys(pairwise_fields))

    constraint_fields = [
        "background_id", "selection_id", "selection_family", "policy", "alpha", "loss_tolerance",
        "selected_horizon_days", "selected_Q", "selected_window",
        "delta_average_utilization_vs_baseline", "delta_class_1_percent_serviced_vs_baseline",
        "delta_class_2_percent_serviced_vs_baseline", "min_class_served_rate_delta_vs_baseline",
        "evaluation_constraint_met", "search_delta_average_utilization_vs_baseline",
        "search_delta_class_1_served_rate_vs_baseline", "search_delta_class_2_served_rate_vs_baseline",
    ] + list(KEY_METADATA) + ["headline_eligible", "demand_regime"]

    validation_fields = [
        "background_id", "selection_id", "selection_family", "policy", "alpha", "loss_tolerance",
        "selected_horizon_days", "selected_Q", "selected_window", "evaluation_seed_count", "evaluation_cell_found",
    ]

    handles = []
    try:
        sfh, sw = _writer(selected_tmp, selection_fields); handles.append(sfh)
        pfh, pw = _writer(pairwise_tmp, pairwise_fields); handles.append(pfh)
        cfh, cw = _writer(constraints_tmp, constraint_fields); handles.append(cfh)
        vfh, vw = _writer(validation_tmp, validation_fields); handles.append(vfh)

        selected_count = pairwise_count = constraint_count = validation_count = 0
        constraint_applicable = constraint_failed = 0

        for idx, background_id in enumerate(sorted(expected_ids), start=1):
            bank_row = bank_idx.loc[background_id]
            metadata_full = _full_bank_metadata(bank_row)
            metadata_key = _key_metadata(bank_row)
            capacity = float(bank_row["slots_per_day"])
            eval_frame = _augment_eval(pd.read_csv(eval_dir / f"{background_id}.csv"))
            sel = _selection_rows_for_background(selection, background_id)

            spec_policy_cells: dict[str, dict[str, pd.DataFrame]] = {}
            spec_rows: dict[str, dict[str, pd.Series]] = {}

            # Load and validate all 48 frozen selections first.
            for r in sel.itertuples(index=False):
                sid = str(r.selection_id)
                policy = str(r.policy)
                cell = _load_eval_cell(
                    eval_frame,
                    int(r.selected_horizon_days),
                    int(r.selected_Q),
                    int(r.selected_window),
                    expected_evaluation_seeds,
                )
                spec_policy_cells.setdefault(sid, {})[policy] = cell
                spec_rows.setdefault(sid, {})[policy] = pd.Series(r._asdict())
                vw.writerow({
                    "background_id": background_id,
                    "selection_id": sid,
                    "selection_family": r.selection_family,
                    "policy": policy,
                    "alpha": r.alpha,
                    "loss_tolerance": r.loss_tolerance,
                    "selected_horizon_days": int(r.selected_horizon_days),
                    "selected_Q": int(r.selected_Q),
                    "selected_window": int(r.selected_window),
                    "evaluation_seed_count": int(cell["seed"].nunique()),
                    "evaluation_cell_found": True,
                })
                validation_count += 1

            # Write selected-policy mean outcomes and OOS constraint checks.
            for sid in sorted(spec_policy_cells):
                policy_cells = spec_policy_cells[sid]
                row_map = spec_rows[sid]
                baseline = policy_cells["baseline"]
                bmean = baseline[["average_utilization", "class_1_percent_serviced", "class_2_percent_serviced"]].mean()

                for policy in POLICIES:
                    cell = policy_cells[policy]
                    sr = row_map[policy]
                    means = _mean_metrics(cell)
                    delta_u = float(means["average_utilization"] - bmean["average_utilization"])
                    delta_c1 = float(means["class_1_percent_serviced"] - bmean["class_1_percent_serviced"])
                    delta_c2 = float(means["class_2_percent_serviced"] - bmean["class_2_percent_serviced"])
                    tol = _as_optional_float(sr.get("loss_tolerance"))
                    family = str(sr["selection_family"])
                    flags = _constraint_flags(
                        family=family, tolerance=tol, policy=policy,
                        delta_u=delta_u, delta_c1=delta_c1, delta_c2=delta_c2,
                    )
                    out = {
                        "background_id": background_id,
                        "selection_id": sid,
                        "selection_family": family,
                        "policy": policy,
                        "alpha": sr.get("alpha"),
                        "loss_tolerance": sr.get("loss_tolerance"),
                        "selected_horizon_days": int(sr["selected_horizon_days"]),
                        "selected_Q": int(sr["selected_Q"]),
                        "selected_window": int(sr["selected_window"]),
                        "selected_source_stage": sr.get("selected_source_stage"),
                        "selected_canonical_policy": sr.get("selected_canonical_policy"),
                        "legacy_reused": sr.get("legacy_reused"),
                        "boundary_extended_both": sr.get("boundary_extended_both"),
                        "n_seeds": int(cell["seed"].nunique()),
                        "delta_average_utilization_vs_baseline": delta_u,
                        "delta_class_1_percent_serviced_vs_baseline": delta_c1,
                        "delta_class_2_percent_serviced_vs_baseline": delta_c2,
                        "min_class_served_rate_delta_vs_baseline": min(delta_c1, delta_c2),
                        **flags,
                        **means,
                        **sr.to_dict(),
                        **metadata_full,
                    }
                    for alpha in ALPHAS:
                        tok = str(alpha).replace(".", "_")
                        out[f"revenue_alpha_{tok}_per_capacity"] = out[f"revenue_alpha_{tok}"] / capacity
                    sw.writerow(out)
                    selected_count += 1

                    if flags["evaluation_constraint_applicable"]:
                        constraint_applicable += 1
                        met = bool(flags["evaluation_constraint_met"])
                        if not met:
                            constraint_failed += 1
                        cw.writerow({
                            "background_id": background_id,
                            "selection_id": sid,
                            "selection_family": family,
                            "policy": policy,
                            "alpha": sr.get("alpha"),
                            "loss_tolerance": sr.get("loss_tolerance"),
                            "selected_horizon_days": int(sr["selected_horizon_days"]),
                            "selected_Q": int(sr["selected_Q"]),
                            "selected_window": int(sr["selected_window"]),
                            "delta_average_utilization_vs_baseline": delta_u,
                            "delta_class_1_percent_serviced_vs_baseline": delta_c1,
                            "delta_class_2_percent_serviced_vs_baseline": delta_c2,
                            "min_class_served_rate_delta_vs_baseline": min(delta_c1, delta_c2),
                            "evaluation_constraint_met": met,
                            "search_delta_average_utilization_vs_baseline": sr.get("search_delta_average_utilization_vs_baseline"),
                            "search_delta_class_1_served_rate_vs_baseline": sr.get("search_delta_class_1_served_rate_vs_baseline"),
                            "search_delta_class_2_served_rate_vs_baseline": sr.get("search_delta_class_2_served_rate_vs_baseline"),
                            **metadata_key,
                        })
                        constraint_count += 1

            # Prepare all 72 paired comparisons for this background, then bootstrap
            # them in one matrix operation using a shared paired-resampling matrix.
            comparison_meta: list[dict[str, Any]] = []
            delta_blocks: list[np.ndarray] = []
            second_revenue_means: list[dict[float, float]] = []

            for sid in sorted(spec_policy_cells):
                row0 = next(iter(spec_rows[sid].values()))
                family = str(row0["selection_family"])
                alpha = _as_optional_float(row0.get("alpha"))
                tol = _as_optional_float(row0.get("loss_tolerance"))
                policy_cells = spec_policy_cells[sid]

                for comparison, first_policy, second_policy in COMPARISONS:
                    first = policy_cells[first_policy].set_index("seed").sort_index()
                    second = policy_cells[second_policy].set_index("seed").sort_index()
                    seeds = sorted(set(first.index) & set(second.index))
                    if len(seeds) != expected_evaluation_seeds:
                        raise ValueError(
                            f"{background_id}/{sid}/{comparison}: paired seeds={len(seeds)}; "
                            f"expected {expected_evaluation_seeds}"
                        )
                    delta = (
                        first.loc[seeds, list(BOOT_METRICS)].to_numpy(float)
                        - second.loc[seeds, list(BOOT_METRICS)].to_numpy(float)
                    )
                    comparison_meta.append({
                        "background_id": background_id,
                        "selection_id": sid,
                        "selection_family": family,
                        "alpha": alpha,
                        "loss_tolerance": tol,
                        "comparison": comparison,
                        "first_policy": first_policy,
                        "second_policy": second_policy,
                        "n_paired_seeds": len(seeds),
                        **metadata_key,
                    })
                    delta_blocks.append(delta)
                    second_revenue_means.append({
                        a: float((a * second.loc[seeds, "class_1_served"] + second.loc[seeds, "class_2_served"]).mean())
                        for a in ALPHAS
                    })

            ncomp = len(delta_blocks)
            m = len(BOOT_METRICS)
            wide = np.hstack(delta_blocks)  # nseed x (ncomp*m)
            point = wide.mean(axis=0)
            if draws > 0 and expected_evaluation_seeds > 1:
                rng = np.random.default_rng(_stable_seed(background_id, "expanded_paired_bootstrap"))
                weights = rng.multinomial(
                    expected_evaluation_seeds,
                    [1.0 / expected_evaluation_seeds] * expected_evaluation_seeds,
                    size=draws,
                ).astype(float) / expected_evaluation_seeds
                boot = weights @ wide
                low, high = np.quantile(boot, [0.025, 0.975], axis=0)
            else:
                boot = None
                low = np.full_like(point, np.nan)
                high = np.full_like(point, np.nan)

            for j, meta in enumerate(comparison_meta):
                sl = slice(j * m, (j + 1) * m)
                p = point[sl]
                lo = low[sl]
                hi = high[sl]
                out = dict(meta)
                for k, metric in enumerate(BOOT_METRICS):
                    out[f"delta_{metric}"] = float(p[k])
                    out[f"delta_{metric}_ci_low"] = float(lo[k])
                    out[f"delta_{metric}_ci_high"] = float(hi[k])

                for metric in ("average_utilization", "class_1_percent_serviced", "class_2_percent_serviced"):
                    k = BOOT_METRICS.index(metric)
                    out[f"{metric}_status"] = _effect_status(float(p[k]), float(lo[k]), float(hi[k]), PRIMARY_TOL)
                    out[f"{metric}_status_1pp"] = _effect_status(float(p[k]), float(lo[k]), float(hi[k]), SENSITIVITY_TOL)

                k1 = BOOT_METRICS.index("class_1_served")
                k2 = BOOT_METRICS.index("class_2_served")
                for a in ALPHAS:
                    tok = str(a).replace(".", "_")
                    rev_point = a * p[k1] + p[k2]
                    if boot is not None:
                        bsl = boot[:, sl]
                        rev_boot = a * bsl[:, k1] + bsl[:, k2]
                        rev_low, rev_high = np.quantile(rev_boot, [0.025, 0.975])
                    else:
                        rev_low = rev_high = math.nan
                    out[f"delta_revenue_alpha_{tok}"] = float(rev_point)
                    out[f"delta_revenue_alpha_{tok}_ci_low"] = float(rev_low)
                    out[f"delta_revenue_alpha_{tok}_ci_high"] = float(rev_high)
                    out[f"delta_revenue_alpha_{tok}_per_capacity"] = float(rev_point / capacity)
                    out[f"delta_revenue_alpha_{tok}_per_capacity_ci_low"] = float(rev_low / capacity)
                    out[f"delta_revenue_alpha_{tok}_per_capacity_ci_high"] = float(rev_high / capacity)
                    denom = second_revenue_means[j][a]
                    out[f"revenue_alpha_{tok}_percent_change_vs_second"] = (
                        float(100.0 * rev_point / denom) if denom != 0 else math.nan
                    )

                kc1 = BOOT_METRICS.index("class_1_percent_serviced")
                kc2 = BOOT_METRICS.index("class_2_percent_serviced")
                out["min_class_served_rate_delta"] = float(min(p[kc1], p[kc2]))
                if boot is not None:
                    bsl = boot[:, sl]
                    min_boot = np.minimum(bsl[:, kc1], bsl[:, kc2])
                    min_low, min_high = np.quantile(min_boot, [0.025, 0.975])
                else:
                    min_low = min_high = math.nan
                out["min_class_served_rate_delta_ci_low"] = float(min_low)
                out["min_class_served_rate_delta_ci_high"] = float(min_high)
                out["access_outcome"] = _access_outcome(
                    out["class_1_percent_serviced_status"], out["class_2_percent_serviced_status"]
                )
                out["access_outcome_1pp"] = _access_outcome(
                    out["class_1_percent_serviced_status_1pp"], out["class_2_percent_serviced_status_1pp"]
                )

                family = str(meta["selection_family"])
                if family == "average":
                    ku = BOOT_METRICS.index("average_utilization")
                    out["target_delta"] = float(p[ku])
                    out["target_delta_ci_low"] = float(lo[ku])
                    out["target_delta_ci_high"] = float(hi[ku])
                elif family == "revenue":
                    a = float(meta["alpha"])
                    tok = str(a).replace(".", "_")
                    out["target_delta"] = out[f"delta_revenue_alpha_{tok}_per_capacity"]
                    out["target_delta_ci_low"] = out[f"delta_revenue_alpha_{tok}_per_capacity_ci_low"]
                    out["target_delta_ci_high"] = out[f"delta_revenue_alpha_{tok}_per_capacity_ci_high"]
                else:
                    out["target_delta"] = out["min_class_served_rate_delta"]
                    out["target_delta_ci_low"] = out["min_class_served_rate_delta_ci_low"]
                    out["target_delta_ci_high"] = out["min_class_served_rate_delta_ci_high"]

                pw.writerow(out)
                pairwise_count += 1

            if idx % 100 == 0 or idx == len(expected_ids):
                print(f"Processed {idx:,}/{len(expected_ids):,} expanded evaluation backgrounds")

        for fh in handles:
            fh.flush()
    finally:
        for fh in handles:
            try:
                fh.close()
            except Exception:
                pass

    # Atomic-ish publish: successful run replaces release tables only after all backgrounds finish.
    selected_final = output_root / "selected_policy_outcomes.csv"
    pairwise_final = output_root / "pairwise_group_deltas.csv"
    constraints_final = output_root / "constraint_validation.csv"
    validation_final = output_root / "selection_validation.csv"
    selected_tmp.replace(selected_final)
    pairwise_tmp.replace(pairwise_final)
    constraints_tmp.replace(constraints_final)
    validation_tmp.replace(validation_final)

    headline_backgrounds = int((bank["contrast_level"].astype(str) != "same").sum())
    main_headline = int(((bank["contrast_level"].astype(str) != "same") & (bank["rho"] <= 2.5)).sum())
    stress_headline = int(((bank["contrast_level"].astype(str) != "same") & (bank["rho"] >= 3.0)).sum())
    fail_rate = 100.0 * constraint_failed / constraint_applicable if constraint_applicable else math.nan

    summary = [
        "# Expanded horizon-reservation postprocessing",
        "",
        f"Backgrounds: {len(bank):,}",
        f"Profiles: {bank['profile_id'].nunique():,}",
        f"Headline-eligible heterogeneous backgrounds: {headline_backgrounds:,}",
        f"Headline main-range backgrounds (rho <= 2.5): {main_headline:,}",
        f"Headline stress backgrounds (rho = 3.0): {stress_headline:,}",
        f"Independent evaluation seeds: {expected_evaluation_seeds}",
        f"Paired bootstrap draws: {draws:,}",
        f"Primary practical-equivalence band: +/-{PRIMARY_TOL:.3f} (0.5 percentage points)",
        f"Sensitivity band: +/-{SENSITIVITY_TOL:.3f} (1.0 percentage point)",
        "",
        "## Output rows",
        "",
        f"- selected_policy_outcomes.csv: {selected_count:,}",
        f"- pairwise_group_deltas.csv: {pairwise_count:,}",
        f"- constraint_validation.csv: {constraint_count:,}",
        f"- selection_validation.csv: {validation_count:,}",
        "",
        "## Constraint validation",
        "",
        f"- Applicable selected-policy rows: {constraint_applicable:,}",
        f"- Evaluation constraint failures: {constraint_failed:,}",
        f"- Evaluation constraint failure rate: {fail_rate:.2f}%",
        "",
        "Same-behavior profiles remain in the release tables as matched controls but are marked headline_eligible=False.",
        "All policy-effect intervals use paired resampling of the independent evaluation seeds.",
    ]
    (output_root / "postprocess_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Final outputs: {output_root}")
    print(f"Selected rows: {selected_count:,}")
    print(f"Pairwise rows: {pairwise_count:,}")
    print(f"Constraint failure rate: {fail_rate:.2f}%")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--expected-evaluation-seeds", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    process(
        raw_root=args.raw_root,
        bank_path=args.bank,
        output_root=args.output_root,
        draws=args.bootstrap_draws,
        expected_evaluation_seeds=args.expected_evaluation_seeds,
    )


if __name__ == "__main__":
    main()
