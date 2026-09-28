#!/usr/bin/env python3
"""Served-rate Pareto search for the 3x3 patient-behavior factorial.

New objective (independent of prior selected winners):
  eligibility: delta SR1 >= 0 and delta SR2 >= 0 vs matched no-policy baseline
  objective: maximize overall served rate
             (served_1 + served_2) / (arrivals_1 + arrivals_2)

Policies: horizon_only, reservation_only, both_flexible.
Design: 5-seed broad screen, 10-seed objective-specific refinement,
20-seed independent validation, H=2..30. Utilization is not used.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.patient_behavior_factorial import (
    EVALUATION_KEY_COLUMNS,
    EVALUATION_SEED_POOL,
    SEARCH_KEY_COLUMNS,
    SEARCH_SEED_POOL,
    _filter_pending,
    _make_task,
    _task_grid,
    load_bank,
    q_coarse_grid,
    q_fine_grid,
    run_tasks,
    shard_path,
    window_coarse_grid,
    window_fine_grid,
)

POLICIES = ("horizon_only", "reservation_only", "both_flexible")
CELL_COLS = ["horizon_days", "Q", "window"]
HORIZONS = tuple(range(2, 31))
BROAD_N_SEEDS = 5
REFINE_N_SEEDS = 10
EVAL_N_SEEDS = 20
NEAR_FEASIBLE_MARGIN = 0.0025
TOP_FEASIBLE = 3
TOP_NEAR = 3
TOP_UNCONSTRAINED = 3
EPS = 1e-12


def _check_seed_pools() -> None:
    if len(SEARCH_SEED_POOL) < REFINE_N_SEEDS:
        raise RuntimeError(f"Need {REFINE_N_SEEDS} search seeds; found {len(SEARCH_SEED_POOL)}")
    if len(EVALUATION_SEED_POOL) < EVAL_N_SEEDS:
        raise RuntimeError(f"Need {EVAL_N_SEEDS} evaluation seeds; found {len(EVALUATION_SEED_POOL)}")


def _sharded_rows(bank: pd.DataFrame, shard_index: int, shard_count: int) -> pd.DataFrame:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard index/count")
    return (
        bank.sort_values("background_id", kind="stable")
        .reset_index(drop=True)
        .iloc[shard_index::shard_count]
    )


def _task_key(task: dict, columns: Iterable[str]) -> tuple:
    return tuple(task["seed"] if c == "seed" else task["extra_cols"][c] for c in columns)


def _dedupe_tasks(tasks: list[dict], columns: Iterable[str]) -> list[dict]:
    seen, out = set(), []
    for task in tasks:
        key = _task_key(task, columns)
        if key not in seen:
            seen.add(key)
            out.append(task)
    return out


def _dedupe_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    cols = ["stage", *CELL_COLS, "seed"]
    sort_cols = cols + (["arm"] if "arm" in frame.columns else [])
    return frame.sort_values(sort_cols, kind="stable").drop_duplicates(cols, keep="first").copy()


def _add_rates(frame: pd.DataFrame) -> pd.DataFrame:
    z = frame.copy()
    required = {"class_1_arrivals", "class_2_arrivals", "class_1_served", "class_2_served"}
    missing = sorted(required - set(z.columns))
    if missing:
        raise KeyError(f"Raw file missing count columns: {missing}")
    a1 = z["class_1_arrivals"].to_numpy(float)
    a2 = z["class_2_arrivals"].to_numpy(float)
    y1 = z["class_1_served"].to_numpy(float)
    y2 = z["class_2_served"].to_numpy(float)
    z["sr1"] = np.divide(y1, a1, out=np.zeros_like(y1), where=a1 > 0)
    z["sr2"] = np.divide(y2, a2, out=np.zeros_like(y2), where=a2 > 0)
    at = a1 + a2
    z["overall_sr"] = np.divide(y1 + y2, at, out=np.zeros_like(y1), where=at > 0)
    return z


def _complete_cell_means(frame: pd.DataFrame, seeds: tuple[int, ...]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    z = _add_rates(_dedupe_rows(frame))
    z = z[z["seed"].isin(seeds)].copy()
    if z.empty:
        return pd.DataFrame()
    g = z.groupby(CELL_COLS, as_index=False).agg(
        overall_sr=("overall_sr", "mean"), sr1=("sr1", "mean"), sr2=("sr2", "mean"),
        n_seeds=("seed", "nunique"),
    )
    return g[g["n_seeds"] == len(seeds)].copy()


def _baseline_rates(search: pd.DataFrame, seeds: tuple[int, ...]) -> tuple[float, float, float]:
    g = _complete_cell_means(search[search["stage"] == "baseline"], seeds)
    if len(g) != 1:
        raise ValueError(f"Expected one complete baseline cell on {len(seeds)} seeds; found {len(g)}")
    r = g.iloc[0]
    return float(r.overall_sr), float(r.sr1), float(r.sr2)


def _broad_candidates(search: pd.DataFrame, policy: str, native_h: int, seeds: tuple[int, ...]) -> pd.DataFrame:
    """Use only exact/coarse search data; never use prior selected winners."""
    search = _dedupe_rows(search)
    if policy == "horizon_only":
        raw = search[(search.stage == "horizon_only") & (search.Q == 0) & search.horizon_days.between(2, 30)].copy()
    elif policy == "reservation_only":
        raw = search[(search.stage == "reservation_only") & (search.arm == "coarse") & (search.horizon_days == native_h)].copy()
        raw = pd.concat([raw, search[search.stage == "baseline"]], ignore_index=True)
    elif policy == "both_flexible":
        raw = search[(search.stage == "both_flexible") & (search.arm == "coarse") & search.horizon_days.between(2, 30)].copy()
        raw = pd.concat([raw, search[(search.stage == "horizon_only") & (search.Q == 0) & search.horizon_days.between(2, 30)]], ignore_index=True)
    else:
        raise ValueError(policy)
    g = _complete_cell_means(raw, seeds)
    if g.empty:
        return g
    _, b1, b2 = _baseline_rates(search, seeds)
    g["d1"] = g.sr1 - b1
    g["d2"] = g.sr2 - b2
    g["worst"] = np.minimum(g.d1, g.d2)
    return g


def _sort_candidates(z: pd.DataFrame) -> pd.DataFrame:
    return z.sort_values(
        ["overall_sr", "worst", "Q", "window", "horizon_days"],
        ascending=[False, False, True, True, True], kind="stable"
    ) if not z.empty else z


def _target_rows(g: pd.DataFrame) -> list[tuple[pd.Series, str]]:
    if g.empty:
        return []
    out = []
    feasible = g[(g.d1 >= -EPS) & (g.d2 >= -EPS)]
    for _, r in _sort_candidates(feasible).head(TOP_FEASIBLE).iterrows():
        out.append((r, "feasible"))
    near = g[g.worst >= -NEAR_FEASIBLE_MARGIN - EPS]
    for _, r in _sort_candidates(near).head(TOP_NEAR).iterrows():
        out.append((r, "near_feasible"))
    for _, r in _sort_candidates(g).head(TOP_UNCONSTRAINED).iterrows():
        out.append((r, "unconstrained"))
    return out


def _neighborhood(target: tuple[int, int, int], capacity: int) -> set[tuple[int, int, int]]:
    h, q, w = target
    cells = {(h, q, w)}
    if q > 0:
        for q2 in q_fine_grid(q, capacity):
            cells.add((h, int(q2), w))
        for w2 in window_fine_grid(w, h):
            cells.add((h, q, int(w2)))
    return cells


def command_extend(args: argparse.Namespace) -> None:
    """Add H=27..30 broad cells on the original five search seeds."""
    _check_seed_pools()
    seeds = tuple(SEARCH_SEED_POOL[:BROAD_N_SEEDS])
    rows = _sharded_rows(load_bank(args.bank), args.shard_index, args.shard_count)
    raw_dir = args.output_dir / "search" / "raw"
    total = pending_total = 0
    for _, row in rows.iterrows():
        bg = str(row.background_id)
        path = raw_dir / f"{bg}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing copied legacy raw shard: {path}")
        tasks = []
        for h in range(27, 31):
            for seed in seeds:
                tasks.append(_make_task(row, horizon=h, q=0, window=-1, seed=seed,
                                        stage="horizon_only", phase="exact", smoke=False))
        for h in range(27, 31):
            for q in q_coarse_grid(int(row.slots_per_day)):
                for w in window_coarse_grid(h):  # retains existing W<=26 rule
                    for seed in seeds:
                        tasks.append(_make_task(row, horizon=h, q=int(q), window=int(w), seed=seed,
                                                stage="both_flexible", phase="coarse", smoke=False))
        tasks = _dedupe_tasks(tasks, SEARCH_KEY_COLUMNS)
        pending = _filter_pending(tasks, path, SEARCH_KEY_COLUMNS)
        total += len(tasks); pending_total += len(pending)
        if not args.dry_run and pending:
            print(f"{bg}: extension {len(pending):,}/{len(tasks):,} pending")
            run_tasks(pending, path, args.workers)
    print(f"Extension planned runs: {total:,}\nExtension pending runs: {pending_total:,}")
    if args.dry_run: print("DRY RUN ONLY")


def command_plan(args: argparse.Namespace) -> None:
    """Plan a fresh served-rate refinement from broad raw data."""
    _check_seed_pools()
    seeds = tuple(SEARCH_SEED_POOL[:BROAD_N_SEEDS])
    bank = load_bank(args.bank).sort_values("background_id", kind="stable")
    raw_dir = args.output_dir / "search" / "raw"
    manifest_rows, diag_rows = [], []
    for i, (_, row) in enumerate(bank.iterrows(), start=1):
        bg, native_h, cap = str(row.background_id), int(row.horizon_days), int(row.slots_per_day)
        search = pd.read_csv(raw_dir / f"{bg}.csv")
        for policy in POLICIES:
            g = _broad_candidates(search, policy, native_h, seeds)
            if g.empty:
                raise ValueError(f"{bg}/{policy}: no complete broad candidates")
            cell_reasons: dict[tuple[int, int, int], set[str]] = {}
            for r, reason in _target_rows(g):
                target = (int(r.horizon_days), int(r.Q), int(r.window))
                cells = {target} if policy == "horizon_only" else _neighborhood(target, cap)
                for cell in cells:
                    cell_reasons.setdefault(cell, set()).add(reason)
            for (h, q, w), reasons in sorted(cell_reasons.items()):
                manifest_rows.append(dict(background_id=bg, policy=policy, horizon_days=h, Q=q,
                                          window=w, reasons=";".join(sorted(reasons))))
            feasible = g[(g.d1 >= -EPS) & (g.d2 >= -EPS)]
            best_f = _sort_candidates(feasible).head(1)
            best_u = _sort_candidates(g).head(1)
            diag_rows.append(dict(
                background_id=bg, policy=policy, n_broad_complete=len(g), n_broad_feasible=len(feasible),
                best_feasible_overall_sr=(float(best_f.overall_sr.iloc[0]) if not best_f.empty else np.nan),
                best_unconstrained_overall_sr=float(best_u.overall_sr.iloc[0]),
                best_broad_worst_gain=(float(best_f.worst.iloc[0]) if not best_f.empty else np.nan),
            ))
        if i % 50 == 0 or i == len(bank): print(f"Planned {i}/{len(bank)} backgrounds")
    out = args.output_dir / "served_rate_pareto"; out.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows).drop_duplicates(["background_id", "policy", *CELL_COLS])
    manifest.to_csv(out / "refinement_manifest.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(out / "planning_diagnostics.csv", index=False)
    print(f"Manifest cells: {len(manifest):,}\nWrote: {out / 'refinement_manifest.csv'}")


def command_refine(args: argparse.Namespace) -> None:
    """Bring baseline, every H-only cell, and target neighborhoods to 10 seeds."""
    _check_seed_pools()
    seeds = tuple(SEARCH_SEED_POOL[:REFINE_N_SEEDS])
    rows = _sharded_rows(load_bank(args.bank), args.shard_index, args.shard_count)
    raw_dir = args.output_dir / "search" / "raw"
    manifest_path = args.output_dir / "served_rate_pareto" / "refinement_manifest.csv"
    if not manifest_path.exists(): raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    total = pending_total = 0
    for _, row in rows.iterrows():
        bg, native_h = str(row.background_id), int(row.horizon_days)
        path = raw_dir / f"{bg}.csv"
        tasks = []
        for seed in seeds:
            tasks.append(_make_task(row, horizon=native_h, q=0, window=-1, seed=seed,
                                    stage="baseline", phase="exact", smoke=False))
        for h in HORIZONS:
            for seed in seeds:
                tasks.append(_make_task(row, horizon=h, q=0, window=-1, seed=seed,
                                        stage="horizon_only", phase="exact", smoke=False))
        m = manifest[(manifest.background_id == bg) & manifest.policy.isin(["reservation_only", "both_flexible"])]
        for r in m.itertuples(index=False):
            for seed in seeds:
                tasks.append(_make_task(row, horizon=int(r.horizon_days), q=int(r.Q), window=int(r.window),
                                        seed=seed, stage=str(r.policy), phase="fine_served_rate_pareto", smoke=False))
        tasks = _dedupe_tasks(tasks, SEARCH_KEY_COLUMNS)
        pending = _filter_pending(tasks, path, SEARCH_KEY_COLUMNS)
        total += len(tasks); pending_total += len(pending)
        if not args.dry_run and pending:
            print(f"{bg}: refine {len(pending):,}/{len(tasks):,} pending")
            run_tasks(pending, path, args.workers)
    print(f"Refinement planned runs: {total:,}\nRefinement pending runs: {pending_total:,}")
    if args.dry_run: print("DRY RUN ONLY")


def _manifest_cells(manifest: pd.DataFrame, bg: str, policy: str) -> set[tuple[int, int, int]]:
    m = manifest[(manifest.background_id == bg) & (manifest.policy == policy)]
    return {(int(r.horizon_days), int(r.Q), int(r.window)) for r in m.itertuples(index=False)}


def _selection_candidates(search: pd.DataFrame, manifest: pd.DataFrame, bg: str,
                          policy: str, native_h: int, seeds: tuple[int, ...]) -> pd.DataFrame:
    search = _dedupe_rows(search)
    if policy == "horizon_only":
        raw = search[(search.stage == "horizon_only") & (search.Q == 0) & search.horizon_days.between(2, 30)].copy()
    else:
        cells = _manifest_cells(manifest, bg, policy)
        tuples = list(zip(search.horizon_days.astype(int), search.Q.astype(int), search.window.astype(int)))
        mask = pd.Series([x in cells for x in tuples], index=search.index)
        raw = search[(search.stage == policy) & mask].copy()
        if policy == "reservation_only":
            raw = pd.concat([raw, search[search.stage == "baseline"]], ignore_index=True)
        else:
            raw = pd.concat([raw, search[(search.stage == "horizon_only") & (search.Q == 0) & search.horizon_days.between(2, 30)]], ignore_index=True)
    g = _complete_cell_means(raw, seeds)
    if g.empty:
        raise ValueError(f"{bg}/{policy}: no complete 10-seed candidates")

    observed = {
        (int(r.horizon_days), int(r.Q), int(r.window))
        for r in g.itertuples(index=False)
    }
    if policy == "horizon_only":
        expected = {(h, 0, -1) for h in HORIZONS}
    elif policy == "reservation_only":
        expected = _manifest_cells(manifest, bg, policy) | {(native_h, 0, -1)}
    else:
        expected = _manifest_cells(manifest, bg, policy) | {(h, 0, -1) for h in HORIZONS}
    missing = expected - observed
    if missing:
        raise ValueError(
            f"{bg}/{policy}: refinement incomplete; "
            f"{len(missing)} expected 10-seed cells missing, examples={sorted(missing)[:5]}"
        )

    b0, b1, b2 = _baseline_rates(search, seeds)
    g["delta_overall"] = g.overall_sr - b0; g["d1"] = g.sr1 - b1; g["d2"] = g.sr2 - b2
    g["worst"] = np.minimum(g.d1, g.d2)
    return g


def _selection_record(bg: str, policy: str, stype: str, r: pd.Series,
                      baseline: tuple[float, float, float], rank: int = 1) -> dict:
    b0, b1, b2 = baseline
    h, q, w = int(r.horizon_days), int(r.Q), int(r.window)
    return dict(
        background_id=bg, policy=policy, selection_type=stype, rank=rank,
        selected_horizon_days=h, selected_Q=q, selected_window=w,
        search_overall_served_rate=float(r.overall_sr), search_class_1_served_rate=float(r.sr1),
        search_class_2_served_rate=float(r.sr2), search_baseline_overall_served_rate=b0,
        search_baseline_class_1_served_rate=b1, search_baseline_class_2_served_rate=b2,
        search_delta_overall=float(r.overall_sr - b0), search_delta_sr1=float(r.sr1 - b1),
        search_delta_sr2=float(r.sr2 - b2), search_worst_class_gain=float(min(r.sr1 - b1, r.sr2 - b2)),
        search_pareto_eligible=bool(r.sr1 - b1 >= -EPS and r.sr2 - b2 >= -EPS),
        genuinely_active_reservation=bool(q > 0 and w >= 1), search_n_seeds=REFINE_N_SEEDS,
    )


def command_select(args: argparse.Namespace) -> None:
    _check_seed_pools()
    seeds = tuple(SEARCH_SEED_POOL[:REFINE_N_SEEDS])
    bank = load_bank(args.bank).sort_values("background_id", kind="stable")
    raw_dir = args.output_dir / "search" / "raw"; out = args.output_dir / "served_rate_pareto"
    manifest = pd.read_csv(out / "refinement_manifest.csv")
    selected_rows, top3_rows = [], []
    for i, (_, row) in enumerate(bank.iterrows(), start=1):
        bg, native_h = str(row.background_id), int(row.horizon_days)
        search = pd.read_csv(raw_dir / f"{bg}.csv"); baseline = _baseline_rates(search, seeds)
        for policy in POLICIES:
            g = _selection_candidates(search, manifest, bg, policy, native_h, seeds)
            feasible = g[(g.d1 >= -EPS) & (g.d2 >= -EPS)].copy()
            if feasible.empty: raise RuntimeError(f"{bg}/{policy}: no eligible cell")
            ranked = _sort_candidates(feasible)
            for rank, (_, rr) in enumerate(ranked.head(3).iterrows(), start=1):
                top3_rows.append(_selection_record(bg, policy, "pareto", rr, baseline, rank))
            selected_rows.append(_selection_record(bg, policy, "pareto", ranked.iloc[0], baseline))
            selected_rows.append(_selection_record(bg, policy, "unconstrained", _sort_candidates(g).iloc[0], baseline))
        if i % 50 == 0 or i == len(bank): print(f"Selected {i}/{len(bank)} backgrounds")
    selected = pd.DataFrame(selected_rows); top3 = pd.DataFrame(top3_rows)
    pareto = selected[selected.selection_type == "pareto"].copy()
    cross = pareto.sort_values(
        ["background_id", "search_overall_served_rate", "search_worst_class_gain", "selected_Q", "selected_window", "selected_horizon_days"],
        ascending=[True, False, False, True, True, True], kind="stable"
    ).groupby("background_id", as_index=False).head(1).copy()
    cross["selection_type"] = "cross_policy_pareto"
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out / "search_selections.csv", index=False)
    top3.to_csv(out / "search_top3_pareto.csv", index=False)
    cross.to_csv(out / "cross_policy_search_winner.csv", index=False)
    print(f"Wrote {len(selected):,} policy selections, {len(top3):,} top-3 rows, {len(cross):,} cross-policy winners")


def command_evaluate(args: argparse.Namespace) -> None:
    _check_seed_pools()
    seeds = tuple(EVALUATION_SEED_POOL[:EVAL_N_SEEDS])
    rows = _sharded_rows(load_bank(args.bank), args.shard_index, args.shard_count)
    out = args.output_dir / "served_rate_pareto"; selected = pd.read_csv(out / "search_selections.csv")
    raw_dir = out / "evaluation" / "raw"; total = pending_total = 0
    for _, row in rows.iterrows():
        bg, native_h = str(row.background_id), int(row.horizon_days)
        cells = {(native_h, 0, -1)}
        for r in selected[selected.background_id == bg].itertuples(index=False):
            cells.add((int(r.selected_horizon_days), int(r.selected_Q), int(r.selected_window)))
        tasks = _task_grid(row, stage="evaluation", phase="served_rate_pareto_validation",
                           cells=sorted(cells), seeds=seeds, smoke=False)
        path = shard_path(raw_dir, bg); pending = _filter_pending(tasks, path, EVALUATION_KEY_COLUMNS)
        total += len(tasks); pending_total += len(pending)
        if not args.dry_run and pending:
            print(f"{bg}: evaluation {len(pending):,}/{len(tasks):,} pending")
            run_tasks(pending, path, args.workers)
    print(f"Evaluation planned runs: {total:,}\nEvaluation pending runs: {pending_total:,}")
    if args.dry_run: print("DRY RUN ONLY")


def _stable_int(*parts: object) -> int:
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % (2**32 - 1)


def _paired_ci(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed); n = len(values); means = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, size=n); means[i] = values[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975]); return float(lo), float(hi)


def _eval_cell(evaluation: pd.DataFrame, cell: tuple[int, int, int], seeds: tuple[int, ...]) -> pd.DataFrame:
    h, q, w = cell; z = _add_rates(_dedupe_rows(evaluation))
    z = z[(z.horizon_days == h) & (z.Q == q) & (z.window == w) & z.seed.isin(seeds)].copy()
    if z.seed.nunique() != len(seeds):
        raise ValueError(f"Cell {cell} has {z.seed.nunique()} eval seeds; expected {len(seeds)}")
    return z.sort_values("seed")


def command_postprocess(args: argparse.Namespace) -> None:
    _check_seed_pools()
    seeds = tuple(EVALUATION_SEED_POOL[:EVAL_N_SEEDS])
    bank = load_bank(args.bank); bank_idx = bank.set_index("background_id")
    out = args.output_dir / "served_rate_pareto"; selected = pd.read_csv(out / "search_selections.csv")
    raw_dir = out / "evaluation" / "raw"; rows, seed_rows = [], []
    for i, s in enumerate(selected.itertuples(index=False), start=1):
        bg = str(s.background_id); ev = pd.read_csv(raw_dir / f"{bg}.csv")
        native_h = int(bank_idx.loc[bg, "horizon_days"])
        b = _eval_cell(ev, (native_h, 0, -1), seeds).set_index("seed").loc[list(seeds)]
        c = _eval_cell(ev, (int(s.selected_horizon_days), int(s.selected_Q), int(s.selected_window)), seeds).set_index("seed").loc[list(seeds)]
        d0 = (c.overall_sr - b.overall_sr).to_numpy(float); d1 = (c.sr1 - b.sr1).to_numpy(float); d2 = (c.sr2 - b.sr2).to_numpy(float)
        ci0 = _paired_ci(d0, args.bootstrap, _stable_int(bg, s.policy, s.selection_type, "overall"))
        ci1 = _paired_ci(d1, args.bootstrap, _stable_int(bg, s.policy, s.selection_type, "sr1"))
        ci2 = _paired_ci(d2, args.bootstrap, _stable_int(bg, s.policy, s.selection_type, "sr2"))
        rec = dict(s._asdict()); rec.update(dict(
            eval_n_seeds=len(seeds), eval_overall_served_rate=float(c.overall_sr.mean()),
            eval_class_1_served_rate=float(c.sr1.mean()), eval_class_2_served_rate=float(c.sr2.mean()),
            eval_baseline_overall_served_rate=float(b.overall_sr.mean()),
            eval_baseline_class_1_served_rate=float(b.sr1.mean()), eval_baseline_class_2_served_rate=float(b.sr2.mean()),
            eval_delta_overall=float(d0.mean()), eval_delta_overall_ci_low=ci0[0], eval_delta_overall_ci_high=ci0[1],
            eval_delta_sr1=float(d1.mean()), eval_delta_sr1_ci_low=ci1[0], eval_delta_sr1_ci_high=ci1[1],
            eval_delta_sr2=float(d2.mean()), eval_delta_sr2_ci_low=ci2[0], eval_delta_sr2_ci_high=ci2[1],
            eval_point_pareto_eligible=bool(d1.mean() >= -EPS and d2.mean() >= -EPS),
            eval_supported_both_improve=bool(ci1[0] > 0 and ci2[0] > 0),
            eval_at_least_one_positive=bool(d1.mean() > 0 or d2.mean() > 0),
        )); rows.append(rec)
        for seed, x0, x1, x2 in zip(seeds, d0, d1, d2):
            seed_rows.append(dict(background_id=bg, policy=s.policy, selection_type=s.selection_type,
                                  seed=seed, delta_overall=x0, delta_sr1=x1, delta_sr2=x2))
        if i % 200 == 0 or i == len(selected): print(f"Postprocessed {i}/{len(selected)} selections")
    validation = pd.DataFrame(rows); seed_deltas = pd.DataFrame(seed_rows)
    p = validation[validation.selection_type == "pareto"].copy(); u = validation[validation.selection_type == "unconstrained"].copy()
    cost = p[["background_id", "policy", "eval_overall_served_rate"]].merge(
        u[["background_id", "policy", "eval_overall_served_rate"]], on=["background_id", "policy"],
        suffixes=("_pareto", "_unconstrained"), validate="one_to_one")
    cost["validation_fairness_cost"] = cost.eval_overall_served_rate_unconstrained - cost.eval_overall_served_rate_pareto
    summary = p.groupby("policy", as_index=False).agg(
        n_backgrounds=("background_id", "size"), point_pareto_share=("eval_point_pareto_eligible", "mean"),
        supported_both_improve_share=("eval_supported_both_improve", "mean"), median_delta_overall=("eval_delta_overall", "median"),
        median_delta_sr1=("eval_delta_sr1", "median"), median_delta_sr2=("eval_delta_sr2", "median"),
    ).merge(cost.groupby("policy", as_index=False).validation_fairness_cost.median(), on="policy", how="left")
    validation.to_csv(out / "validation_results.csv", index=False); seed_deltas.to_csv(out / "validation_seed_deltas.csv", index=False)
    cost.to_csv(out / "fairness_cost.csv", index=False); summary.to_csv(out / "validation_summary.csv", index=False)
    print("\nValidation summary\n" + summary.to_string(index=False)); print(f"\nOutputs: {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("extend", "plan", "refine", "select", "evaluate", "postprocess"))
    p.add_argument("--bank", type=Path, default=Path("outputs/hypotheses/patient_behavior_factorial_bank.csv"))
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--workers", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0); p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--bootstrap", type=int, default=2000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    {"extend": command_extend, "plan": command_plan, "refine": command_refine,
     "select": command_select, "evaluate": command_evaluate, "postprocess": command_postprocess}[args.command](args)


if __name__ == "__main__":
    main()
