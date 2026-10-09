#!/usr/bin/env python3
"""Experiment 2 follow-up: 54 backgrounds; systematic 5-seed discovery only."""
from __future__ import annotations
import argparse
import sys
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.patient_behavior_factorial import (
    SEARCH_SEED_POOL, EVALUATION_SEED_POOL, SEARCH_KEY_COLUMNS,
    EVALUATION_KEY_COLUMNS, _filter_pending, _task_grid, load_bank,
    run_tasks, shard_path, q_coarse_grid, window_coarse_grid,
)
from experiments.patient_behavior_factorial_bank import _profiles

def bank_frame():
    rows = []
    for profile in _profiles():
        for rho, capacity in product((1.7, 2.0), (30, 40, 50)):
            context = f"E2F_R{rho:g}_S80_C{capacity}"
            rows.append(dict(
                **profile, background_id=f"PBF_{profile['profile_id']}_{context}",
                clinic_context_id=context, design_note="experiment2_share80_followup",
                patient_characteristic="balk_noshow_factorial",
                class2_reference="fixed_late_thresholds", horizon_days=100,
                rho=rho, class1_share=0.8, slots_per_day=capacity,
                lambda_1=rho*capacity*0.8, lambda_2=rho*capacity*(1-0.8),
                cap_thresholds_to_horizon=False,
            ))
    bank = pd.DataFrame(rows)
    assert len(bank) == 54 and bank.background_id.nunique() == 54
    assert bank.profile_id.nunique() == 9 and bank.clinic_context_id.nunique() == 6
    return bank

def search_tasks(row):
    seeds = SEARCH_SEED_POOL[:5]
    tasks = _task_grid(row, stage="baseline", phase="exact",
                       cells=[(100, 0, -1)], seeds=seeds, smoke=False)
    cells = [(h, q, w) for h in range(2, 31)
             for q in q_coarse_grid(int(row.slots_per_day))
             for w in window_coarse_grid(h)]
    tasks += _task_grid(row, stage="both_flexible", phase="coarse",
                        cells=cells, seeds=seeds, smoke=False)
    return tasks

def checked_bank(path):
    actual = load_bank(path).sort_values('background_id').reset_index(drop=True)
    expected = bank_frame().sort_values('background_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)
    return actual

def do_search(args):
    bank = checked_bank(args.bank)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid shard')
    bank = bank.iloc[args.shard_index::args.shard_count]
    total = pending_total = 0
    for _, row in bank.iterrows():
        tasks = search_tasks(row)
        path = args.output_dir / 'search/raw' / f"{row.background_id}.csv"
        pending = _filter_pending(tasks, path, SEARCH_KEY_COLUMNS)
        total += len(tasks); pending_total += len(pending)
        if pending and not args.dry_run:
            run_tasks(pending, path, args.workers)
    print(f'Search planned runs: {total:,}\nSearch pending runs: {pending_total:,}')
    if args.dry_run: print('DRY RUN ONLY')

SEEDS = {1000, 1001, 1002, 1003, 1004}
CELL = ["horizon_days", "Q", "window"]
EPS = 1e-12

POLICIES = ("both_flexible",)
SYSTEMATIC_ARMS = {"exact", "coarse"}

ARM_RANK = {
    "exact": 0,
    "coarse": 1,
}

def dedupe(x):
    x = x.copy()
    x["_rank"] = x["arm"].map(ARM_RANK).fillna(99)
    return (
        x.sort_values(CELL + ["seed", "_rank"], kind="stable")
        .drop_duplicates(CELL + ["seed"], keep="first")
        .drop(columns="_rank")
    )

def process(path):
    raw = pd.read_csv(path)
    raw = raw[raw["seed"].isin(SEEDS)].copy()
    bg = str(raw["source_background_id"].iloc[0])

    b = dedupe(raw[raw["stage"] == "baseline"])
    if set(b["seed"]) != SEEDS:
        raise ValueError(f"{bg}: incomplete 5-seed baseline")

    base = b[[
        "seed",
        "class_1_percent_serviced",
        "class_2_percent_serviced",
        "average_utilization",
    ]].rename(columns={
        "class_1_percent_serviced": "baseline_sr1",
        "class_2_percent_serviced": "baseline_sr2",
        "average_utilization": "baseline_utilization",
    })

    out = []

    for policy in POLICIES:
        x = raw[
            (raw["stage"] == policy)
            & (raw["arm"].isin(SYSTEMATIC_ARMS))
        ].copy()

        # Genuine Class-1 reservation only.
        if policy in {"reservation_only", "both_flexible"}:
            x = x[
                (x["Q"] > 0)
                & (x["window"] >= 2)
            ].copy()

        if x.empty:
            continue

        x = dedupe(x)
        x = x.merge(base, on="seed", how="inner")

        x["d1_seed"] = (
            x["class_1_percent_serviced"] - x["baseline_sr1"]
        )
        x["d2_seed"] = (
            x["class_2_percent_serviced"] - x["baseline_sr2"]
        )

        g = (
            x.groupby(CELL, as_index=False)
            .agg(
                n_seeds=("seed", "nunique"),
                sr1=("class_1_percent_serviced", "mean"),
                sr2=("class_2_percent_serviced", "mean"),
                average_utilization=("average_utilization", "mean"),
                baseline_sr1=("baseline_sr1", "mean"),
                baseline_sr2=("baseline_sr2", "mean"),
                baseline_utilization=("baseline_utilization", "mean"),
                delta_sr1=("d1_seed", "mean"),
                delta_sr2=("d2_seed", "mean"),
            )
        )

        g = g[g["n_seeds"] == 5].copy()

        g["background_id"] = bg
        g["policy"] = policy

        g["pareto"] = (
            (g["delta_sr1"] >= -EPS)
            & (g["delta_sr2"] >= -EPS)
            & (
                (g["delta_sr1"] > EPS)
                | (g["delta_sr2"] > EPS)
            )
        )

        g["strict_both_improve"] = (
            (g["delta_sr1"] > EPS)
            & (g["delta_sr2"] > EPS)
        )

        g["worst_class_gain"] = np.minimum(
            g["delta_sr1"], g["delta_sr2"]
        )

        g["delta_utilization_vs_baseline"] = (
            g["average_utilization"] - g["baseline_utilization"]
        )

        out.append(g)

    return pd.concat(out, ignore_index=True)


def do_catalog(args):
    bank = checked_bank(args.bank)
    chunks = []
    for _, row in bank.iterrows():
        path = args.output_dir / 'search/raw' / f"{row.background_id}.csv"
        pending = _filter_pending(search_tasks(row), path, SEARCH_KEY_COLUMNS)
        if pending: raise ValueError(f"{row.background_id}: {len(pending)} search runs missing")
        chunks.append(process(path))
    cells = pd.concat(chunks, ignore_index=True)
    # Discovery threshold is exactly the historical meaningful-catalog screen.
    cells['meaningful_pareto'] = ((cells.delta_sr1 >= 0) & (cells.delta_sr2 >= 0)
        & (cells[['delta_sr1', 'delta_sr2']].max(axis=1) >= 0.005))
    dest = args.output_dir / 'served_rate_pareto/pareto_catalog_5seed'
    dest.mkdir(parents=True, exist_ok=True)
    cells.to_csv(dest / 'all_systematic_cells_5seed.csv', index=False)
    candidates = cells[cells.meaningful_pareto]
    candidates.to_csv(dest / 'meaningful_pareto_cells_5seed.csv', index=False)
    print(f'Systematic cells: {len(cells):,}; frozen candidates: {len(candidates):,}')

def do_postprocess(args):
    root = args.output_dir
    dest = root / 'served_rate_pareto/pareto_catalog_5seed'
    candidates = dest / 'meaningful_pareto_cells_5seed.csv'
    frame = pd.read_csv(candidates)
    if not frame.empty:
        # Reuse the historical postprocessor's exact calculations and bootstrap RNG.
        historical_postprocess(root)
    bank = checked_bank(args.bank)
    valpath = dest / 'independent_evaluation/validated_meaningful_pareto_cells.csv'
    if frame.empty:
        validated = set()
    else:
        validation = pd.read_csv(valpath)
        validated = set(validation.loc[validation.validated_meaningful_pareto, 'background_id'])
    bank['policy'] = 'both_flexible'
    bank['validated_pareto_exists'] = bank.background_id.isin(validated)
    bank.to_csv(dest / 'validated_background_policy_universe.csv', index=False)
    frequency = bank.groupby(['rho', 'class1_share', 'slots_per_day'], as_index=False).agg(
        backgrounds=('background_id', 'size'), validated_backgrounds=('validated_pareto_exists', 'sum'),
        validated_share=('validated_pareto_exists', 'mean'))
    frequency.to_csv(dest / 'followup_frequency_by_capacity_rho.csv', index=False)
    bank[['rho','slots_per_day','noshow_level','balk_level','validated_pareto_exists']].to_csv(
        dest / 'followup_behavior_breakdown.csv', index=False)
    print(frequency.to_string(index=False))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('bank','search','catalog','evaluate','postprocess'))
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.command == 'bank':
        args.bank.parent.mkdir(parents=True, exist_ok=True)
        bank_frame().to_csv(args.bank, index=False)
        print(f'Wrote 54 backgrounds: {args.bank}')
    elif args.command == 'search': do_search(args)
    elif args.command == 'catalog': do_catalog(args)
    elif args.command == 'postprocess': do_postprocess(args)
    else:
        checked_bank(args.bank)
        sys.argv = [sys.argv[0], '--bank', str(args.bank), '--output-dir', str(args.output_dir),
                    '--workers', str(args.workers), '--shard-index', str(args.shard_index),
                    '--shard-count', str(args.shard_count), '--n-seeds', '20']
        if args.dry_run: sys.argv.append('--dry-run')
        evaluate_main()

def baseline_cell(raw: pd.DataFrame) -> tuple[int, int, int]:
    x = raw[raw["stage"] == "baseline"][CELL].drop_duplicates()

    if len(x) != 1:
        raise ValueError(
            f"Expected one baseline cell, found {len(x)}"
        )

    r = x.iloc[0]
    return (
        int(r["horizon_days"]),
        int(r["Q"]),
        int(r["window"]),
    )


def evaluate_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not 1 <= args.n_seeds <= len(EVALUATION_SEED_POOL):
        raise ValueError("Invalid --n-seeds")

    seeds = tuple(EVALUATION_SEED_POOL[:args.n_seeds])

    root = args.output_dir.resolve()

    candidate_path = (
        root
        / "served_rate_pareto"
        / "pareto_catalog_5seed"
        / "meaningful_pareto_cells_5seed.csv"
    )

    search_raw = root / "search" / "raw"

    eval_raw = (
        root
        / "served_rate_pareto"
        / "pareto_catalog_5seed"
        / "independent_evaluation"
        / "raw"
    )
    eval_raw.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(candidate_path)
    candidate_bgs = set(candidates["background_id"].astype(str))

    bank = load_bank(args.bank)
    bank = bank[
        bank["background_id"].astype(str).isin(candidate_bgs)
    ].copy()

    bank = (
        bank.sort_values("background_id", kind="stable")
        .reset_index(drop=True)
    )

    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")

    bank = bank.iloc[
        args.shard_index :: args.shard_count
    ]

    total_cells = 0
    total_planned = 0
    total_pending = 0
    backgrounds_pending = 0

    for i, (_, bank_row) in enumerate(bank.iterrows(), 1):
        bg = str(bank_row["background_id"])

        c = candidates[
            candidates["background_id"].astype(str) == bg
        ]

        candidate_cells = {
            (
                int(r["horizon_days"]),
                int(r["Q"]),
                int(r["window"]),
            )
            for _, r in c.iterrows()
        }

        raw_path = search_raw / f"{bg}.csv"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

        raw = pd.read_csv(raw_path)

        # Add matched no-policy baseline.
        cells = set(candidate_cells)
        cells.add(baseline_cell(raw))
        cells = sorted(cells)

        tasks = _task_grid(
            bank_row,
            stage="evaluation",
            phase="meaningful_pareto_independent_evaluation",
            cells=cells,
            seeds=seeds,
            smoke=False,
        )

        path = shard_path(eval_raw, bg)

        pending = _filter_pending(
            tasks,
            path,
            EVALUATION_KEY_COLUMNS,
        )

        total_cells += len(cells)
        total_planned += len(tasks)
        total_pending += len(pending)
        backgrounds_pending += int(len(pending) > 0)

        if not args.dry_run and pending:
            print(
                f"{bg}: {len(cells)} cells; "
                f"{len(pending)} pending"
            )
            run_tasks(pending, path, args.workers)

        if args.dry_run and (
            i % 20 == 0 or i == len(bank)
        ):
            print(
                f"[{i}/{len(bank)}] "
                f"{total_cells} cumulative cells; "
                f"{total_pending} cumulative pending"
            )

    print("\nMeaningful-Pareto independent evaluation")
    print(f"Backgrounds:             {len(bank)}")
    print(f"Unique cells + baseline: {total_cells}")
    print(f"Planned simulation runs: {total_planned}")
    print(f"Pending simulation runs: {total_pending}")
    print(f"Backgrounds pending:     {backgrounds_pending}")

    if args.dry_run:
        print("DRY RUN ONLY")


def historical_postprocess(root):
    ROOT = root
    CANDIDATES = root / 'served_rate_pareto/pareto_catalog_5seed/meaningful_pareto_cells_5seed.csv'
    SEARCH_RAW = root / 'search/raw'
    OUT = root / 'served_rate_pareto/pareto_catalog_5seed/independent_evaluation'
    EVAL_RAW = OUT / 'raw'
    SEEDS = set(range(2000, 2020))
    CELL = ["horizon_days", "Q", "window"]
    BOOT = 2000
    RNG = np.random.default_rng(20260928)
    EPS = 1e-12
    MEANINGFUL = 0.005
    
    
    def dedupe(x):
        return (
            x.sort_values(CELL + ["seed"], kind="stable")
            .drop_duplicates(CELL + ["seed"], keep="first")
        )
    
    
    def ci_bootstrap(values):
        values = np.asarray(values, dtype=float)
        n = len(values)
    
        idx = RNG.integers(0, n, size=(BOOT, n))
        means = values[idx].mean(axis=1)
    
        return (
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        )
    
    
    def baseline_cell(bg):
        p = SEARCH_RAW / f"{bg}.csv"
        raw = pd.read_csv(p)
    
        b = raw[raw["stage"] == "baseline"][CELL].drop_duplicates()
    
        if len(b) != 1:
            raise ValueError(
                f"{bg}: expected one baseline cell, found {len(b)}"
            )
    
        r = b.iloc[0]
        return (
            int(r["horizon_days"]),
            int(r["Q"]),
            int(r["window"]),
        )
    
    
    candidates = pd.read_csv(CANDIDATES)
    
    rows = []
    seed_rows = []
    
    for i, (bg, c_bg) in enumerate(
        candidates.groupby("background_id", sort=False),
        1,
    ):
        bg = str(bg)
    
        path = EVAL_RAW / f"{bg}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
    
        ev = pd.read_csv(path)
        ev = ev[ev["seed"].isin(SEEDS)].copy()
        ev = dedupe(ev)
    
        bh, bq, bw = baseline_cell(bg)
    
        base = ev[
            (ev["horizon_days"] == bh)
            & (ev["Q"] == bq)
            & (ev["window"] == bw)
        ].copy()
    
        if set(base["seed"]) != SEEDS:
            raise ValueError(
                f"{bg}: baseline does not contain all 20 evaluation seeds"
            )
    
        base = base.set_index("seed")
    
        for _, cand in c_bg.iterrows():
            h = int(cand["horizon_days"])
            q = int(cand["Q"])
            w = int(cand["window"])
    
            x = ev[
                (ev["horizon_days"] == h)
                & (ev["Q"] == q)
                & (ev["window"] == w)
            ].copy()
    
            if set(x["seed"]) != SEEDS:
                raise ValueError(
                    f"{bg} {cand['policy']} {(h,q,w)}: "
                    "missing evaluation seeds"
                )
    
            x = x.set_index("seed").loc[sorted(SEEDS)]
            b = base.loc[sorted(SEEDS)]
    
            d1 = (
                x["class_1_percent_serviced"].to_numpy()
                - b["class_1_percent_serviced"].to_numpy()
            )
            d2 = (
                x["class_2_percent_serviced"].to_numpy()
                - b["class_2_percent_serviced"].to_numpy()
            )
            du = (
                x["average_utilization"].to_numpy()
                - b["average_utilization"].to_numpy()
            )
    
            m1 = float(d1.mean())
            m2 = float(d2.mean())
            mu = float(du.mean())
    
            l1, u1 = ci_bootstrap(d1)
            l2, u2 = ci_bootstrap(d2)
            lu, uu = ci_bootstrap(du)
    
            validated = (
                m1 >= -EPS
                and m2 >= -EPS
                and max(m1, m2) >= MEANINGFUL
            )
    
            strict_both = (
                m1 > EPS
                and m2 > EPS
                and max(m1, m2) >= MEANINGFUL
            )
    
            # Strong statistical evidence that neither class is harmed.
            ci_supported_nonharm = (
                l1 >= -EPS
                and l2 >= -EPS
            )
    
            # At least one class has statistically positive gain.
            ci_supported_gain = (
                l1 > EPS
                or l2 > EPS
            )
    
            rows.append({
                "background_id": bg,
                "policy": cand["policy"],
                "horizon_days": h,
                "Q": q,
                "window": w,
    
                "search_delta_sr1": cand["delta_sr1"],
                "search_delta_sr2": cand["delta_sr2"],
    
                "eval_delta_sr1": m1,
                "eval_delta_sr1_ci_low": l1,
                "eval_delta_sr1_ci_high": u1,
    
                "eval_delta_sr2": m2,
                "eval_delta_sr2_ci_low": l2,
                "eval_delta_sr2_ci_high": u2,
    
                "eval_delta_utilization": mu,
                "eval_delta_utilization_ci_low": lu,
                "eval_delta_utilization_ci_high": uu,
    
                "validated_meaningful_pareto": validated,
                "validated_strict_both_improve": strict_both,
    
                "ci_supported_nonharm_both": ci_supported_nonharm,
                "ci_supported_positive_gain": ci_supported_gain,
    
                "strongly_supported_pareto": (
                    validated
                    and ci_supported_nonharm
                    and ci_supported_gain
                ),
            })
    
            for seed, a, b2, u in zip(
                sorted(SEEDS), d1, d2, du
            ):
                seed_rows.append({
                    "background_id": bg,
                    "policy": cand["policy"],
                    "horizon_days": h,
                    "Q": q,
                    "window": w,
                    "seed": seed,
                    "delta_sr1": a,
                    "delta_sr2": b2,
                    "delta_utilization": u,
                })
    
        if i % 20 == 0 or i == candidates["background_id"].nunique():
            print(
                f"Processed {i}/"
                f"{candidates['background_id'].nunique()} backgrounds"
            )
    
    results = pd.DataFrame(rows)
    seed_deltas = pd.DataFrame(seed_rows)
    
    summary = (
        results.groupby("policy")
        .agg(
            discovery_candidates=("background_id", "size"),
            validated_cells=("validated_meaningful_pareto", "sum"),
            strongly_supported_cells=("strongly_supported_pareto", "sum"),
            validated_backgrounds=(
                "background_id",
                lambda x: results.loc[
                    x.index,
                    "validated_meaningful_pareto"
                ].groupby(results.loc[x.index, "background_id"]).max().sum()
            ),
        )
        .reset_index()
    )
    
    bg_summary = (
        results.groupby(["background_id", "policy"])
        .agg(
            n_candidates=("background_id", "size"),
            n_validated=("validated_meaningful_pareto", "sum"),
            validated_pareto_exists=(
                "validated_meaningful_pareto", "max"
            ),
            n_strongly_supported=("strongly_supported_pareto", "sum"),
            strongly_supported_exists=("strongly_supported_pareto", "max"),
        )
        .reset_index()
    )
    
    OUT.mkdir(parents=True, exist_ok=True)
    
    results.to_csv(
        OUT / "validated_meaningful_pareto_cells.csv",
        index=False,
    )
    
    seed_deltas.to_csv(
        OUT / "validated_meaningful_pareto_seed_deltas.csv",
        index=False,
    )
    
    bg_summary.to_csv(
        OUT / "validated_background_policy_summary.csv",
        index=False,
    )
    
    summary.to_csv(
        OUT / "validation_summary.csv",
        index=False,
    )
    
    print("\nVALIDATION SUMMARY")
    print(summary.to_string(index=False))
    
    print("\nBACKGROUNDS WITH VALIDATED MEANINGFUL PARETO")
    print(
        bg_summary.groupby("policy")[
            "validated_pareto_exists"
        ].agg(["sum", "count", "mean"])
    )
    
    print("\nBACKGROUNDS WITH STRONGLY SUPPORTED PARETO")
    print(
        bg_summary.groupby("policy")[
            "strongly_supported_exists"
        ].agg(["sum", "count", "mean"])
    )
    
    print(f"\nWrote outputs to: {OUT}")

if __name__ == '__main__': main()
