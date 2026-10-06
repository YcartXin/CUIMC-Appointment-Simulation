#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "/scratch/yy3694/"
    "patient_behavior_factorial_served_rate_pareto"
)

CANDIDATES = (
    ROOT
    / "served_rate_pareto"
    / "pareto_catalog_5seed"
    / "meaningful_pareto_cells_5seed.csv"
)

EVAL_RAW = (
    ROOT
    / "served_rate_pareto"
    / "pareto_catalog_5seed"
    / "independent_evaluation"
    / "raw"
)

SEARCH_RAW = ROOT / "search" / "raw"

OUT = (
    ROOT
    / "served_rate_pareto"
    / "pareto_catalog_5seed"
    / "independent_evaluation"
)

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
