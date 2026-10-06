#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "/scratch/yy3694/patient_behavior_factorial_served_rate_pareto"
)

EVAL = (
    ROOT
    / "served_rate_pareto"
    / "pareto_catalog_5seed"
    / "independent_evaluation"
)

SEARCH_RAW = ROOT / "search" / "raw"
RAW = EVAL / "raw"

CELLS = pd.read_csv(
    EVAL / "validated_meaningful_pareto_cells.csv"
)

BANK = pd.read_csv(
    "outputs/hypotheses/patient_behavior_factorial_bank.csv"
)

SEEDS = list(range(2000, 2020))
CELL = ["horizon_days", "Q", "window"]

BOOT = 2000
RNG = np.random.default_rng(20260929)


def boot_ci(x):
    x = np.asarray(x, dtype=float)
    idx = RNG.integers(0, len(x), size=(BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def baseline_cell(bg):
    x = pd.read_csv(SEARCH_RAW / f"{bg}.csv")
    b = x[x["stage"] == "baseline"][CELL].drop_duplicates()

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


# Only independently validated both-flexible cells.
bf = CELLS[
    (CELLS["policy"] == "both_flexible")
    & (CELLS["validated_meaningful_pareto"])
].copy()

rows = []

for bg, g in bf.groupby("background_id", sort=False):

    ev = pd.read_csv(RAW / f"{bg}.csv")
    ev = ev[ev["seed"].isin(SEEDS)].copy()

    ev = (
        ev.sort_values(CELL + ["seed"])
        .drop_duplicates(CELL + ["seed"], keep="first")
    )

    bh, bq, bw = baseline_cell(bg)

    baseline = ev[
        (ev["horizon_days"] == bh)
        & (ev["Q"] == bq)
        & (ev["window"] == bw)
    ].set_index("seed").loc[SEEDS]

    for _, r in g.iterrows():

        H = int(r["horizon_days"])
        Q = int(r["Q"])
        W = int(r["window"])

        flex = ev[
            (ev["horizon_days"] == H)
            & (ev["Q"] == Q)
            & (ev["window"] == W)
        ].set_index("seed").loc[SEEDS]

        # Exact same horizon, but reservations removed.
        horizon = ev[
            (ev["horizon_days"] == H)
            & (ev["Q"] == 0)
            & (ev["window"] == -1)
        ].set_index("seed").loc[SEEDS]

        # -----------------------
        # Horizon vs baseline
        # -----------------------

        h1 = (
            horizon["class_1_percent_serviced"].to_numpy()
            - baseline["class_1_percent_serviced"].to_numpy()
        )

        h2 = (
            horizon["class_2_percent_serviced"].to_numpy()
            - baseline["class_2_percent_serviced"].to_numpy()
        )

        hu = (
            horizon["average_utilization"].to_numpy()
            - baseline["average_utilization"].to_numpy()
        )

        # -----------------------
        # Both-flexible vs horizon
        # = marginal reservation effect
        # -----------------------

        r1 = (
            flex["class_1_percent_serviced"].to_numpy()
            - horizon["class_1_percent_serviced"].to_numpy()
        )

        r2 = (
            flex["class_2_percent_serviced"].to_numpy()
            - horizon["class_2_percent_serviced"].to_numpy()
        )

        ru = (
            flex["average_utilization"].to_numpy()
            - horizon["average_utilization"].to_numpy()
        )

        l1, u1 = boot_ci(r1)
        l2, u2 = boot_ci(r2)
        lu, uu = boot_ci(ru)

        rows.append({
            "background_id": bg,
            "horizon_days": H,
            "Q": Q,
            "window": W,

            "horizon_vs_baseline_sr1": h1.mean(),
            "horizon_vs_baseline_sr2": h2.mean(),
            "horizon_vs_baseline_util": hu.mean(),

            "reservation_increment_sr1": r1.mean(),
            "reservation_increment_sr1_ci_low": l1,
            "reservation_increment_sr1_ci_high": u1,

            "reservation_increment_sr2": r2.mean(),
            "reservation_increment_sr2_ci_low": l2,
            "reservation_increment_sr2_ci_high": u2,

            "reservation_increment_util": ru.mean(),
            "reservation_increment_util_ci_low": lu,
            "reservation_increment_util_ci_high": uu,

            "reservation_increment_pareto": (
                r1.mean() >= 0
                and r2.mean() >= 0
                and (
                    r1.mean() > 0
                    or r2.mean() > 0
                )
            ),

            "reservation_significant_sr1_gain": l1 > 0,
            "reservation_significant_sr2_gain": l2 > 0,
            "reservation_significant_sr1_loss": u1 < 0,
            "reservation_significant_sr2_loss": u2 < 0,
        })


d = pd.DataFrame(rows)

context = [
    "background_id",
    "rho",
    "class1_share",
    "slots_per_day",
    "noshow_level",
    "balk_level",
]

d = d.merge(
    BANK[context],
    on="background_id",
    how="left",
)

# -----------------------
# Overall summary
# -----------------------

print("\nOVERALL MARGINAL RESERVATION EFFECT")
print("Validated both-flexible cells:", len(d))

print(
    "\nMedian reservation increment:"
)
print(
    d[
        [
            "reservation_increment_sr1",
            "reservation_increment_sr2",
            "reservation_increment_util",
        ]
    ].median()
)

print(
    "\nCells where reservation itself is Pareto-improving "
    "relative to same horizon:"
)
print(
    f"{d['reservation_increment_pareto'].sum()} / {len(d)} "
    f"= {d['reservation_increment_pareto'].mean():.3f}"
)

# -----------------------
# By clinic context
# -----------------------

summary = (
    d.groupby(
        ["rho", "class1_share", "slots_per_day"]
    )
    .agg(
        cells=("background_id", "size"),
        backgrounds=("background_id", "nunique"),

        median_horizon_vs_baseline_sr1=(
            "horizon_vs_baseline_sr1", "median"
        ),
        median_horizon_vs_baseline_sr2=(
            "horizon_vs_baseline_sr2", "median"
        ),

        median_reservation_increment_sr1=(
            "reservation_increment_sr1", "median"
        ),
        median_reservation_increment_sr2=(
            "reservation_increment_sr2", "median"
        ),
        median_reservation_increment_util=(
            "reservation_increment_util", "median"
        ),

        reservation_increment_pareto_share=(
            "reservation_increment_pareto", "mean"
        ),
    )
    .reset_index()
    .sort_values(
        ["rho", "class1_share", "slots_per_day"]
    )
)

print("\nBY CLINIC CONTEXT")
print(summary.to_string(index=False))

# -----------------------
# Special rho=1.2 context
# -----------------------

special = d[
    (d["rho"] == 1.2)
    & (d["class1_share"] == 0.1)
    & (d["slots_per_day"] == 50)
]

print("\nSPECIAL CONTEXT: rho=1.2, share1=0.1, capacity=50")

print(
    special[
        [
            "noshow_level",
            "balk_level",
            "horizon_days",
            "Q",
            "window",
            "horizon_vs_baseline_sr1",
            "horizon_vs_baseline_sr2",
            "reservation_increment_sr1",
            "reservation_increment_sr2",
            "reservation_increment_util",
        ]
    ]
    .sort_values(["noshow_level", "balk_level"])
    .to_string(index=False)
)

OUT = EVAL / "condition_analysis"
OUT.mkdir(exist_ok=True)

d.to_csv(
    OUT / "reservation_increment_cells.csv",
    index=False,
)

summary.to_csv(
    OUT / "reservation_increment_by_clinic_context.csv",
    index=False,
)
