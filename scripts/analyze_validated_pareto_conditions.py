#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(
    "/scratch/yy3694/"
    "patient_behavior_factorial_served_rate_pareto/"
    "served_rate_pareto/pareto_catalog_5seed"
)

BANK = pd.read_csv(
    "outputs/hypotheses/patient_behavior_factorial_bank.csv"
)

VAL = ROOT / "independent_evaluation"

cells = pd.read_csv(
    VAL / "validated_meaningful_pareto_cells.csv"
)

# Keep only independently validated Pareto cells.
v = cells[cells["validated_meaningful_pareto"]].copy()

context_cols = [
    "background_id",
    "profile_id",
    "noshow_level",
    "balk_level",
    "rho",
    "class1_share",
    "slots_per_day",
    "lambda_1",
    "lambda_2",
]

v = v.merge(
    BANK[context_cols],
    on="background_id",
    how="left",
)

# Construct all 540 x 3 background-policy rows so denominators are correct.
policies = [
    "horizon_only",
    "reservation_only",
    "both_flexible",
]

universe = (
    BANK[context_cols]
    .assign(key=1)
    .merge(
        pd.DataFrame({"policy": policies, "key": 1}),
        on="key",
    )
    .drop(columns="key")
)

exists = (
    v.groupby(["background_id", "policy"])
    .agg(
        validated_pareto_exists=("background_id", "size"),
        n_validated_cells=("background_id", "size"),
        max_delta_sr1=("eval_delta_sr1", "max"),
        max_delta_sr2=("eval_delta_sr2", "max"),
        median_delta_utilization=("eval_delta_utilization", "median"),
    )
    .reset_index()
)

exists["validated_pareto_exists"] = True

u = universe.merge(
    exists,
    on=["background_id", "policy"],
    how="left",
)

u["validated_pareto_exists"] = (
    u["validated_pareto_exists"].fillna(False)
)
u["n_validated_cells"] = u["n_validated_cells"].fillna(0)

# Headline rho <= 2.5
h = u[u["rho"] <= 2.5].copy()
vh = v[v["rho"] <= 2.5].copy()


def table(factor):
    return (
        h.groupby(["policy", factor])
        .agg(
            backgrounds=("background_id", "nunique"),
            pareto_backgrounds=("validated_pareto_exists", "sum"),
            pareto_rate=("validated_pareto_exists", "mean"),
        )
        .reset_index()
    )


print("\nOVERALL HEADLINE")
print(
    h.groupby("policy")["validated_pareto_exists"]
    .agg(["sum", "count", "mean"])
)

for factor in [
    "rho",
    "class1_share",
    "slots_per_day",
    "noshow_level",
    "balk_level",
]:
    print(f"\nBY {factor.upper()}")
    print(table(factor).to_string(index=False))


print("\n3x3 BEHAVIOR GRID")
behavior = (
    h.groupby(["policy", "noshow_level", "balk_level"])
    .agg(
        backgrounds=("background_id", "nunique"),
        pareto_backgrounds=("validated_pareto_exists", "sum"),
        pareto_rate=("validated_pareto_exists", "mean"),
    )
    .reset_index()
)
print(behavior.to_string(index=False))


print("\nVALIDATED POLICY SETTINGS")
settings = (
    vh.groupby("policy")
    .agg(
        validated_cells=("background_id", "size"),
        validated_backgrounds=("background_id", "nunique"),
        median_H=("horizon_days", "median"),
        q25_H=("horizon_days", lambda x: x.quantile(.25)),
        q75_H=("horizon_days", lambda x: x.quantile(.75)),
        median_Q=("Q", "median"),
        q25_Q=("Q", lambda x: x.quantile(.25)),
        q75_Q=("Q", lambda x: x.quantile(.75)),
        median_W=("window", "median"),
        q25_W=("window", lambda x: x.quantile(.25)),
        q75_W=("window", lambda x: x.quantile(.75)),
        median_delta_sr1=("eval_delta_sr1", "median"),
        median_delta_sr2=("eval_delta_sr2", "median"),
        median_delta_utilization=("eval_delta_utilization", "median"),
    )
    .reset_index()
)
print(settings.to_string(index=False))

out = VAL / "condition_analysis"
out.mkdir(exist_ok=True)

u.to_csv(out / "validated_background_policy_universe.csv", index=False)
behavior.to_csv(out / "validated_pareto_behavior_3x3.csv", index=False)
settings.to_csv(out / "validated_pareto_policy_settings.csv", index=False)

for factor in [
    "rho",
    "class1_share",
    "slots_per_day",
    "noshow_level",
    "balk_level",
]:
    table(factor).to_csv(
        out / f"validated_pareto_by_{factor}.csv",
        index=False,
    )
