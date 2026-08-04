"""
Step 2: Summarize the booking-horizon sweep.

Reads the raw per-class and aggregate results and writes seed-averaged summaries
with 95% confidence intervals. Two objectives are evaluated, both additive and
applied as pure post-processing over a grid of Class-1 weights:

    objective - service rate
        U = w1*(Y1/A1) + w2*(Y2/A2)

    objective - average slot utilization
        U = w1*U1 + w2*U2,
        where Ui = Yi / (S * D)
        and D is the number of measurement days.

The Class-1 weight grid is the union of the headline regimes (0.25, 1, 2, 3)
and the finer 0.5-step grid used for the Class-2-threshold counterbalance
figure.

Outputs
-------
outputs/booking_horizon/summary/objectives.csv
outputs/booking_horizon/summary/class_summary.csv
outputs/booking_horizon/summary/no_offer_rates.csv
outputs/booking_horizon/summary/aggregate_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

OUTPUT_DIR = REPO_DIR / "outputs" / "booking_horizon"
RAW_DIR = OUTPUT_DIR / "raw"
SUMMARY_DIR = OUTPUT_DIR / "summary"

CONFIG_KEYS = ["h1", "h2", "tau1", "tau2", "balk_high", "arrival_rate"]
W2 = 1.0
WEIGHTS_W1 = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

CLASS_METRICS = [
    "percent_serviced",
    "slot_utilization",
    "balking_rate",
    "mean_offered_booking_delay",
    "mean_accepted_booking_delay",
]

AGG_METRICS = [
    "average_utilization",
    "overall_percent_serviced",
    "mean_offered_booking_delay",
    "overall_balking_rate",
    "total_served",
]


def _stats(grouped: pd.core.groupby.generic.SeriesGroupBy) -> pd.DataFrame:
    """mean/std/n/se/ci95 from a groupby on a single value column."""
    out = grouped.agg(["mean", "std", "count"])
    out.columns = ["mean", "std", "n"]
    out["se"] = out["std"] / np.sqrt(out["n"])
    out["ci95"] = 1.96 * out["se"]
    return out.reset_index()


def summarize_objectives(cls: pd.DataFrame) -> pd.DataFrame:
    # one row per (config, seed) with both classes side by side.
    # We pivot in the per-class average slot_utilization directly, so the
    # weighted slot objective remains normalized by the full measurement window
    # rather than scaling with the number of simulated days.
    wide = cls.pivot_table(
        index=CONFIG_KEYS + ["seed"],
        columns="class_id",
        values=["served", "arrivals", "slot_utilization"],
    )
    wide.columns = [f"{metric}_{cid}" for metric, cid in wide.columns]
    wide = wide.reset_index()

    wide["c1"] = wide["served_1"] / wide["arrivals_1"]
    wide["c2"] = wide["served_2"] / wide["arrivals_2"]

    records = []
    for w1 in WEIGHTS_W1:
        wide["obj_service_rate"] = w1 * wide["c1"] + W2 * wide["c2"]
        wide["obj_slot_util"] = (
            w1 * wide["slot_utilization_1"] + W2 * wide["slot_utilization_2"]
        )

        agg = {}
        for col in ["obj_service_rate", "obj_slot_util", "c1", "c2"]:
            agg[col] = _stats(wide.groupby(CONFIG_KEYS)[col]).set_index(CONFIG_KEYS)

        block = agg["obj_service_rate"][["mean", "std", "se", "ci95"]].add_prefix(
            "obj_service_rate_"
        )
        block = block.join(
            agg["obj_slot_util"][["mean", "std", "se", "ci95"]].add_prefix(
                "obj_slot_util_"
            )
        )
        block = block.join(
            agg["c1"][["mean", "std", "se", "ci95"]].add_prefix(
                "c1_served_rate_"
            )
        )
        block = block.join(
            agg["c2"][["mean", "std", "se", "ci95"]].add_prefix(
                "c2_served_rate_"
            )
        )
        block = block.reset_index()
        block.insert(len(CONFIG_KEYS), "weight_regime", f"w1={w1:g}")
        block.insert(len(CONFIG_KEYS) + 1, "w1", w1)
        block.insert(len(CONFIG_KEYS) + 2, "w2", W2)
        records.append(block)

    return pd.concat(records, ignore_index=True)


def summarize_class_metrics(cls: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for metric in CLASS_METRICS:
        s = _stats(cls.groupby(CONFIG_KEYS + ["class_id"])[metric])
        s["metric"] = metric
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def summarize_no_offer(cls: pd.DataFrame) -> pd.DataFrame:
    cls = cls.copy()
    cls["no_offer_rate"] = cls["no_offer"] / cls["arrivals"]
    return _stats(cls.groupby(CONFIG_KEYS + ["class_id"])["no_offer_rate"])


def summarize_aggregate(agg: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for metric in AGG_METRICS:
        s = _stats(agg.groupby(CONFIG_KEYS)[metric])
        s["metric"] = metric
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    cls = pd.read_csv(RAW_DIR / "class_results.csv")
    agg = pd.read_csv(RAW_DIR / "aggregate_results.csv")

    summarize_objectives(cls).to_csv(SUMMARY_DIR / "objectives.csv", index=False)
    summarize_class_metrics(cls).to_csv(SUMMARY_DIR / "class_summary.csv", index=False)
    summarize_no_offer(cls).to_csv(SUMMARY_DIR / "no_offer_rates.csv", index=False)
    summarize_aggregate(agg).to_csv(SUMMARY_DIR / "aggregate_summary.csv", index=False)

    print("Wrote summaries to", SUMMARY_DIR)
    for name in ["objectives", "class_summary", "no_offer_rates", "aggregate_summary"]:
        n = len(pd.read_csv(SUMMARY_DIR / f"{name}.csv"))
        print(f"  {name}.csv ({n:,} rows)")


if __name__ == "__main__":
    main()
