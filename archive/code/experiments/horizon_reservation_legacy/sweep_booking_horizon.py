"""
Step 1: Simulation sweep for the class-specific booking-horizon analysis.

Runs the pooled-FCFS clinic model over a grid of per-class booking horizons
(H1, H2), balking thresholds, post-threshold balking rates, and arrival rates,
across a fixed set of seeds.  Writes per-class and aggregate raw results.

The simulation conventions below were validated against the recovered
historical raw data (commit 2aa83d3): with these constants the
engine reproduces the historical per-run counts bit-for-bit (see
tests/test_booking_horizon_reconstruction.py).

Outputs
-------
outputs/booking_horizon/raw/class_results.csv
outputs/booking_horizon/raw/aggregate_results.csv
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from simulation.engine import ClinicAppointmentSimulation
from simulation.model import PatientClassParams, SimulationConfig, ThresholdRule

OUTPUT_DIR = REPO_DIR / "outputs" / "booking_horizon"
RAW_DIR = OUTPUT_DIR / "raw"

# ------------------------------------------------------------------
# Validated simulation conventions (do not change without re-validating)
# ------------------------------------------------------------------
SLOTS_PER_DAY = 32
BURN_IN_DAYS = 30
MEASURE_DAYS = 365
COOLDOWN_DAYS = 14
CANCEL_PROB = 0.10
BALK_LOW = 0.0
NO_SHOW_THRESHOLD = 6
NO_SHOW_LOW = 0.0
NO_SHOW_HIGH = 0.30

TOTAL_SLOTS = MEASURE_DAYS * SLOTS_PER_DAY  # measurement-window slot capacity

# ------------------------------------------------------------------
# Experimental design (policy + problem parameters)
# ------------------------------------------------------------------
# Policy parameters
H_VALUES = list(range(2, 15))                       # booking horizons 2..14
# Problem parameters
THRESHOLD_PAIRS = [(9, 9), (5, 9), (9, 5), (12, 12), (5, 12)]
BALK_HIGHS = [0.3, 0.5, 0.7]                         # post-threshold balking rate
ARRIVAL_RATES = [25, 50]                             # per-class arrivals/day
SEEDS = list(range(1, 31))                           # 30 seeds

# Weight regimes are applied later in post-processing (summarize step); they do
# not affect the simulation and so are not part of the sweep.

CLASS_FIELDS = [
    "arrivals", "booked", "balked", "offered", "no_offer", "canceled",
    "no_show", "served", "mean_accepted_booking_delay",
    "mean_offered_booking_delay", "percent_serviced", "slot_utilization",
    "balking_rate", "total_booking_delay", "total_offered_booking_delay",
]


def build_config(h1, h2, tau1, tau2, balk_high, arrival_rate, seed):
    def cls(class_id, horizon, tau):
        return PatientClassParams(
            class_id=class_id,
            lambda_per_day=arrival_rate,
            balk_prob=ThresholdRule(tau, BALK_LOW, balk_high),
            cancel_prob=CANCEL_PROB,
            no_show_prob=ThresholdRule(NO_SHOW_THRESHOLD, NO_SHOW_LOW, NO_SHOW_HIGH),
            value=1.0,
            horizon_days=int(horizon),
        )

    return SimulationConfig(
        slots_per_day=SLOTS_PER_DAY,
        horizon_days=max(int(h1), int(h2)),
        burn_in_days=BURN_IN_DAYS,
        measure_days=MEASURE_DAYS,
        cooldown_days=COOLDOWN_DAYS,
        classes={1: cls(1, h1, tau1), 2: cls(2, h2, tau2)},
        seed=int(seed),
    )


def run_task(task):
    """Run one configuration; return (two class rows, one aggregate row)."""
    h1, h2, tau1, tau2, balk_high, arrival_rate, seed = task
    res = ClinicAppointmentSimulation(
        build_config(h1, h2, tau1, tau2, balk_high, arrival_rate, seed)
    ).run()

    keys = dict(h1=h1, h2=h2, tau1=tau1, tau2=tau2, balk_high=balk_high,
                arrival_rate=arrival_rate, seed=seed)

    class_rows = []
    tot = dict(arrivals=0, booked=0, balked=0, offered=0, no_offer=0,
               canceled=0, no_show=0, served=0, booking_delay=0.0, offered_delay=0.0)
    for cid in (1, 2):
        m = res.class_metrics[cid]
        offered = m.offered
        row = dict(keys, class_id=cid,
                   arrivals=m.arrivals, booked=m.booked, balked=m.balked,
                   offered=offered, no_offer=m.no_offer, canceled=m.canceled,
                   no_show=m.no_show, served=m.served,
                   mean_accepted_booking_delay=m.mean_accepted_booking_delay,
                   mean_offered_booking_delay=m.mean_offered_booking_delay,
                   percent_serviced=(m.served / m.arrivals if m.arrivals else 0.0),
                   slot_utilization=m.served / TOTAL_SLOTS,
                   balking_rate=(m.balked / offered if offered else 0.0),
                   total_booking_delay=m.total_booking_delay,
                   total_offered_booking_delay=m.total_offered_booking_delay)
        class_rows.append(row)
        tot["arrivals"] += m.arrivals
        tot["booked"] += m.booked
        tot["balked"] += m.balked
        tot["offered"] += offered
        tot["no_offer"] += m.no_offer
        tot["canceled"] += m.canceled
        tot["no_show"] += m.no_show
        tot["served"] += m.served
        tot["booking_delay"] += m.total_booking_delay
        tot["offered_delay"] += m.total_offered_booking_delay

    unresolved = tot["booked"] - tot["canceled"] - tot["no_show"] - tot["served"]
    agg_row = dict(keys,
                   average_utilization=res.average_utilization,
                   overall_percent_serviced=res.overall_percent_serviced,
                   total_served=res.total_served,
                   total_value=res.total_value,
                   mean_accepted_booking_delay=(tot["booking_delay"] / tot["booked"]
                                                if tot["booked"] else 0.0),
                   mean_offered_booking_delay=(tot["offered_delay"] / tot["offered"]
                                               if tot["offered"] else 0.0),
                   overall_balking_rate=(tot["balked"] / tot["offered"]
                                         if tot["offered"] else 0.0),
                   total_arrivals=tot["arrivals"], total_booked=tot["booked"],
                   total_offered=tot["offered"], total_balked=tot["balked"],
                   total_no_offer=tot["no_offer"], total_canceled=tot["canceled"],
                   total_no_show=tot["no_show"], total_unresolved_booked=unresolved)
    return class_rows, agg_row


def build_tasks(smoke=False):
    h_values = [2, 7, 14] if smoke else H_VALUES
    thresholds = [(9, 9), (9, 5)] if smoke else THRESHOLD_PAIRS
    balks = [0.5] if smoke else BALK_HIGHS
    arrivals = [50] if smoke else ARRIVAL_RATES
    seeds = [1, 2] if smoke else SEEDS
    tasks = []
    for h1, h2 in itertools.product(h_values, h_values):
        for tau1, tau2 in thresholds:
            for bh in balks:
                for ar in arrivals:
                    for sd in seeds:
                        tasks.append((h1, h2, tau1, tau2, bh, ar, sd))
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="parallel worker processes (default: all cores)")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny subset for a quick end-to-end check")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(smoke=args.smoke)
    print(f"Booking-horizon sweep: {len(tasks):,} runs on {args.workers} workers")

    class_rows, agg_rows = [], []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_task, t) for t in tasks]
        for fut in as_completed(futures):
            crows, arow = fut.result()
            class_rows.extend(crows)
            agg_rows.append(arow)
            done += 1
            if done % 2000 == 0 or done == len(tasks):
                rate = done / (time.time() - t0)
                eta = (len(tasks) - done) / rate if rate else 0
                print(f"  {done:,}/{len(tasks):,}  ({rate:.0f}/s, ETA {eta/60:.1f} min)")

    sort_keys = ["h1", "h2", "tau1", "tau2", "balk_high", "arrival_rate", "seed"]
    class_df = pd.DataFrame(class_rows).sort_values(sort_keys + ["class_id"])
    agg_df = pd.DataFrame(agg_rows).sort_values(sort_keys)
    class_df.to_csv(RAW_DIR / "class_results.csv", index=False)
    agg_df.to_csv(RAW_DIR / "aggregate_results.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"  {RAW_DIR/'class_results.csv'}  ({len(class_df):,} rows)")
    print(f"  {RAW_DIR/'aggregate_results.csv'}  ({len(agg_df):,} rows)")


if __name__ == "__main__":
    main()
