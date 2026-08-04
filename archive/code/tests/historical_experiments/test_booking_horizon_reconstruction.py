"""
Regression guard for the reconstructed booking-horizon sweep.

The original sweep code behind the recovered historical outputs (commit
2aa83d3) was never committed; it was rebuilt in
experiments/sweep_booking_horizon.py from the
committed engine.  This test pins that reconstruction to a sample of the
recovered historical raw data (commit 2aa83d3, arrival rate 50): per-run counts
must match exactly, except that the served count is allowed to differ by at most
one patient per class because of a single booking resolving on the
measurement-window boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from experiments.sweep_booking_horizon import run_task

FIXTURE = Path(__file__).parent / "fixtures" / "booking_horizon_recovered_sample.csv"
EXACT_FIELDS = ["arrivals", "booked", "balked", "no_offer", "canceled", "no_show"]
ARRIVAL_RATE = 50


def _configs():
    df = pd.read_csv(FIXTURE)
    keys = df[["h1", "h2", "tau1", "tau2", "balk_high", "seed"]].drop_duplicates()
    return [tuple(r) for r in keys.itertuples(index=False)]


@pytest.mark.parametrize("cfg", _configs())
def test_sweep_reproduces_recovered(cfg):
    h1, h2, tau1, tau2, balk_high, seed = cfg
    rec = pd.read_csv(FIXTURE)
    class_rows, _ = run_task((h1, h2, tau1, tau2, balk_high, ARRIVAL_RATE, seed))
    sim = {row["class_id"]: row for row in class_rows}

    for cid in (1, 2):
        r = rec[(rec.h1 == h1) & (rec.h2 == h2) & (rec.tau1 == tau1) &
                (rec.tau2 == tau2) & (rec.balk_high == balk_high) &
                (rec.seed == seed) & (rec.class_id == cid)].iloc[0]
        for f in EXACT_FIELDS:
            assert int(sim[cid][f]) == int(r[f]), (
                f"{f} mismatch for {cfg} class {cid}: {sim[cid][f]} != {int(r[f])}")
        assert abs(int(sim[cid]["served"]) - int(r["served"])) <= 1, (
            f"served off by >1 for {cfg} class {cid}")
