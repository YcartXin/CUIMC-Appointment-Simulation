"""Shared background-scenario bank for the H1/H2 comprehensive sweeps.

This generates one reusable bank of "background" scenarios spanning every
shared dimension named in the H1/H2 experiment design at full user-specified
range, per-class asymmetric wherever that makes physical sense. It does not
run any simulations; it only produces the parameter table.

A full factorial over these dimensions is not run anywhere in this repo: it
is on the order of 10^8 combinations before either hypothesis's own focal
parameters or seeds enter. Instead this module follows the same approach as
experiments/robustness/scenario_design.py: scrambled-Sobol sampling mapped
onto the discrete value grids, with rejection filtering for the validity
constraints below, stratified evenly across horizon_days so every horizon
is represented at the same count regardless of how the other dimensions
happen to sample.

Both H1 and H2 consume the SAME bank. No-show threshold is sampled fully
independently per class (not forced class_1 < class_2), which is what makes
one bank usable for both: H1 needs a mix of threshold_1 < threshold_2 (its
stated condition) and threshold_1 >= threshold_2 (the condition-violating
control), while H2 wants both symmetric and asymmetric no-show pairs. Both
come out of the same unconstrained-ordering sample; each experiment script
labels which of its own rows satisfy its stated hypothesis condition at
classification time, rather than the bank enforcing a direction.

Validity constraints enforced during sampling (per class i in {1, 2}):
    balk_low_i    <  balk_high_i
    noshow_low_i  <  noshow_high_i
    balk_threshold_i > noshow_threshold_i

The low/high probability pairs are strictly ordered (not <=): a pre-
threshold probability equal to its own post-threshold probability would
describe a rule with no actual threshold effect (p(tau) is constant
regardless of tau), which defeats the purpose of a ThresholdRule and
would silently produce a "flat" background masquerading as a delay-
dependent one. BALK_LOW_VALUES/NOSHOW_LOW_VALUES top out at 0.3, one
step below their _HIGH_VALUES counterparts' maximum of 0.4, specifically
so every low-value candidate always has at least one larger high value
to pair with -- 0.4 itself is reserved as a post-threshold-only value
and is never sampled as a pre-threshold one.

Both balk_threshold_i and noshow_threshold_i are kept from ever exceeding
the booking horizon, but NOT by filtering rows out of the bank at
generation time. Instead, experiments/hypothesis_common.py's build_config
caps each threshold at horizon_days - 1 when it actually builds a
simulation config: a threshold sampled at or beyond that point behaves as
if it were exactly horizon_days - 1, since offered delay tau never reaches
horizon_days anyway. This keeps every (threshold, horizon) combination in
the bank usable, including short horizons, rather than needing horizon-
dependent rejection sampling here. See build_config's docstring for the
capping logic itself.

horizon_days is still a per-background column here (used by H2's
horizon-based condition, and as H1's background-level default), but H1's
own experiment script additionally treats horizon_days as a tested policy
lever, sweeping it directly rather than only reading each row's assigned
value -- see experiments/h1_short_horizon_reservation.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_DIR / "outputs" / "hypotheses" / "background_scenarios.csv"

# ---------------------------------------------------------------------
# Value grids, exactly as specified
# ---------------------------------------------------------------------

RHO_VALUES = (0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 3.0)
HORIZON_VALUES = (2, 6, 10, 14, 18, 22, 26)
CLASS1_SHARE_VALUES = (0.1, 0.3, 0.5, 0.7, 0.9)
CAPACITY_VALUES = (20, 30, 40, 50)
CANCEL_VALUES = (0.1, 0.2, 0.3)
BALK_THRESHOLD_VALUES = (5, 9, 16, 22, 24)
BALK_LOW_VALUES = (0.05, 0.1, 0.2, 0.3)
BALK_HIGH_VALUES = (0.1, 0.2, 0.3, 0.4)
NOSHOW_THRESHOLD_VALUES = (4, 7, 14, 20, 22)
NOSHOW_LOW_VALUES = (0.05, 0.1, 0.2, 0.3)
NOSHOW_HIGH_VALUES = (0.1, 0.2, 0.3, 0.4)

# Order of the 17 Sobol dimensions sampled per horizon stratum (horizon
# itself is fixed per stratum, not one of these).
_DIMENSIONS: list[tuple[str, tuple]] = [
    ("rho", RHO_VALUES),
    ("class1_share", CLASS1_SHARE_VALUES),
    ("slots_per_day", CAPACITY_VALUES),
    ("cancel_1", CANCEL_VALUES),
    ("cancel_2", CANCEL_VALUES),
    ("balk_threshold_1", BALK_THRESHOLD_VALUES),
    ("balk_low_1", BALK_LOW_VALUES),
    ("balk_high_1", BALK_HIGH_VALUES),
    ("balk_threshold_2", BALK_THRESHOLD_VALUES),
    ("balk_low_2", BALK_LOW_VALUES),
    ("balk_high_2", BALK_HIGH_VALUES),
    ("noshow_threshold_1", NOSHOW_THRESHOLD_VALUES),
    ("noshow_low_1", NOSHOW_LOW_VALUES),
    ("noshow_high_1", NOSHOW_HIGH_VALUES),
    ("noshow_threshold_2", NOSHOW_THRESHOLD_VALUES),
    ("noshow_low_2", NOSHOW_LOW_VALUES),
    ("noshow_high_2", NOSHOW_HIGH_VALUES),
]

# A small set of clean, easy-to-interpret anchor corners per horizon, so the
# extremes of rho and class1_share are always represented at "textbook"
# behavioral settings, not left purely to chance.
_ANCHOR_RHOS = (min(RHO_VALUES), max(RHO_VALUES))
_ANCHOR_SHARES = (0.1, 0.5, 0.9)


def _anchor_rows(horizon: int) -> list[dict]:
    rows = []
    for rho in _ANCHOR_RHOS:
        for share in _ANCHOR_SHARES:
            rows.append(
                {
                    "horizon_days": horizon,
                    "rho": rho,
                    "class1_share": share,
                    "slots_per_day": 30,
                    "cancel_1": 0.1,
                    "cancel_2": 0.1,
                    "balk_threshold_1": 9,
                    "balk_low_1": 0.05,
                    "balk_high_1": 0.1,
                    "balk_threshold_2": 9,
                    "balk_low_2": 0.05,
                    "balk_high_2": 0.1,
                    "noshow_threshold_1": 4,
                    "noshow_low_1": 0.05,
                    "noshow_high_1": 0.1,
                    "noshow_threshold_2": 4,
                    "noshow_low_2": 0.05,
                    "noshow_high_2": 0.1,
                    "design_note": "anchor",
                }
            )
    return rows


def _is_valid(df: pd.DataFrame) -> pd.Series:
    return (
        (df["balk_low_1"] < df["balk_high_1"])
        & (df["balk_low_2"] < df["balk_high_2"])
        & (df["noshow_low_1"] < df["noshow_high_1"])
        & (df["noshow_low_2"] < df["noshow_high_2"])
        & (df["balk_threshold_1"] > df["noshow_threshold_1"])
        & (df["balk_threshold_2"] > df["noshow_threshold_2"])
    )


def _sobol_rows_for_horizon(
    horizon: int, *, n_target: int, seed: int, max_exponent: int = 21
) -> pd.DataFrame:
    """Draw and filter Sobol points for one horizon stratum.

    Doubles the draw count (as a power of two, required by
    Sobol.random_base2) until n_target valid rows are found or
    max_exponent is hit, whichever first. Filtering is fully vectorized
    with pandas/numpy, so even a million draws filters in well under a
    second.
    """
    exponent = 10  # start at 1024 draws
    while True:
        # A fresh sampler each iteration keeps the draw count an exact
        # power of two, which is what random_base2 requires for its
        # low-discrepancy guarantee.
        sampler = qmc.Sobol(d=len(_DIMENSIONS), scramble=True, seed=seed)
        points = sampler.random_base2(m=exponent)

        columns = {}
        for k, (name, values) in enumerate(_DIMENSIONS):
            n = len(values)
            idx = np.minimum((points[:, k] * n).astype(int), n - 1)
            columns[name] = np.asarray(values)[idx]
        df = pd.DataFrame(columns)
        df["horizon_days"] = horizon

        valid = df[_is_valid(df)].copy()
        if len(valid) >= n_target or exponent >= max_exponent:
            valid["design_note"] = "sobol"
            return valid.head(n_target)
        exponent += 1


def generate_background_bank(
    *, n_per_horizon: int = 120, seed: int = 20260712, horizons: tuple[int, ...] = HORIZON_VALUES
) -> pd.DataFrame:
    strata = []
    for i, horizon in enumerate(horizons):
        anchors = pd.DataFrame(_anchor_rows(horizon))
        anchors = anchors[_is_valid(anchors)]
        remaining = max(0, n_per_horizon - len(anchors))
        sobol_rows = _sobol_rows_for_horizon(
            horizon, n_target=remaining, seed=seed + i * 97
        )
        stratum = pd.concat([anchors, sobol_rows], ignore_index=True)
        strata.append(stratum)

    bank = pd.concat(strata, ignore_index=True)
    bank["lambda_1"] = bank["rho"] * bank["slots_per_day"] * bank["class1_share"]
    bank["lambda_2"] = bank["rho"] * bank["slots_per_day"] * (1 - bank["class1_share"])
    bank.insert(0, "background_id", [f"BG{i:05d}" for i in range(1, len(bank) + 1)])

    column_order = [
        "background_id",
        "design_note",
        "horizon_days",
        "rho",
        "class1_share",
        "slots_per_day",
        "lambda_1",
        "lambda_2",
        "cancel_1",
        "cancel_2",
        "balk_threshold_1",
        "balk_low_1",
        "balk_high_1",
        "balk_threshold_2",
        "balk_low_2",
        "balk_high_2",
        "noshow_threshold_1",
        "noshow_low_1",
        "noshow_high_1",
        "noshow_threshold_2",
        "noshow_low_2",
        "noshow_high_2",
    ]
    return bank[column_order]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-horizon", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bank = generate_background_bank(n_per_horizon=args.n_per_horizon, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bank.to_csv(args.output, index=False)
    print(f"Background bank: {len(bank)} scenarios -> {args.output}")
    print(bank.groupby("horizon_days").size().rename("count"))
    condition_h1 = (bank["noshow_threshold_1"] < bank["noshow_threshold_2"]).mean()
    condition_h2 = ((bank["horizon_days"] >= 21) & (bank["rho"] >= 2.0)).mean()
    print(f"Share satisfying H1's threshold_1 < threshold_2: {condition_h1:.2%}")
    print(f"Share satisfying H2's horizon>=21 and rho>=2: {condition_h2:.2%}")


if __name__ == "__main__":
    main()
