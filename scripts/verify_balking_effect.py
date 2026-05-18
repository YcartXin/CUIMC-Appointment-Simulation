"""
Diagnostic: verify whether suppressing balking changes utilization and access.

Runs baseline and balking variants from scratch using a fixed seed list.
Never reads pre-existing simulation CSVs for the comparison.

Outputs
-------
results/balking_effect_verification.csv  -- scenarios A-D, per-seed rows
results/balking_effect_sweep.csv         -- sweep over b_high, aggregated
figures/balking_effect_verification.png  -- sweep plot

Run from repo root:
    python scripts/verify_balking_effect.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.plot_style import (
    ARRIVAL_COLOR,
    BALKING_COLOR,
    BASELINE_COLOR,
    NO_SHOW_COLOR,
    UTILIZATION_COLOR,
)

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from simulation.config_loader import load_config
from simulation.engine import ClinicAppointmentSimulation
from simulation.model import ThresholdRule
from analysis.metrics import outcome_rates_from_result, result_metrics_from_result

# ── output dirs ──────────────────────────────────────────────────────────────
RESULTS_DIR = REPO_DIR / "results"
FIGURES_DIR = REPO_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── fixed parameters ─────────────────────────────────────────────────────────
BASE_CONFIG   = load_config(REPO_DIR / "configs" / "baseline.yaml")
SEEDS         = list(range(1001, 1031))          # 30 independent seeds
SWEEP_B_HIGH  = [round(v, 2) for v in np.linspace(0.0, 1.0, 11)]

# Baseline balking rule from config (applied symmetrically to both classes)
_base_rule    = BASE_CONFIG.classes[1].balk_prob
BASELINE_B_LOW  = _base_rule.low    # 0.00
BASELINE_B_HIGH = _base_rule.high   # 0.50
BASELINE_THRESHOLD = _base_rule.threshold  # 9

# ── helpers ───────────────────────────────────────────────────────────────────

def set_balk(config, b_low: float, b_high: float):
    """Return config with both classes set to the given balking probabilities.

    Preserves balking threshold and all other parameters unchanged.
    """
    classes = {}
    for cid, params in config.classes.items():
        rule = ThresholdRule(
            threshold=params.balk_prob.threshold,
            low=float(b_low),
            high=float(b_high),
        )
        classes[cid] = replace(params, balk_prob=rule)
    return replace(config, classes=classes)


def run_one(config, seed: int) -> dict:
    seeded = replace(config, seed=seed)
    result = ClinicAppointmentSimulation(seeded).run()
    metrics  = result_metrics_from_result(result)
    outcomes = outcome_rates_from_result(result)
    # booked_rate = fraction of arrivals that accepted an offer and were booked
    arrivals = outcomes["total_arrivals"]
    booked   = outcomes["total_booked"]
    booked_rate = booked / arrivals if arrivals > 0 else float("nan")
    decomp_sum = (
        outcomes["served_rate"]
        + outcomes["balked_rate"]
        + outcomes["no_offer_rate"]
        + outcomes["canceled_rate"]
        + outcomes["no_show_rate"]
        + outcomes["unresolved_booked_rate"]
    )
    return {
        "seed": seed,
        "average_utilization":         metrics["average_utilization"],
        "overall_percent_serviced":    metrics["overall_percent_serviced"],
        "mean_offered_booking_delay":  metrics["mean_offered_booking_delay"],
        "mean_accepted_booking_delay": metrics["mean_accepted_booking_delay"],
        "served_rate":                 outcomes["served_rate"],
        "booked_rate":                 booked_rate,
        "balked_rate":                 outcomes["balked_rate"],
        "no_offer_rate":               outcomes["no_offer_rate"],
        "canceled_rate":               outcomes["canceled_rate"],
        "no_show_rate":                outcomes["no_show_rate"],
        "unresolved_booked_rate":      outcomes["unresolved_booked_rate"],
        "decomposition_sum":           decomp_sum,
    }


def run_scenario(config, seeds: list[int]) -> pd.DataFrame:
    return pd.DataFrame([run_one(config, s) for s in seeds])


METRIC_COLS = [
    "average_utilization",
    "overall_percent_serviced",
    "mean_offered_booking_delay",
    "mean_accepted_booking_delay",
    "served_rate",
    "booked_rate",
    "balked_rate",
    "no_offer_rate",
    "canceled_rate",
    "no_show_rate",
    "unresolved_booked_rate",
    "decomposition_sum",
]


def aggregate(df: pd.DataFrame, label: str, b_high: float) -> dict:
    row = {"scenario": label, "b_high": b_high, "n_seeds": len(df)}
    for col in METRIC_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        row[f"{col}_mean"] = vals.mean()
        row[f"{col}_std"]  = vals.std(ddof=1)
        row[f"{col}_se"]   = vals.std(ddof=1) / np.sqrt(len(vals))
    return row


def diff_vs_baseline(agg_row: dict, baseline_row: dict, cols=METRIC_COLS) -> dict:
    """Append difference and 95% CI columns relative to baseline."""
    n = agg_row["n_seeds"]
    out = dict(agg_row)
    for col in cols:
        mk = f"{col}_mean"
        sk = f"{col}_se"
        if mk not in agg_row or mk not in baseline_row:
            continue
        d = agg_row[mk] - baseline_row[mk]
        # SE of difference assuming independence
        se_d = np.sqrt(agg_row.get(sk, 0) ** 2 + baseline_row.get(sk, 0) ** 2)
        out[f"{col}_diff"]   = d
        out[f"{col}_ci_low"] = d - 1.96 * se_d
        out[f"{col}_ci_high"]= d + 1.96 * se_d
    return out


# ── scenario definitions ─────────────────────────────────────────────────────

SCENARIOS = [
    ("A_baseline",      BASELINE_B_LOW, BASELINE_B_HIGH),
    ("B_no_balking",    0.00, 0.00),
    ("C_half_balking",  0.00, BASELINE_B_HIGH / 2),
    ("D_full_balking",  0.00, 1.00),
]

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Baseline config: {REPO_DIR / 'configs' / 'baseline.yaml'}")
    print(f"Seeds: {SEEDS[0]}..{SEEDS[-1]}  (n={len(SEEDS)})")
    print(f"Balking threshold: {BASELINE_THRESHOLD} days  "
          f"(b_low={BASELINE_B_LOW}, b_high={BASELINE_B_HIGH})")
    print()

    # ── run named scenarios ─────────────────────────────────────────────────
    per_seed_rows = []
    agg_rows = []

    for label, b_low, b_high in SCENARIOS:
        print(f"Running {label}  (b_low={b_low}, b_high={b_high}) × {len(SEEDS)} seeds …")
        config = set_balk(BASE_CONFIG, b_low, b_high)
        df = run_scenario(config, SEEDS)
        df["scenario"] = label
        df["b_high"]   = b_high
        per_seed_rows.append(df)
        agg_rows.append(aggregate(df, label, b_high))

    per_seed_df = pd.concat(per_seed_rows, ignore_index=True)
    per_seed_df.to_csv(RESULTS_DIR / "balking_effect_verification.csv", index=False)

    baseline_agg = agg_rows[0]
    summary_rows = [diff_vs_baseline(r, baseline_agg) for r in agg_rows]
    summary_df = pd.DataFrame(summary_rows)

    # ── sweep over b_high ───────────────────────────────────────────────────
    print("\nRunning sweep over b_high …")
    sweep_rows = []
    for b_high in SWEEP_B_HIGH:
        config = set_balk(BASE_CONFIG, 0.00, b_high)
        df = run_scenario(config, SEEDS)
        sweep_rows.append(aggregate(df, f"sweep_{b_high:.2f}", b_high))

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(RESULTS_DIR / "balking_effect_sweep.csv", index=False)

    # ── plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fig.suptitle("Effect of balking probability on simulation outcomes\n"
                 "(b_low = 0, all other parameters at baseline)", fontsize=12)

    SERIES = [
        ("average_utilization",      "avg utilization",     axes[0, 0], UTILIZATION_COLOR),
        ("overall_percent_serviced", "overall served rate", axes[0, 1], ARRIVAL_COLOR),
        ("no_show_rate",             "no-show share",       axes[1, 0], NO_SHOW_COLOR),
        ("balked_rate",              "balked share",        axes[1, 1], BALKING_COLOR),
    ]

    x = sweep_df["b_high"]
    for col, ylabel, ax, color in SERIES:
        y    = sweep_df[f"{col}_mean"]
        yerr = 1.96 * sweep_df[f"{col}_se"]
        ax.plot(x, y, marker="o", markersize=4, linewidth=1.6, color=color)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color)
        ax.axvline(BASELINE_B_HIGH, color=BASELINE_COLOR, linewidth=0.9,
                   linestyle="--", label=f"baseline b_high={BASELINE_B_HIGH}")
        ax.set_xlabel("b_high (high-delay balking probability)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.capitalize())
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(FIGURES_DIR / "balking_effect_verification.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    # ── console summary ─────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("FACTUAL SUMMARY")
    print("="*72)

    base = baseline_agg
    nobk = agg_rows[1]   # B_no_balking

    def fmt(label, col):
        bm = base[f"{col}_mean"];  bse = base[f"{col}_se"]
        nm = nobk[f"{col}_mean"]; nse = nobk[f"{col}_se"]
        d  = nm - bm
        se_d = np.sqrt(bse**2 + nse**2)
        lo, hi = d - 1.96*se_d, d + 1.96*se_d
        print(f"  {label}:")
        print(f"    Baseline    = {bm:.4f} ± {bse:.4f} (SE)")
        print(f"    No-balking  = {nm:.4f} ± {nse:.4f} (SE)")
        print(f"    Difference  = {d:+.4f},  95% CI [{lo:+.4f}, {hi:+.4f}]")

    fmt("Average utilization",       "average_utilization")
    fmt("Overall served rate",       "overall_percent_serviced")
    fmt("Booked rate",               "booked_rate")
    fmt("No-show share",             "no_show_rate")
    fmt("Balked share",              "balked_rate")
    fmt("Cancelled share",           "canceled_rate")
    fmt("No-offer share",            "no_offer_rate")
    fmt("Mean offered wait (days)",  "mean_offered_booking_delay")
    fmt("Mean accepted wait (days)", "mean_accepted_booking_delay")

    # Decomposition check
    print(f"\n  Decomposition sum (should ≈ 1.0):")
    print(f"    Baseline    = {base['decomposition_sum_mean']:.6f}")
    print(f"    No-balking  = {nobk['decomposition_sum_mean']:.6f}")

    print("\n" + "-"*72)
    util_up = nobk["average_utilization_mean"] > base["average_utilization_mean"]
    svc_up  = nobk["overall_percent_serviced_mean"] > base["overall_percent_serviced_mean"]
    print(f"Q1. Does no-balking INCREASE avg utilization?  "
          f"{'YES' if util_up else 'NO'} "
          f"({base['average_utilization_mean']:.4f} → {nobk['average_utilization_mean']:.4f})")
    print(f"Q2. Does no-balking INCREASE overall served rate? "
          f"{'YES' if svc_up else 'NO'} "
          f"({base['overall_percent_serviced_mean']:.4f} → {nobk['overall_percent_serviced_mean']:.4f})")

    ns_up   = nobk["no_show_rate_mean"] > base["no_show_rate_mean"]
    wait_dn = nobk["mean_accepted_booking_delay_mean"] > base["mean_accepted_booking_delay_mean"]
    print(f"Q3. Does no-balking INCREASE no-show share?    "
          f"{'YES' if ns_up else 'NO'} "
          f"({base['no_show_rate_mean']:.4f} → {nobk['no_show_rate_mean']:.4f})")
    print(f"Q4. Does no-balking INCREASE mean accepted wait? "
          f"{'YES' if wait_dn else 'NO'} "
          f"({base['mean_accepted_booking_delay_mean']:.4f} → "
          f"{nobk['mean_accepted_booking_delay_mean']:.4f})")

    bk_delta  = nobk["booked_rate_mean"] - base["booked_rate_mean"]
    ns_delta  = nobk["no_show_rate_mean"] - base["no_show_rate_mean"]
    print(f"Q5. Booking-conversion Δ = {bk_delta:+.4f};  "
          f"no-show Δ = {ns_delta:+.4f}")
    if bk_delta > 0 and ns_delta > 0:
        print("    → More bookings but more no-shows; "
              "net utilization effect depends on magnitudes.")
    elif bk_delta > 0 and ns_delta <= 0:
        print("    → More bookings with no added no-show exposure: "
              "conversion effect dominates.")
    else:
        print("    → Booking-conversion did not increase; "
              "check no-offer dynamics.")

    print("\n" + "-"*72)
    print("CANCELLATION MECHANICS (engine.py:232-253)")
    print("  apply_start_of_day_cancellations() runs every simulated day.")
    print("  For each future slot r in range(1, horizon_days):")
    print("    each booking draws Bernoulli(cancel_prob) independently.")
    print("  → Daily hazard model: P(survive k days) = (1 - cancel_prob)^k")
    print(f"  Baseline cancel_prob = {BASE_CONFIG.classes[1].cancel_prob}")
    import math
    avg_tau = base["mean_accepted_booking_delay_mean"]
    surv = (1 - BASE_CONFIG.classes[1].cancel_prob) ** avg_tau
    print(f"  P(not cancelled by service | mean τ={avg_tau:.2f} d) ≈ {surv:.3f}")

    print("\nNO-SHOW MECHANICS (engine.py:202-206)")
    print("  serve_today() evaluates no-show at service time.")
    print("  tau = booking.booking_delay  # the ORIGINAL offered delay at booking")
    print("  if rng.random() < no_show_prob(tau): → no-show")
    print("  → No-show probability depends on original offered delay, not residual wait.")
    print(f"  Baseline no-show threshold = {BASE_CONFIG.classes[1].no_show_prob.threshold} days")
    print(f"  Baseline no-show high prob = {BASE_CONFIG.classes[1].no_show_prob.high}")

    print("\n" + "="*72)
    print(f"Outputs saved to:")
    print(f"  {RESULTS_DIR / 'balking_effect_verification.csv'}")
    print(f"  {RESULTS_DIR / 'balking_effect_sweep.csv'}")
    print(f"  {FIGURES_DIR / 'balking_effect_verification.png'}")


if __name__ == "__main__":
    main()
