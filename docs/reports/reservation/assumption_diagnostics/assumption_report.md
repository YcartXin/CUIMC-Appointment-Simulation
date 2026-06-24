# Strict Reservation Assumption Diagnostics

> **Conclusion.** The strict-reservation simulation passed all hard implementation checks across 28,800 tasks. Most business hypotheses were supported. The main break is conceptual: additive and pooled objectives do not always rank reservation quantities in the same order. Offered-wait improvements also need access checks, because lower offered wait can coincide with higher no-offer rates.

## 1. Purpose

This report is diagnostic. It does not rank reservation quantities, recommend a policy, or assume monotonicity in `Q`. Its goal is to check whether strict Class 1 reservation behaves consistently with the business hypotheses used for later policy comparison.

## 2. Experiment Grid

- 15 behavior regimes: 5 threshold pairs times 3 post-threshold balking rates.
- `C1` and `C2` are class-specific balking thresholds in days; `high` is the post-threshold balking probability.
- `Q = [0,1,4,8,16,24,31,32]` reserved slots per day.
- Total daily demand `[24,32,50,100]` and Class 1 shares `[0.25,0.50,0.75]`.
- 20 seeds per cell, `S=32`, horizon 14, burn-in 30, measurement 365, cooldown 14.
- Profile `standard`; sweep complete: `True`.

## 3. Hard-Check Diagnostics

All hard checks passed: 0 checks had failures and the failed task-count sum was 0. These checks cover configuration validity, accounting, capacity bounds, reservation ownership, deterministic replay, `Q=0` FCFS equivalence, and `Q=32` Class 2 exclusion.

![Violation counts](figures/violation_counts.png)

## 4. Business Hypotheses Tested

Verdicts use paired seed-level bootstrap 95% confidence intervals. `supported` means the full interval satisfies the stated tolerance; `contradicted` means the full interval crosses the material boundary in the opposite direction; otherwise the result is `inconclusive`.

| hypothesis | supported | inconclusive | contradicted |
|---|---|---|---|
| Additive and pooled ordering match | 30 | 1 | 29 |
| Balk 0.3 vs 0.7 little effect | 31 | 1 | 0 |
| C1 not worse than FCFS | 1256 | 4 | 0 |
| C2 not better than FCFS | 1254 | 6 | 0 |
| Symmetric FCFS class rates match | 24 | 0 | 0 |

![Business hypothesis status](figures/business_hypothesis_status.png)

## 5. Main Findings

- Hard implementation checks passed for 28,800 evaluated tasks.
- Business-hypothesis cells: 2,595 supported, 12 inconclusive, and 29 contradicted.
- The Class 1 protection hypothesis was not contradicted: Class 1 service did not materially fall versus matched FCFS in the tested cells.
- The Class 2 non-improvement hypothesis was not contradicted: strict reservation did not materially improve Class 2 service versus FCFS.
- The symmetric-FCFS check passed, so class labels behave as expected when the inputs are symmetric.
- Additive and pooled objectives disagreed in some equal-demand settings; this means objective choice matters and should be named explicitly in selection work.

## 6. Contradicted Or Inconclusive Hypotheses

| hypothesis | status | cells | demand_values | q_values | regimes | minimum_effect | maximum_effect |
|---|---|---|---|---|---|---|---|
| Additive and pooled ordering match | contradicted | 29 | 50, 100 | n/a | 15 | 0.0125 | 0.2304 |
| Additive and pooled ordering match | inconclusive | 1 | 100 | n/a | 1 | 0.0089 | 0.0089 |
| Balk 0.3 vs 0.7 little effect | inconclusive | 1 | 100 | 31 | 0 | 0.003 | 0.003 |
| C1 not worse than FCFS | inconclusive | 4 | 50 | 1 | 4 | -0.0032 | 0.0013 |
| C2 not better than FCFS | inconclusive | 6 | 50, 100 | 1, 4, 8 | 5 | -0.0006 | 0.0032 |

The only contradicted hypothesis is that additive and pooled objectives always order `Q` values consistently. Equal demand here means equal configured arrival rates; realized Poisson arrivals can still differ between classes within a seed, so the additive and pooled formulas are not guaranteed to be proportional. In the figure, `C` marks contradicted cells and `I` marks inconclusive cells.

![Objective ordering disagreement](figures/objective_ordering_disagreement.png)

## 7. Composition Effects: Offered Wait Versus No-Offer Rate

302 of 1,260 checked cells show a confidently lower offered wait together with a confidently higher no-offer rate. This is a composition warning, not a policy benefit. 255 of those flags occur at `Q=31` or `Q=32`, where protected capacity can remove patients from the offered population.

| total_demand | flagged_cells |
|---|---|
| 24 | 30 |
| 32 | 45 |
| 50 | 92 |
| 100 | 135 |

![Composition flags](figures/composition_flags.png)

## 8. Limitations

- The checks test model behavior under the chosen sweep; they do not prove the policy is operationally acceptable.
- Same seeds provide matched labels, but policy-dependent RNG use means they are not exact common-random-number experiments.
- The 0.005 tolerance is a practical threshold, not a clinical or operational access requirement.
- Offered waiting time is conditional on receiving an offer, so it must be read with no-offer and service-rate diagnostics.
- Objective-ordering disagreement is expected when realized arrivals differ; the result is a warning against treating additive and pooled objectives as interchangeable.

## 9. Next Step: Use These Checks In Policy Comparison

When comparing strict reservation with the booking-window policy, carry forward the same hard checks, the same business-hypothesis status summary, and the same composition diagnostic. This keeps objective improvements separate from access artifacts.

## Files

- Hard-check summary: `tables/violation_summary.csv`
- Business assumptions: `tables/business_assumptions.csv`
- Composition flags: `tables/composition_flags.csv`
- Cell summary: `tables/cell_summary.csv`
- Compact hypothesis status: `tables/hypothesis_status_summary.csv`
- Non-supported hypotheses: `tables/non_supported_hypotheses.csv`
- Composition summary: `tables/composition_summary.csv`
