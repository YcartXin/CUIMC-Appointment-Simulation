# Completed Experiment 2: results and figure interpretation

The downloaded `outputs/experiment2_completed_results.tar.gz` was extracted into `outputs/experiment2_ratio_horizon/`. The original archive and previous datasets remain unchanged. No simulations were rerun locally and no PowerPoint file was edited.

## Coverage and checks

Both experiments use the same 1,134-scenario bank: 3 Class 1 no-show post probabilities × 3 balking post probabilities × 6 rho levels × 7 Class 1 arrival shares × 3 system sizes. Shares are 10%, 30%, 50%, 60%, 70%, 80%, 90%; sizes are 30, 40, 50 slots/day. Rho values are 1.2, 1.4, 1.7, 2, 2.5, 3.

Discovery has 10,784,340/10,784,340 planned runs available. Trade-off evaluation has 47,140/47,140, and Pareto evaluation has 69,400/69,400. All inventories report zero pending. Frozen hashes, bank agreement, frozen policy settings, distinct independent evaluation seeds and their recomputed mean deltas were checked. The evaluated Pareto catalog contains 3,115 candidates.

## What is selected

Served-rate gains and losses are **absolute percentage-point differences from the matched FCFS baseline (100-day booking horizon, no reservations) in the same scenario**, with paired simulation seeds. A 5pp gain is not a 5% relative increase.

**Trade-off:** Discovery selects the policy with the smallest signed Class 2 loss / Class 1 gain ratio, subject to at least 5pp Class 1 gain and the tested Class 2 loss ceiling. Class 2 loss is baseline minus policy served rate: no loss gives ratio zero; Class 2 improvement gives a negative ratio. Both-flexible may choose Q=0 (horizon only). Each budget has its own discovery-frozen choice; independent evaluation does not change that choice.

The principal comparisons use the **fixed 10pp-budget choice** per scenario. There are 659 discovery-feasible both-flexible choices. Of these, 581 independently meet the Class 1 target and Class 2 ceiling (51.2% of all 1,134 scenarios). Three of those successful choices have Q=0. The remaining scenarios stay failures in success-rate denominators; their policies are not replaced by another evaluated budget choice.

**Pareto:** Candidates are screened in discovery and evaluated independently. Validated meaningful Pareto cells do not harm either class on the evaluated mean and improve at least one class by 0.5pp. There are 117 successful both-flexible scenarios (10.3% of 1,134). For descriptive gain/settings/utilization figures, retain the validated cell with the largest evaluated Class 1 gain in each successful scenario, as previously requested. Thirty-six selected cells have Q=0. This retrospective selection remains descriptive and can introduce selection optimism; it differs from the discovery-frozen trade-off choice. Reservation-only has zero validated Pareto scenarios and is not graphed.

The Pareto and trade-off gain distributions have **different qualifying scenario cohorts and thresholds**, despite sharing the full scenario bank. Their median difference is not a paired causal effect of relaxing Class 2 nonharm alone.

## What the figures summarize

- Main horizon, reservation, and gain comparisons: 117 Pareto scenarios versus 581 successful fixed-10pp trade-off scenarios; one policy each. Blue is Pareto, orange is Class 2 loss allowed. Median/min/max lines describe variation across scenarios, not estimation uncertainty. Missing Pareto rho groups have no qualifying scenarios and are left blank.
- Median booking horizons by low/medium/high Class 1 no-show post probability are Pareto 2/3/3 days and trade-off 7/4/4 days.
- Slide 25 keeps its rho <= 2.5 scope, now including the completed share and size additions: 945 scenarios, 315 per no-show level. Pareto successes are 3, 44, 68 (about 1.0%, 14.0%, 21.6%). All points use the combined complete dataset, with no separate follow-up series.
- The two system-size figures use identical rho 1.7 and 2.0 scope. Each share/size point has 18 scenarios. Colors are blue for 30, green for 40, orange for 50 slots/day. These are complete-design rates, not variability summaries needing min/max bars.
- Pareto utilization and rho success summaries use the full completed rho range.
- The success-by-tested-budget curve uses each budget's frozen choice and all 1,134 scenarios as denominator. Counts from 0 through 10pp are 0, 13, 47, 165, 306, 353, 384, 489, 581 at budgets 0, 0.5, 1, 2, 3, 4, 5, 7.5, 10pp respectively.

## Minimum tested budget versus actual loss

For each scenario, the **minimum tested budget** is the smallest tested ceiling whose frozen choice passes independent evaluation (Class 1 gain >=5pp and Class 2 loss within that ceiling). Then the plotted point is the **median of these scenario-specific minimum budgets** within the share/rho group. Faint bars show their smallest and largest values. This does not establish the mathematical minimum over every possible policy or an untested budget.

The **actual evaluated loss at that budget** is the measured Class 2 served-rate difference of that particular frozen policy, which can be smaller than its budget ceiling. For example, a policy first passing at a 3pp tested ceiling might have an actual evaluated loss of 2.4pp. The budget is the constraint; the loss is the measured outcome.

There are 591 scenarios with a successful frozen choice at at least one budget. Budget and actual-loss distribution panels use these 591 scenarios once each at their first successful tested budget. This count can exceed the fixed-10pp success count because different budgets select different policies and independent evaluation need not remain successful when the selected policy changes.

Slide 32 also uses one observation per scenario among those 591, but reports the **smallest observed actual loss across its successful frozen tested choices**, not necessarily its first successful budget choice. Class 2 gains are included in the zero-loss bin. This is a descriptive diagnostic; it does not retrospectively change the principal 10pp selection.

## Reproduce

From the repository root, run `python scripts/plot_completed_experiment2.py`. This regenerates PNGs, audit CSVs and `validation_summary.json`. The generator uses only the completed replacement dataset; it does not concatenate old output tables or count multiple cells within a scenario in these summaries. Existing figure versions are backed up before replacement.
