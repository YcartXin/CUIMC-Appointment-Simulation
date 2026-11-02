# Slides 25-30: one best Pareto case per background

Recreate from the repository root:

```bash
python scripts/plot_pareto_background_best.py
```

The figures are in `figures/pareto_background_best/`, as PNG and SVG pairs.
The audit table and plotted summaries are in `outputs/pareto_background_best/`.
Original images and the PowerPoint are preserved. These are replacement figures
for slides 25, 26, 27, 29 and 30. Slide 28 contains only text.

## Selection and interpretation

For each background and each policy separately:

1. Retain cases whose `validated_meaningful_pareto` flag is true.
2. Choose the case with the largest evaluated mean Class 1 served-rate gain.
3. Break exact ties by larger Class 2 gain, larger utilization gain, then smaller
   horizon, reserved slot count, and reservation window, in that order.
4. Use that same case for utilization, horizon, and reservation percentage.
   Do not independently maximize these other outcomes or average the remaining cases.

This selects 76 background-policy rows (43 horizon-only and 33 both-flexible) from
212 qualifying cases (73 horizon-only and 139 both-flexible). Reservation-only
has no qualifying backgrounds. Counts are background-policy pairs, so a
background that qualifies for both policies has one observation in each policy.
The selected-case CSV preserves identifiers, settings, and evaluated metrics.

This is a retrospective descriptive choice based on evaluated outcomes, as
requested. Selection changes the interpretation of the original independent
evaluation: do not present the best evaluated result as a newly independently
tested optimizer or attach the selected case's intervals as selection-adjusted
uncertainty. Qualification follows the existing mean-based Pareto definition,
not confidence-interval-supported nonharm.

## Slide mapping and findings

- Slide 25: no-show background rates. Original experiment, rho <=2.5. Rates
  remain unchanged because the old figure already counted each background once.
- Slide 26: both-flexible background rates by share and scale, rho=1.7/2.0.
  Rates remain unchanged. The 80% follow-up has success status only, reconstructed
  from the previously reported breakdown. It cannot supply a winning case.
- Slide 27: utilization at the selected case, then the median across backgrounds.
  Original experiment, rho <=2.5. Horizon-only: 1.1242pp across 43 backgrounds.
  Both-flexible: 1.4027pp across 33 backgrounds. All selected cases enter this
  median, including the one with a slightly negative utilization change.
- Slide 29: median selected horizon across qualifying backgrounds, all original
  demand/share/scale levels. Horizon-only: no low-severity successes, 3 days at
  medium, 4 days at high. Both-flexible: 2, 2, and 3 days at low/medium/high.
- Slide 30: median selected reserved slots / daily capacity x 100, across
  qualifying both-flexible backgrounds. Low/medium/high: 30%, 30%, 35%.
  The old slide statement that the median stays constant should be revised.

Suggested accompanying text for slide 27:
"At each background's best Class 1 Pareto case, median utilization increases
by 1.12pp for horizon-only and 1.40pp for both-flexible."

Suggested replacement summary for slide 28:
"Reservation-only has no qualifying Pareto backgrounds. Higher Class 1 no-show
severity is associated with more qualifying backgrounds. The favorable system
scale shifts toward smaller systems as the Class 1 share rises. At each
background's best Class 1 Pareto case, both-flexible has a 1.40pp median
utilization increase and horizon-only has a 1.12pp median increase."

Suggested accompanying text for slide 29:
"Selected horizons remain short. At high no-show severity, the median is 4 days
for horizon-only and 3 days for both-flexible."

Suggested accompanying text for slide 30:
"The median reservation percentage is 30% at low and medium no-show severity
and 35% at high severity."

## Inputs and missing data

The script uses the original `validated_meaningful_pareto_cells.csv`,
`patient_behavior_factorial_bank.csv`, and
`validated_background_policy_universe.csv` in `outputs/pareto_presentation/`,
plus `outputs/pareto_combined/combined_background_policy_universe.csv` for
slide 26. It does not substitute Experiment 2 frontier data for the original
Pareto experiment. The follow-up is excluded from utilization and setting
summaries, consistent with the original slides' coverage.

To add the 80% follow-up to those summaries, obtain its
`independent_evaluation/validated_meaningful_pareto_cells.csv` and the matching
`validated_background_policy_universe.csv` from the Grid follow-up output
under `served_rate_pareto/pareto_catalog_5seed`. They must retain background IDs,
policy settings H/Q/W, evaluated Class 1/Class 2 deltas, utilization delta, and
qualification status. The local combined README describes the missing data.

## Figure 5 of the frontier analysis

This is a different experiment and a different selection rule: the first tested
Class 2 loss budget that reaches at least 5pp Class 1 gain within the 10pp ceiling.
The left panel is a cumulative distribution of that minimum tested budget.
For example, a value of 60% at 4pp means 60% of that policy's successful
backgrounds reach the target with a tested budget of 4pp or less.
The right panel shows actual evaluated Class 2 loss at those first qualifying
budgets. A budget is an allowed ceiling, so actual loss can be lower.
The median tested budget is 3pp and median actual loss is about 2.90pp for both
policies. The curves use different successful samples: reservation-only n=485,
both-flexible n=519. They exclude failed backgrounds, so curve height is not
the policy's overall success rate. Overall success is 485/972 and 519/972,
respectively. A budget of 3pp does not guarantee a 3pp loss in every background.

## Subtle range-bar revision

For the current slide 27, 29 and 30 figures run:

```bash
python scripts/plot_pareto_ranges.py
```

This command writes both-flexible-only median figures to the existing
`figures/pareto_background_best/` paths and range summaries to
`outputs/pareto_background_best/`. Each background contributes its single
largest-Class-1-gain qualifying case using the documented tie-breaking rule.
Bars show the observed min and max across these selected cases, not confidence
intervals. Dashed bars use 30% opacity so the median markers dominate.
A missing visible bar can mean min=median=max, as for low-severity horizons.
No explanatory footnotes are included in the figures.

Slide 27 retains rho<=2.5; slides 29/30 retain all original demand levels.
The older 80% follow-up remains excluded because case-level settings and
utilization are unavailable. No frontier results are substituted for it.
The earlier `plot_pareto_background_best.py` is the historical full-figure
version and includes horizon-only; use `plot_pareto_ranges.py` for these
current replacements and `plot_slides25_26_revision.py` for slides 25/26.

The frontier budget figure now uses similarly faint dashed range bars:
`python scripts/plot_experiment2_frontier.py --figure budget`.
Success-rate/count graphs and empirical cumulative distributions do not need
min-max bars; they already show proportions/counts or the full distribution.
