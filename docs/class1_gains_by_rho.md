# Class 1 gains versus baseline, separated by rho

Regenerate with `python scripts/plot_class1_gains_by_rho.py`.
Figure PNG/SVG: `figures/Trade-off/policy_settings_comparison/class1_served_rate_gain_by_rho`.
Data: `outputs/policy_settings_comparison/class1_gains_by_rho_summary.csv` and
`class1_gains_selected_backgrounds.csv`.

Class 1 served rate is served / arrivals (simulation/model.py). Gain in pp is
100 * (policy served rate - matched no-policy baseline served rate), not a
relative percent change. The background and simulation seeds are matched when
evaluating deltas. A baseline of 40% and policy outcome of 48% gives +8pp.
The frontier baseline is H=100, Q=0, W=-1 (no reservation), as defined in
analysis/experiment2_frontier.py. The legacy postprocessor reads the matched
baseline cell from each background's search records.

Select the largest evaluated mean Class 1 gain per background among qualifying
both-flexible settings. Horizon-only is excluded. The same selection rules and
blue/orange mapping as the policy-setting comparison are used. Faint dashed
bars show observed min-max ranges across selected background outcomes, markers
show medians. All successful backgrounds have equal weight within each rho.
No additional cases, tested budgets, or seeds add observations.

Blue Pareto: original experiment, no mean served-rate harm to either class and
at least 0.5pp gain to one class, then maximize Class 1 gain. Counts by rho are
9 at 1.2, 12 at 1.7, 12 at 2.0. There are no qualifying blue observations at
1.4, 2.5 or 3.0, so those points are absent, not zero. The older 80% follow-up
has no local Class 1 gain records, so it cannot enter the blue series.

Orange: frontier experiment, validated Class 1 gain >=5pp within a tested
Class 2 loss budget <=10pp, then choose maximum evaluated Class 1 gain.
Counts at rho 1.2/1.4/1.7/2.0/2.5/3.0 are 81/81/81/98/91/87.
This is the best gain within the ceiling, not the gain at the minimum budget
that first reaches 5pp. Exact tie-breaking follows the setting comparison.

These series use different successful samples, thresholds, search procedures,
and coverage of patient shares/system sizes. Grouping by rho alone does not
make the other factors matched. The orange curve is screened for >=5pp while
the blue curve is not. Therefore differences must not be presented as the
isolated effect of allowing Class 2 harm or as a paired treatment effect.
The best evaluated case is a retrospective descriptive selection, not an
independently tested optimizer. Range bars are not confidence intervals.
