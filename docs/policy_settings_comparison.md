# Policy settings comparison for slides 33 and 34

Run `python scripts/plot_policy_settings_comparison.py` from the repository root.
PNG/SVG figures are in `figures/policy_settings_comparison/`; plotted summaries
and selected background records are in `outputs/policy_settings_comparison/`.
The PowerPoint is not edited. Blue always denotes Pareto, orange always denotes
Class 2 loss allowed. Markers show medians and faint dashed bars show the full
observed minimum-to-maximum range. Figures contain no explanatory footnotes.

Both series use both-flexible only, one selected case per qualifying background.
Horizon-only is omitted in accordance with the user's preference.

## Selection

Pareto: original independently evaluated mean-based Pareto qualifier, then the
case with the largest Class 1 gain per background. Existing tie rules from
`plot_pareto_background_best.select_best` are retained. There are 33 qualifying
backgrounds, with low/medium/high no-show counts 3/12/18. The old 80% follow-up
lacks case-level settings, so it cannot enter this series.

Class 2 loss allowed: use `evaluated_selected_cells.csv` from the newer frontier
experiment. Retain validated both-flexible cells at tested budgets <=0.10 with
Class 1 gain >=0.05. Select the largest evaluated Class 1 gain per background,
then break exact ties by smaller budget, larger Class 2 delta, smaller horizon,
smaller Q, and smaller window, in that order. There are 519 qualifying backgrounds
(low/medium/high: 160/168/191). This includes 80% share and 40-slot system sizes.
Every selected case actually has negative Class 2 served-rate change, verified
by the script. Repeated settings and budgets never add observations.

This differs from the minimum-required-budget figures: here the chosen case
maximizes Class 1 gain within the 10pp ceiling, rather than choosing the first
budget to reach 5pp. The same selected case supplies both horizon and reservation
percentage. Reservation percentage is Q / slots_per_day * 100, applying to the
selected reservation window. 100% does not mean every appointment throughout
the entire booking horizon is reserved.

## Median results (low / medium / high no-show severity)

- Pareto horizon: 2 / 2 / 3 days.
- Loss-allowing horizon: 26.5 / 24 / 11 days.
- Pareto reservation percentage: 30% / 30% / 35%.
- Loss-allowing reservation percentage: 100% / 100% / 87.5%.

A 26.5-day median can occur with an even number of backgrounds even though no
individual chosen horizon is 26.5 days.

These are descriptive comparisons between different qualifying samples, design
coverage, gain targets and search/selection procedures. They do not isolate the
causal effect of relaxing Class 2 nonharm. Selection is retrospective using
evaluated gains; the summaries are not independent validation of a newly chosen
optimizer. No confidence or significance claims are implied by the range bars.
