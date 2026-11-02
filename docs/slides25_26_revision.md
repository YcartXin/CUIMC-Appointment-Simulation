# Revised slide 25 and 26 figures

Run `python scripts/plot_slides25_26_revision.py` from the repository root.
Figures: `figures/slides25_26_revision/` (PNG and SVG).
Audit and summary data: `outputs/slides25_26_revision/`.
No source files, previous figures, or PowerPoint slides are overwritten.
All plots omit horizon-only and contain no explanatory footnotes or subtitles.

## Slide 25

`slide25_legacy_pareto_with_80pct` uses both-flexible background success status
from the original experiment and the older 80% follow-up. It does not use the
new frontier selection procedure. Each background counts once.

Original headline subset: rho <=2.5, all original shares/scales. At low,
medium, and high no-show severity, success is 3/150 (2%), 12/150 (8%), and
18/150 (12%). Older 80% follow-up: rho=1.7/2.0 and 30/40/50 slots per day.
Its success counts are 0/18 (0%), 10/18 (55.6%), and 16/18 (88.9%). The
follow-up contains 54 backgrounds, of which 26 qualify.

The figure pools the original and older follow-up backgrounds into one blue
series, as requested. Low/medium/high counts are 3/168 (1.8%), 22/168 (13.1%),
and 34/168 (20.2%). These are ratios of combined counts, not averages of the
two source percentages. Each of the 504 distinct backgrounds counts once.
The source-specific audit CSV is retained alongside the pooled summary.

The two sources cover different demand/share/scale combinations and differ in
search coverage. The pooled rates describe that combined design rather than
an isolated effect of no-show severity. The follow-up success status was
reconstructed from previously reported counts rather than downloaded case
records. No frontier results enter slide 25.

Reservation-only remains absent from this figure: the original experiment
has zero qualifying backgrounds and the older follow-up did not evaluate it.
Omitting it avoids suggesting the follow-up tested reservation-only.

## Slide 26

`slide26_pareto_scale` reproduces the older both-flexible Pareto background
rates with distinct blue, green and orange series for 30/40/50 slots per day,
plus distinct markers and line styles. The 40-slot Pareto result is available
only at 80% share. Missing points are untested, not zero. As previously,
this figure uses rho=1.7/2.0, with 18 backgrounds at each plotted share/scale.
The original qualitative conclusion is retained for these Pareto results:
50 slots/day has more successes at 30-70% share, while 30 slots/day has more
at 80-90% share. At 10% share, neither size succeeds.

`slide26_frontier_allowed_loss_scale` separately shows the newer frontier
results for at least 5pp Class 1 gain within a 10pp Class 2 loss ceiling,
again at rho=1.7/2.0 and 18 backgrounds per point. All tested budgets up to
the ceiling are checked, with each background counted once. Audit checks
confirm that every successful background also has a qualifying evaluated
case with an actual negative Class 2 served-rate delta.

Success percentages (30 / 40 / 50 slots per day):

- 10% Class 1 share: 100 / 100 / 100
- 30%: 100 / 100 / 100
- 50%: 94.4 / 94.4 / 100
- 70%: 16.7 / 33.3 / 16.7
- 80%: 16.7 / 5.6 / 16.7
- 90%: 0 / 0 / 0

These results do not clearly support a general shift from larger to smaller
systems as Class 1 share rises. The sizes mostly tie at low shares. At 70%,
40 slots/day does best. At 80%, 30 and 50 tie. At 90%, all fail the target.
The stronger finding is declining success at high Class 1 shares.
This comparison changes the gain threshold (old Pareto: at least 0.5pp gain
for either class with neither harmed; new figure: at least 5pp Class 1 gain),
as well as search/selection and allowed Class 2 harm. It cannot establish the
isolated effect of allowing Class 2 loss. These are finite-grid descriptive
rates, not estimates of population probabilities or significance tests.

## Data still unavailable

There is enough local data for all three figures above. The older 80% follow-up
has no local evaluated case-level metrics or policy settings. Its utilization,
selected horizon and reservation percentages cannot be recreated under the old
selection procedure without the follow-up evaluated-case CSV and matching
background-policy table. The newer frontier has these metrics, but substituting
it would change the experiment and selection rule.

## Counting unit verification

Slide 25 checks unique background IDs after restricting to both-flexible.
Both slide 26 figures also use one row per background, with 18 distinct
backgrounds per share/scale point. Multiple qualifying cases or loss budgets
do not contribute extra observations. Success means at least one qualifying
case exists; these rate figures do not average cases or require selecting
one case's performance metrics.
