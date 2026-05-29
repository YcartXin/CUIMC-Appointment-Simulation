# Hypothesis Synthesis Reconciliation: What Changed

This companion note summarizes the differences between the original
`hypothesis_synthesis.qmd` and the new reconciled version:

- `hypothesis_synthesis.qmd` and `hypothesis_synthesis.html` were restored and
  left as the original report.
- `hypothesis_synthesis_reconciled.qmd` is the new source report.
- `hypothesis_synthesis_reconciled.html` is the rendered standalone HTML.

## Main Structural Changes

1. Replaced the broken draft summary table with a hypothesis crosswalk.
2. Added a conclusion-by-conclusion check against the conclusion reports.
3. Added a "New Hypotheses Flagged" section for findings that were absent from
   the original synthesis.
4. Kept revised H1-H4 sections so the original hypotheses can still be traced
   back to their evidence plots.

## Key Substantive Changes

- H1 cancellation was corrected. The original text claimed cancellation
  shortens offered delay, but its evidence paragraph said offered-delay lines
  overlap. The reconciled report uses the sweep and regression evidence:
  cancellation lowers own-class served rate, can raise other-class access, and
  can shorten offered wait under high demand.
- H2 no-show was qualified. The stress-test no-show sweep supports no-show as
  unrebookable service-day loss, but newer conclusion experiments show the
  utilization effect is negligible at the realistic demand ratio and activates
  only at higher demand.
- H3 balking was reframed as selection plus reallocation. Lower accepted wait
  is not a true improvement when served rate falls.
- H4 was upgraded from ambiguous "partially supported" wording to a cleaner
  affected-class qualifier: aggregate served rate is nearly flat under balking
  because released access is absorbed by the other class.

## New Hypotheses Flagged

- H8: Class-gap driver hierarchy is regime-specific.
- H9: No-show utilization effects are demand-regime activated.
- H10: Arrival share alone does not create a class gap under symmetric behavior.
- H11: Booking horizon amplifies the realistic class gap until about 21 days,
  then plateaus.
- H12: Threshold ordering creates an accept-then-miss interval.

## Source Priority Used

The reconciled report treats the following as the main evidence hierarchy:

- Full mechanics source: `fcfs_research_note_final.tex`
- Meeting conclusions: conclusion sheet, supervisor brief, and deck
- New conclusion results: `docs/reports/conclusions/deck/UPDATE.md`
- Raw CSV checks from `outputs/`, `results/`, and `docs/reports/metric_analysis/data/`
