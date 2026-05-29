# Professor Comparison Report: What Changed

This note summarizes the new professor-facing report:

- Source: `hypothesis_synthesis_professor_comparison.qmd`
- Rendered HTML: `hypothesis_synthesis_professor_comparison.html`
- Audit record: `hypothesis_synthesis_professor_comparison_audit.md`
- The original `hypothesis_synthesis.qmd` and `hypothesis_synthesis.html` were not modified.

## Format Change

The new report follows the original hypothesis-synthesis format but now leads
with the final professor-facing conclusions. It includes:

- a final P1-P9 hypothesis table;
- conclusion anchors and minimum reporting set;
- code-backed functional mechanics;
- a source comparison table;
- a consolidation map;
- an evidence appendix.

## Main Content Changes

1. Moved the final 9-row professor-facing hypothesis set to the top of the
   report.
2. Added a conclusion-anchor section with the minimum reporting set: served
   rate, utilization, offered wait, no-offer share, outcome decomposition, and
   class-level served rates.
3. Added baseline outcome decomposition: served about 26.9%, balked about
   35.5-35.6%, canceled about 32.3-32.5%, no-show about 5.1%, no-offer 0.0%.
4. Added a functional-mechanics table tying conclusions to the implementation:
   pooled day-level FCFS, earliest-open-day booking, pre-booking balking,
   future-only cancellation, and unrebooked service-day no-show.
5. Marked the no-show conclusion as qualified rather than universally true:
   no-show loss is real, but utilization effects become material in tested
   regimes when demand creates enough no-show-risk delay exposure or when
   low-delay no-show probability is nonzero.
6. Marked the arrival-share claim as not supported as stated: arrival share
   alone does not create a class access gap under symmetric behavior.
7. Kept cancellation as qualified true: future cancellations can be reabsorbed
   under high demand, but this should not be generalized as a recommendation
   that cancellations improve access.
8. Folded threshold ordering into the no-show evidence row rather than keeping
   it as a standalone headline hypothesis, and corrected the interval to
   `theta_no_show < tau <= theta_balk`.
9. Added a caution that regression coefficients are standardized associations
   over sampled configurations, not standalone policy causal effects.

## Redundancy Review Result

The subagent recommended these merges:

- Merge no-show mechanism, no-show demand gating, and threshold ordering into
  one no-show hypothesis.
- Merge balking selection and balking-vs-no-show utilization discussion into
  one balking hypothesis.
- Merge cancellation reabsorption with the cancellation half of the loss-timing
  discussion.
- Merge FCFS class-neutrality and arrival-share evidence into one class-gap
  origin hypothesis.

The final compact table therefore uses 9 adapted hypotheses, labelled P1-P9.
The audit verdict is pass for the current implemented simulation.
