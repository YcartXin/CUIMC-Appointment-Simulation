# Audit: Professor-Facing Hypothesis Synthesis

Audited file:

- `docs/reports/Hypothesis Synthesis/hypothesis_synthesis_professor_comparison.qmd`

Rendered file:

- `docs/reports/Hypothesis Synthesis/hypothesis_synthesis_professor_comparison.html`

Original files intentionally left unchanged:

- `docs/reports/Hypothesis Synthesis/hypothesis_synthesis.qmd`
- `docs/reports/Hypothesis Synthesis/hypothesis_synthesis.html`

## Audit Scope

The audit treated this document as the conclusion layer, not another
investigation report. The check asked whether the final hypotheses are supported
by:

- the conclusion reports in `docs/reports/conclusions`;
- the final mechanics note
  `docs/reports/metric_analysis/research_style/final/note/fcfs_research_note_final.tex`;
- raw and summary experiment outputs in `outputs/` and `results/`;
- the simulation implementation in `simulation/engine.py`,
  `simulation/model.py`, and `analysis/metrics.py`.

## Subagent Audit Results

Three read-only subagents reviewed the work:

1. Evidence and numbers audit: conditional pass. No material numeric
   mismatches. Recommended tightening no-show demand-gating language,
   arrival-share scope, and regression-screen wording.
2. Code mechanics audit: pass. Confirmed pooled day-level FCFS, offered delay
   before balking, pre-booking balking, future-only cancellation, service-day
   no-show loss, and metric denominators. Recommended precision on threshold
   intervals and no-show assumptions.
3. Structure/redundancy audit: pass with presentation revisions. Recommended
   putting final hypotheses first, then using the comparison table only as
   traceability.

## Revisions Made From Audit

- Moved the final P1-P9 hypothesis table to the top of the report.
- Added a conclusion-anchor section with the required minimum reporting set:
  served rate, utilization, offered wait, no-offer share, outcome
  decomposition, and class-level served rates.
- Added baseline outcome decomposition: served about 26.9%, balked about
  35.5-35.6%, canceled about 32.3-32.5%, no-show about 5.1%, no-offer 0.0%.
- Added a functional-mechanics table linking code behavior to conclusions.
- Tightened no-show wording from a universal "only when" statement to:
  effects become material in tested regimes when offered delays frequently
  cross the no-show-risk region or low-delay no-show probability is nonzero.
- Scoped arrival-share wording to the tested symmetric range, Class 1 share
  0.30-0.70.
- Reworded the class-gap driver result to distinguish broad regression-screen
  associations from realistic local attribution.
- Corrected the threshold-ordering interval to
  `theta_no_show < tau <= theta_balk`; with thresholds 6 and 9, this is integer
  delays 7 through 9.
- Replaced the repeated reduced hypothesis table with a consolidation map.
- Removed draft-like phrases such as "keep if space allows" and
  "mostly supported but needs sharper wording."

## Final Audit Verdict

Pass.

The professor-facing report is conclusive for the current implemented
simulation. No additional experiments are needed to support the nine final
hypotheses within this model. Further work would be external-validity or
policy-extension work: clinic calibration, alternative no-show timing, or
non-FCFS priority rules.

