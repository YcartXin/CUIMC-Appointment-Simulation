# Changelog: fcfs_research_note_merged.tex

Describes what was imported from each source, figure placement decisions,
and remaining validation items.

---

## 1. Base Version

**`metric_analysis_research_note.pdf`** (represented by
`outputs/reports/metric_analysis/rendered/metric_analysis.tex` and the
canonical Quarto source `docs/reports/metric_analysis.qmd`)

Used as the structural base. The interpretation framing, metric definitions,
sensitivity methodology, regression screen, and section ordering are all
drawn from this version. The final takeaways and bottom-line language
were absorbed and rewritten into the new Summary for Discussion section.

---

## 2. What Was Imported from Each Other Version

### `metric_analysis_supervisor_version.pdf`
- **Metric hierarchy table** (Section 6): four-column format (metric / what it measures / interpretation / main caveat).
- **Baseline summary table** (Section 3): compact two-scenario layout with six metrics.
- **Concise baseline summary prose**: short paragraph framing utilization vs. served rate.

### `metric_analysis.pdf`
- **Full regression table** (Appendix D): all 19 rows with standardized coefficients, 95% HC3 CIs, and p-values.
- **Regression coefficient figure** (`regression_significant_standardized_coefficients.png`): retained in appendix.
- **FCFS stress curves figure** (`fcfs_capacity_stress_curves.png`): retained in Section 5.
- **Arrival outcome decomposition figure** (`fcfs_arrival_outcome_decomposition.png`): retained in Section 4.
- **Scenario comparison figure** and all supplementary heatmaps: moved to Appendix C.

### `metric_analysis-1.pdf`
- **Exact metric formulas** (Appendix A): full aligned-equation block with signed gap definitions.
- **Behavioral step-function notation**: piecewise definition of b_i(tau) and xi_i(tau).
- **Fine-grid sensitivity design table** (Appendix B): parameter ranges and step sizes.
- **Balking deep-dive design descriptions** (Appendix B): 100-seed per point detail, threshold sweep range 3-13 days.

### `simulation_documentation.html`
- **Section 2 (Simulation Mechanics)**: 13-bullet enumeration from Daily Simulation Order, Probability Rules, and Metric Definitions.
- **Utilization definition**: exact formula and note that no-shows do not count.
- **Offered vs accepted delay distinction**: from audit record (i, tau) discussion.
- **No-offer mechanics** and **no-show anchoring to original tau**.

---

## 3. Figures Kept in the Main Body

| Figure file | Section | Reason |
|---|---|---|
| `fcfs_arrival_outcome_decomposition.png` | §4 | Central diagnostic; balking/cancellation split |
| `metric_access_drivers.png` | §5 | Served rate response to demand |
| `fcfs_capacity_stress_curves.png` | §5 | Utilization, served rate, no-offer over wide demand range |
| `metric_utilization_drivers.png` | §7 | Utilization response to no-show vs. demand |
| `metric_wait_drivers.png` | §7 | Offered-wait response to demand, balking, cancellation |
| `percent_serviced_by_class.png` | §8 | Served rate under Class 1 balking sweep |
| `mean_accepted_delay_by_class.png` | §8 | Selection effect on accepted wait |

---

## 4. Figures Moved to Appendix C

| Figure file | Reason for moving |
|---|---|
| `no_show_step_interaction_heatmap.png` | Dense; pattern summarized in text |
| `no_show_threshold_interaction_heatmap.png` | Pattern described in sensitivity screen |
| `cancellation_probability_interaction_heatmap.png` | Supports validation check |
| `balking_step_interaction_heatmap.png` | Covered by text |
| `balking_threshold_interaction_heatmap.png` | Covered by text |
| `arrival_mix_interaction_heatmap.png` | Demand effect shown by stress curves |
| `scenario_metric_comparison.png` | Scenario 2 is comparison point only |
| `balking_threshold_jump_heatmaps.png` | Remains in appendix |
| `no_show_threshold_jump_heatmaps.png` | Remains in appendix |
| `regression_significant_standardized_coefficients.png` | Regression is appendix material |

---

## 5. Remaining Validation Checks

| Check | Priority | Description |
|---|---|---|
| Cancellation loop logic | High | Confirm one draw per patient per residual day; verify rebooked slot rate |
| Positive cancellation coefficient | High | Run targeted experiment: how many canceled slots are rebooked, and by which class |
| No-offer propagation within a day | Medium | Confirm all remaining queue patients get no-offer when horizon full; permutation re-drawn each day |
| No-show slots never rebooked | Medium | Confirm in code; document as explicit design assumption |
| Same-day cancellations excluded | Medium | Confirm r=0 patients skip cancellation draw |
| Class-gap stability | Low | Re-run heatmaps with 3-5 seeds per grid point |
| Demand/capacity ratio calibration | Low | 100 arrivals / 32 slots: stress scenario or realistic baseline? |
| Behavioral parameter calibration | Low | Balking step 0.50, no-show step 0.30, cancellation 0.10 are assumed; require empirical grounding |
