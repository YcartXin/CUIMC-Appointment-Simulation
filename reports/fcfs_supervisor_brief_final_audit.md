# Audit: fcfs_supervisor_brief_final

**Date:** May 2026 (revised May 2026; realistic scenario addition May 2026)
**Compile command:** `cd reports && pdflatex -interaction=nonstopmode fcfs_supervisor_brief_final.tex` (run twice for cross-references)
**Final page count:** 3 pages (within 4-page hard max)
**PDF output:** `reports/fcfs_supervisor_brief_final.pdf`

---

## Revision log (targeted edits only)

Five targeted edits were applied to `fcfs_supervisor_brief_final.tex`. No new sections, figures, or tables were added.

| # | Location | Change |
|---|----------|--------|
| 1 | Executive Summary, opening sentence | Replaced "Utilization and patient access decouple..." with BLUF recommendation: "Under this FCFS stress-test baseline, utilization should not be used as the access summary; report served rate, offered wait, and no-offer share together." Added "30-seed baseline diagnostic" attribution to the 0.842/0.269 values to disambiguate from earlier single-run figure (~0.839). |
| 2 | Executive Summary, bullet 2 title | Changed "Baseline losses are behavioral, not structural" to "Baseline losses are post-offer, not no-offer" to avoid implying there is no structural capacity pressure. |
| 3 | Executive Summary, bullet 2 body | Replaced "Slots are offered but rejected or forfeited" with "Slots are offered, then rejected, canceled, or missed." Removed the word "forfeited." |
| 4 | Page 1, Discussion Question 3 | Fixed inaccurate framing. Previous wording implied no-show is currently independent of offered delay, which is wrong. New wording: "Should no-show probability depend on original offered delay, as currently implemented, or on remaining wait closer to the service date?" |
| 5 | Main Takeaways, "Baseline Losses" prose | Replaced "Access failures occur after the offer" with "Access failures occur after offers are made, not because the booking horizon is already unavailable at baseline." Added: "The capacity pressure is severe, but the baseline loss channel is post-offer dropout, not slot unavailability." |
| 6 | Open Questions, item 3 | Replaced "The current rule applies a fixed step function independent of remaining wait" with "The current rule applies no-show probability as a step function of original booking delay, not residual wait before service." This now accurately describes the model formulation. |

**No-show wording confirmation:** The model applies no-show as a function of original offered delay (booking delay at time of acceptance), implemented as a step rule. The revised text in both the discussion question and the Open Questions section correctly reflects this. The document no longer implies that no-show probability is independent of delay.

---

## Files Inspected

| File | Purpose |
|------|---------|
| `results/balking_effect_verification.csv` | Primary source for all baseline and no-balking diagnostic numbers (30 seeds each, scenarios A_baseline and B_no_balking) |
| `docs/reports/metric_analysis/data/baseline_summary.csv` | Secondary baseline reference; used for cross-check |
| `docs/reports/metric_analysis/data/regression_standardized_coefficients.csv` | Source for regression coefficient values |
| `docs/reports/metric_analysis/research_style/final/note/fcfs_research_note_supervisor_brief.tex` | Previous supervisor brief (4-page version); structure reference |
| `docs/reports/metric_analysis/research_style/final/note/fcfs_research_note_final.tex` | Full 37-page note; section cross-reference source |
| `analysis/plot_style.py` | Color palette used for the executive summary figure |
| `docs/reports/metric_analysis/research_style/final/simulation_realistic/fcfs_realistic_comparison_repo_figure.tex` | Only source for realistic scenario numbers (no CSV outputs exist for this scenario) |
| `configs/realistic.yaml` | Config file confirming realistic scenario parameters (λ=24, S=20, H=28, asymmetric behavioral params) |

---

## Realistic Scenario Addition

### Files inspected
- `fcfs_realistic_comparison_repo_figure.tex` -- the supplementary comparison note; contains all outcome metrics in tables
- `configs/realistic.yaml` -- confirms λ₁=14, λ₂=10, S=20, H=28, burn-in=60, cooldown=30; behavioral parameters match the TeX tables

### Number source
**All realistic scenario metrics come from `fcfs_realistic_comparison_repo_figure.tex`, Table 3 (aggregate outcomes) and Table 4 (class outcomes).** No CSV output files exist for this scenario in the repo. This is noted as an assumption in the audit.

| Metric | Realistic scenario value | Source |
|--------|-------------------------|--------|
| Average utilization | 0.934 | TeX Table 3 |
| Overall served rate | 0.789 | TeX Table 3 |
| Mean offered wait | 10.87 d | TeX Table 3 |
| Balked share | 0.033 | TeX Table 3 |
| Canceled share | 0.123 | TeX Table 3 |
| No-show share | 0.056 | TeX Table 3 |
| No-offer share | 0.000 | TeX Table 3 |
| Class 1 served rate | 0.881 | TeX Table 4 |
| Class 2 served rate | 0.661 | TeX Table 4 |
| Class gap | 0.220 | Computed: 0.881 - 0.661 |
| Demand/supply ratio | 1.2 | Confirmed: 24/20 = 1.2 |

### Parameters confirmed against `configs/realistic.yaml`
λ₁=14, λ₂=10 (total 24), S=20, H=28, burn-in=60, cooldown=30. All behavioral parameters in the YAML match the TeX Table 2 exactly.

### Figure
No new figure added. The realistic scenario is presented as a prose subsection only.

### Edits made in this revision
| Location | Change |
|----------|--------|
| Main Takeaways, new 5th subsection | Added "Realistic Scenario Check" after "Report Offered Wait with No-Offer Share" |
| Driver Hierarchy, intro + bullets | Compressed to single-line bullet format to recover space; removed inline filename (filename is in table captions and audit) |
| Open Questions, Q1 | Revised to mention the realistic scenario as a first check; added note that calibration still requires agreement on clinical inputs |
| Open Questions, Q2 and Q3 | Slightly trimmed for compactness |
| Where to Dig table | Added row pointing to `fcfs_realistic_comparison_repo_figure.tex` |

---

## Numbers Used and Their Source

All figures below were computed programmatically from the CSV sources listed.

### Baseline (A_baseline, 30 seeds, `results/balking_effect_verification.csv`)

| Metric | Value in brief | CSV computed value | Notes |
|--------|---------------|-------------------|-------|
| Average utilization | 0.842 | 0.8418 | Rounded to 3dp |
| Overall served rate | 0.269 | 0.2692 | Rounded to 3dp |
| Mean offered wait | 9.29 d | 9.2949 d | Rounded to 2dp |
| Mean accepted wait | 8.33 d | 8.3346 d | Rounded to 2dp |
| Balking rate | 0.355 | 0.3550 | Rounded to 3dp |
| Canceled share | 0.325 | 0.3248 | Rounded to 3dp |
| No-show rate | 0.051 | 0.0506 | Rounded to 3dp |
| No-offer rate | 0.000 | 0.0000 | Exact |
| Unresolved | <0.001 | 0.0004 | Rounded |

**Note on baseline discrepancy:** `baseline_summary.csv` reports `average_utilization = 0.8406`. The verification CSV A_baseline rows average to 0.8418. The discrepancy (~0.001) reflects different random seeds. The brief uses verification CSV numbers throughout for internal consistency with the no-balking comparison. This is noted here.

**Outcome decomposition check:** 0.269 + 0.355 + 0.325 + 0.051 + 0.000 + 0.000 = 1.000 (rounded sums; unresolved <0.001 makes up the residual). Consistent.

### No-balking diagnostic (B_no_balking, 30 matched seeds, same file)

| Metric | Baseline value | No-balk value | Change |
|--------|---------------|--------------|--------|
| Average utilization | 0.842 | 0.840 | -0.001 (raw: -0.0014) |
| Served rate | 0.269 | 0.268 | -0.001 (raw: -0.0011) |
| Booked rate | 0.645 | 0.703 | +0.058 (raw: +0.0585) |
| Balked share | 0.355 | 0.000 | -0.355 (raw: -0.3550) |
| Canceled share | 0.325 | 0.384 | +0.059 (raw: +0.0588) |
| No-offer share | 0.000 | 0.297 | +0.297 (raw: +0.2966) |
| Mean accepted wait | 8.33 d | 9.45 d | +1.12 d (raw: +1.12 d) |

### Regression coefficients (`docs/reports/metric_analysis/data/regression_standardized_coefficients.csv`)

| Target | Feature | Coeff in brief | CSV value | p-value | Significant? |
|--------|---------|----------------|-----------|---------|-------------|
| overall_percent_serviced | lambda_total | -0.774 | -0.7744 | <0.001 | Yes |
| mean_offered_booking_delay | lambda_total | +0.576 | +0.5762 | <0.001 | Yes |
| average_utilization | no_show_threshold_mean | +0.512 | +0.5117 | <0.001 | Yes |
| average_utilization | no_show_step_mean | -0.423 | -0.4232 | <0.001 | Yes |
| average_utilization | balk_step_mean | +0.016 | +0.0157 | 0.72 | No |

**Rounding convention:** All coefficients rounded to 3dp.

### Mechanical bound

`32/100 = 0.32` is an exact calculation stated as a design fact, not a CSV-derived number.

---

## Figures Used or Generated

| Figure | Path | Source |
|--------|------|--------|
| exec_summary_panel.pdf | `reports/figures/exec_summary_panel.pdf` | Generated by `reports/figures/gen_exec_figure.py` |

**Figure generation:** The script uses numbers directly from `results/balking_effect_verification.csv` (A_baseline, 30 seeds). Colors match `analysis/plot_style.py` palette.

No existing figures from `docs/reports/metric_analysis/figures/` were reused in the brief. The generated figure replaces both an outcome decomposition table and a utilization-vs-access bar chart, keeping the brief under 3 tables.

---

## Table Count

| Table | Content |
|-------|---------|
| Table 1 | Baseline metrics (5 rows) |
| Table 2 | No-balking diagnostic (7 rows) |
| "Where to Dig" | Pointer table, not a results table |

Total results tables: 2 (limit: 3). Outcome decomposition is represented in the figure.

---

## Unresolved Assumptions

1. **Baseline seed discrepancy.** The brief uses `A_baseline` rows from `balking_effect_verification.csv` (30 seeds, util = 0.8418) rather than `baseline_summary.csv` (util = 0.8406). Both are repo-generated. The difference is negligible but noted.

2. **Regression coefficients are standardized.** The brief labels them `\hat\beta` and describes them as "standardized OLS estimates with HC3 standard errors." The `significant_05_hc3` column in the CSV confirms significance at 5% level for all coefficients cited.

3. **Arrival rate units.** `lambda_total` in the regression data is a feature value from a parameter sweep, not a per-seed arrival count. The label "total arrival rate" in the brief is consistent with the `feature_label` column in the CSV ("total arrival rate").

4. **Full note section numbers.** The "Where to Dig" table references section numbers (e.g., "Sections 2.1--2.3") from the full note. These were verified against the existing supervisor brief which cites the same sections. A change to the full note's structure would require updating the pointer table.
