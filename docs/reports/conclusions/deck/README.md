# FCFS Conclusions Package

May 2026 — Two-class discrete-event FCFS appointment simulation.

This package contains the slide deck and supporting figures for the five main conclusions from the FCFS simulation work. The key numbers below list their source files.

---

## Files

```
deck/
├── fcfs_conclusions_deck.pdf        ← compiled slide deck (5 slides, 16:9)
├── fcfs_conclusions_deck.tex        ← LaTeX source (beamer 18pt)
├── README.md                        ← this file
└── figures/
    ├── s1_outcome_decomposition.png       ← Slide 1: arrival outcome breakdown
    ├── s1_scenario_metric_comparison.png  ← Slide 1: util vs served rate across scenarios
    ├── s2_capacity_stress_curves.png      ← Slide 2: stress-test capacity curves
    ├── s2_stress_access_wait.png          ← Slide 2: access and wait under stress
    ├── s3_balking_step_slice_access.png   ← Slide 3: balking step vs served rate (flat)
    ├── s3_balking_step_slice_utilization.png  ← Slide 3: balking step vs utilization (flat)
    ├── s3_balking_effect_verification.png ← Slide 3: 4-scenario diagnostic (A/B/C/D)
    ├── s4_no_show_slice_utilization.png   ← Slide 4: no-show rate drives utilization down
    ├── s4_no_show_slice_access.png        ← Slide 4: no-show rate drives access down
    ├── s5_class_served_rate_gap.png       ← Slide 5: class access gap, realistic scenario
    ├── s5_class_gap_drivers.png           ← Slide 5: regression screen, class gap drivers
    ├── bg_regression_drivers.png          ← Background: full regression driver chart
    └── bg_exec_summary_panel.png          ← Background: executive summary panel
```

---

## Slide–Figure Map

| Slide | Headline | Supporting figures |
|-------|----------|--------------------|
| 1 | Utilization alone is not an access summary. | `s1_outcome_decomposition`, `s1_scenario_metric_comparison` |
| 2 | Under high demand, high utilization can coexist with poor access. | `s2_capacity_stress_curves`, `s2_stress_access_wait` |
| 3 | Balking reallocates access; it does not recover aggregate capacity. | `s3_balking_step_slice_access`, `s3_balking_step_slice_utilization`, `s3_balking_effect_verification` |
| 4 | Cancellation can be reabsorbed; no-show capacity is lost. | `s4_no_show_slice_utilization`, `s4_no_show_slice_access` |
| 5 | Aggregate metrics can hide class-level access gaps. | `s5_class_served_rate_gap`, `s5_class_gap_drivers` |

---

## Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Stress-test arrivals λ | 100 /day | `configs/baseline.yaml` |
| Stress-test slots S | 32 /day | `configs/baseline.yaml` |
| Baseline utilization | 0.841 | `docs/reports/metric_analysis/data/baseline_summary.csv` |
| Baseline served rate | 0.269 | same |
| No-balking no-offer share | ~0.297 | `results/balking_effect_verification.csv`, B_no_balking rows |
| Realistic class gap | 0.220 | `docs/reports/metric_analysis/research_style/final/simulation_realistic/fcfs_realistic_comparison.tex` |
| Symmetric baseline class gap | ~0 (−0.000354) | `baseline_summary.csv`, `access_advantage_class_1` |

---

## Compile

```bash
cd docs/reports/conclusions/deck
pdflatex -interaction=nonstopmode fcfs_conclusions_deck.tex
pdflatex -interaction=nonstopmode fcfs_conclusions_deck.tex
```

Output: 5 pages.

---

## Background reports (not in this package, in the repo)

| Report | Location |
|--------|----------|
| Full research note | `docs/reports/metric_analysis/research_style/final/note/fcfs_research_note_final.pdf` |
| Realistic scenario comparison | `docs/reports/metric_analysis/research_style/final/simulation_realistic/fcfs_realistic_comparison.pdf` |
| Conclusion sheet (2-page memo) | `docs/reports/conclusions/paper/fcfs_conclusion_sheet.pdf` |
| Thorough metric analysis | `docs/reports/metric_analysis/thorough/metric_analysis.pdf` |
