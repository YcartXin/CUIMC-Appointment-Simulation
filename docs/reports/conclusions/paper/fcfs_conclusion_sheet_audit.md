# Audit: fcfs_conclusion_sheet.tex

Date: May 2026

## Edits Made

- Rebuilt the conclusion sheet as an intentionally airy 2-page document.
- Switched from a dense two-column 10 pt layout to a one-column 11 pt layout.
- Added a short intro box with the purpose of the sheet.
- Reduced visible conclusion bullets to the highest-value claims.
- Replaced separate cancellation, balking, and no-show sections with one compact mechanism table.
- Removed regression-style evidence dumps, long sweep ranges, and calculation details from the visible sheet.
- Added a highlighted reporting recommendation box.
- Ended with one meeting-decision line.
- Kept the document figure-free.

## Numbers Retained in the Visible Sheet

| Visible value | Why retained | Source |
| --- | --- | --- |
| `lambda = 100`, `S = 32` | Identifies the stress-test baseline. | `configs/baseline.yaml` |
| Utilization `0.841` | Shows that high slot use can coexist with poor access. | `docs/reports/metric_analysis/data/baseline_summary.csv`, `average_utilization = 0.840585...` |
| Served rate `0.269` | Paired with utilization to clarify the access problem. | `docs/reports/metric_analysis/data/baseline_summary.csv`, `overall_percent_serviced = 0.268907...` |
| Realistic scenario served-rate gap `0.220` | Preserves the main class-level conclusion. | `docs/reports/metric_analysis/research_style/final/simulation_realistic/fcfs_realistic_comparison.tex`, class table: `0.881 - 0.661 = 0.220` |

## Numbers Removed from the Visible Sheet

| Removed detail | Reason removed | Source retained for traceability |
| --- | --- | --- |
| Regression coefficients `beta = -0.774` and `beta = +0.576` for arrival pressure. | Too much analysis detail for a conclusion sheet. | `docs/reports/metric_analysis/data/regression_standardized_coefficients.csv` |
| No-show regression coefficients, including `-0.289` to `-0.294` and aggregate parameterization `-0.423`. | Coefficients are not needed for the meeting-level conclusion. | `docs/reports/metric_analysis/research_style/final/note/fcfs_research_note_final.tex`; `regression_standardized_coefficients.csv` |
| Cancellation survival calculation `0.9^8.3 ~= 0.42`, accepted wait `8.338`, and cancellation share `32.3%`. | Removed as requested; it is a proof-style calculation, not a conclusion. | `configs/baseline.yaml`; `baseline_summary.csv`; `fcfs_research_note_final.tex` |
| No-balking diagnostic values: `Delta util = -0.001`, CI `[-0.003, 0.000]`, no-offer `0.297`, cancellation `0.384`. | Too detailed for the visible summary. | `results/balking_effect_sweep.csv`; `fcfs_research_note_final.tex` |
| Class 1 balking sweep values `0.298 -> 0.237` and Class 2 `0.240 -> 0.305`. | Replaced by the mechanism-table conclusion. | `docs/reports/metric_analysis/research_style/final/note/fcfs_research_note_final.tex`; `outputs/class1_balking/summary/aggregate_summary.csv` |
| Aggregate served-rate range near `0.269` across the balking sweep. | Replaced by the concise reallocates-access conclusion. | `outputs/class1_balking/summary/aggregate_summary.csv` |
| No-show sweep values, including utilization `0.920 -> 0.681` and Class 2 served rate `0.269`. | Replaced by the concise missed-slots conclusion. | `outputs/class1_no_show/summary/aggregate_summary.csv`; `outputs/class1_no_show/summary/class_summary.csv` |
| Symmetric baseline class gap `0.001`. | Kept only as the qualitative conclusion "near zero." | `baseline_summary.csv`, `access_advantage_class_1 = -0.000354...` |
| Realistic scenario aggregate values `0.934` utilization and `0.789` served rate; Class 1 `0.881`, Class 2 `0.661`. | The visible sheet keeps only the class gap to avoid crowding. | `docs/reports/metric_analysis/research_style/final/simulation_realistic/fcfs_realistic_comparison.tex` |

## Compile Command

```bash
cd reports
pdflatex -interaction=nonstopmode fcfs_conclusion_sheet.tex
pdflatex -interaction=nonstopmode fcfs_conclusion_sheet.tex
```

## Verification

- LaTeX output: `Output written on fcfs_conclusion_sheet.pdf (2 pages, 110925 bytes).`
- macOS metadata check: `kMDItemNumberOfPages = 2`.
- Log check: no overfull or underfull box warnings after the final table adjustment.
- Visual check: rendered both pages with Ghostscript at 120 dpi and inspected them. The PDF is substantially less dense, with clear section breaks, short bullets, an airy mechanism table, and ample whitespace.

## Final Page Count

2 pages.
