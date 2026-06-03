# Exploratory Notebooks

Open these notebooks in order:

1. `01_simulation_basics.ipynb` explains the core code path from YAML config to simulation results and shared metric helpers.
2. `02_metric_driver_slices.ipynb` recreates simple one-driver metric slices where Class 1 varies and Class 2 stays fixed.
3. `03_heatmaps_and_report_artifacts.ipynb` builds compact interaction heatmaps and inspects the generated metric-report manifest.
4. `04_metric_report_walkthrough.ipynb` mirrors `docs/reports/metric_analysis/metric_analysis.qmd` in metric-first order and then shows every generated report figure not embedded in the prose.
5. `05_reserved_slot_strategy.ipynb` compares strict Class 1 reserved slots with released reserved slots that Class 2 can backfill.
6. `06_reserved_slot_visual_walkthrough.ipynb` shows the same arrivals under pooled, strict reserved, and released reserved booking with slot-grid allocation views.
7. `appointment_simulation_analysis.ipynb` is the broader scenario walkthrough, now adapted to the cleaned `simulation/` package layout.
8. `parameter_sensitivity_analysis.ipynb` is the larger exploratory sweep workbook, using the shared metric and plot-style helpers.

The first two notebooks are intentionally smaller than `scripts/generate_metric_analysis_figures.py`.
They are for understanding and dynamic exploration. The metric report walkthrough is the notebook version of the rendered report plus its generated-figure appendix.

All notebooks locate the repository root automatically, so they can be opened from either the repository root or this `notebooks/` directory. Outputs are intentionally cleared in version control; rerun the cells locally when exploring.
