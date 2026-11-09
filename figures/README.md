# Current presentation figures

PNG figures live under FCFS, Policy, Pareto, and Trade-off. PowerPoint is not edited by the figure scripts.

## Completed Experiment 2 (November 9, 2026)

Run from the repository root:

```bash
python scripts/plot_completed_experiment2.py
```

With the completed data present, the ordinary `python scripts/update_presentation_figures.py` command also runs this generator. Its explicit input-directory flags retain the older workflow; do not use that older workflow for the new ratio-selection results.

Inputs are the extracted archive under `outputs/experiment2_ratio_horizon/{tradeoff,pareto}/`. The generator checks frozen hashes, matching scenario banks, complete coverage, selected settings, and independent evaluation seed results before plotting. It writes audit tables under `outputs/experiment2_ratio_horizon/summaries/`. Previous figures are preserved under `archive/pre_completed_ratio_figures_20261109/` before replacement. FCFS figures and slides 19/20 are unchanged.

All new comparisons use **both flexible**, including eligible zero-reservation settings. Blue is Pareto; orange is Class 2 loss allowed. Policy-setting and gain summaries use one policy per qualifying scenario. Markers show medians and faint dashed lines show observed minima and maxima, not confidence intervals.

| Figure | File under figures/ |
|---|---|
| Design parameters: 1,134 scenarios | Policy/slide17_design_parameters.png |
| Selected Pareto served-rate gains | Pareto/validated_scatter_paretos.png |
| Pareto success by no-show post probability | Pareto/slide25_pareto_success.png |
| Pareto system-size comparison | Pareto/slide26_pareto_scale.png |
| Trade-off system-size comparison | Trade-off/scale/slide26_frontier_allowed_loss_scale.png |
| Selected Pareto utilization | Pareto/best_cases/slide27_utilization_background_best.png |
| Class 1 gain distributions by rho | Trade-off/slide30_class1_gain_boxplots.png |
| Minimum observed loss among successful tested choices | Trade-off/slide32_success_by_class2_loss.png |
| Booking horizons | Trade-off/policy_settings_comparison/slide33_horizon_pareto_vs_allowed_loss.png |
| Reservation percentages | Trade-off/policy_settings_comparison/slide34_reservation_pareto_vs_allowed_loss.png |
| Reservation periods | Trade-off/slide35_reservation_periods.png |
| Class 1 median/min/max gain by rho | Trade-off/policy_settings_comparison/class1_served_rate_gain_by_rho.png |
| Success by tested loss budget | Trade-off/success_rate_by_tested_budget.png |
| Share/rho success, first successful tested budget, failure, budget/loss distributions | Trade-off/experiment2_frontier/fig1*, fig2*, fig3*, fig5* |

The slide numbers in filenames are historical; the deck may now use different page numbers. The two older Pareto scale paths are retained as identical aliases. The obsolete policy-exclusive fig6 is archived because the requested plots use both flexible only.

See `docs/experiment2_completed_results.md` for selection rules, denominators, and interpretation. In particular, first-successful-budget summaries are descriptive across frozen budget choices; they do not replace the fixed 10pp-budget selection in the principal comparisons.

## Unchanged earlier figures

Slide 19 intervals: 1,000 paired-seed bootstrap samples, retaining dependence across scenarios and policies. The statistic is the high-minus-low difference of median utilization gains across balking and clinic scenarios. These intervals condition on the fixed scenario design and selected settings.

Slide 20 intervals: 1,500 clinic-context bootstrap samples, preserving the three balking levels within each context. These show uncertainty when treating clinic contexts as a sampled population; they are not simulation-replication intervals or optimizer-selection intervals. Reservation-only's fixed 100-day horizon is omitted from the horizon panel; its utilization is shown.

Displayed scenarios correspond to the internal `background_id` field. Slide 8 remains deferred until its original CI source data is located.
