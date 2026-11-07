# Current presentation figures

PNG figures live under FCFS, Policy, Pareto, and Trade-off. Replaced figures have been removed; SVG copies are not kept.

Run from the repository root:

    python scripts/update_presentation_figures.py

Use completed replacement data without merging with older tables:

    python scripts/update_presentation_figures.py --pareto-dir PATH_TO_NEW_PARETO --frontier-dir PATH_TO_NEW_FRONTIER --policy-release PATH_TO_NEW_POLICY_RELEASE

The Pareto folder must contain validated_meaningful_pareto_cells.csv, patient_behavior_factorial_bank.csv, and validated_background_policy_universe.csv. The frontier folder must contain evaluated_selected_cells.csv and background_policy_targets.csv. The policy release must contain paired seed deltas and selected policy outcomes, in the existing release schema.

| Slide | Current figure |
|---|---|
| 17 | Policy/slide17_design_parameters.png |
| 19 | Policy/slide19_demand_activation.png |
| 20 | Policy/slide20_horizon_utilization.png |
| 24 | Pareto/validated_scatter_paretos.png |
| 25 | Pareto/slide25_pareto_success.png |
| 30 | Trade-off/slide30_class1_gain_boxplots.png |
| 32 | Trade-off/slide32_success_by_class2_loss.png |
| 34 | Trade-off/policy_settings_comparison/slide33_horizon_pareto_vs_allowed_loss.png |
| 35 | Trade-off/policy_settings_comparison/slide34_reservation_pareto_vs_allowed_loss.png |
| New companion to 35 | Trade-off/slide35_reservation_periods.png |

Slide 19 intervals: 1,000 paired-seed bootstrap samples, retaining dependence across scenarios and policies. The statistic is the high-minus-low difference of median utilization gains across balking and clinic scenarios. These intervals condition on the fixed scenario design and selected settings.

Slide 20 intervals: 1,500 clinic-context bootstrap samples, preserving the three balking levels within each context. These show uncertainty when treating clinic contexts as a sampled population; they are not simulation-replication intervals or optimizer-selection intervals. Reservation-only's fixed 100-day horizon is omitted from the horizon panel; its utilization is shown.

Slide 24 uses all validated policy settings in the supplied dataset. Added 60%/80%/40-slot policy-level results are not present locally yet, so no new points were invented. Current Pareto coverage is 540 original scenarios. The partial 80% follow-up supplies status only, so it cannot provide new scatter points.

Slide 25 uses the common original scenario universe for all policies (rho <= 2.5), rather than mixing the partial both-flexible-only 80% follow-up into one bar series. This changes the both-flexible percentages from the earlier mixed-coverage figure. New complete input tables will replace this universe directly.

Slide 30 uses one selected both-flexible policy per scenario, separately for Pareto and Class 2 loss allowed. Boxes show quartiles and median; whiskers extend to 1.5 IQR, with outliers shown. These samples have different qualification criteria and scenario coverage.

Slide 32 counts each successful scenario once for each policy, in the bin of the minimum observed actual Class 2 loss among validated tested policies reaching at least 5pp Class 1 gain within the 10pp budget. Bins are 0, >0–1, >1–2, >2–5, >5–10 pp. Zero includes Class 2 gains. Counts are not shares and the two policy series can include the same scenarios.

The reservation-period figure uses the same retrospective best-Class-1-gain settings as the horizon and reservation-percentage comparisons. Existing periods range from 2 to 26 days. All three figures use consistent blue/orange result-group colors.

Displayed cells are called policies, and severity is called post probability. Internal CSV fields and existing source filenames are retained for compatibility. PowerPoint and speaker notes are not edited. Slide 8 remains deferred until its original CI source data is located.
