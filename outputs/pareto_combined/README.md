# Combined Experiment 2 Pareto data

The background-policy table combines the original experiment with the 80% Class 1 follow-up. Each row represents one behavioral background and one evaluated policy. All original demand levels are retained.

Sources: original local outputs/pareto_presentation validation and design bank; follow-up local design bank plus the complete 54-background validated-status breakdown reported by the user in FCFS Queueing Analysis (conversation 6a794410-c00c-83e8-bf74-2b4a2b0ebe0e, October 9, 2026). Follow-up status is reconstructed from that breakdown, not from downloaded Grid evaluation files. No new simulations were run.

The follow-up evaluated only both_flexible at Class 1 share 0.8, rho 1.7 and 2.0, capacities 30/40/50. Missing follow-up rows for other policies or demand levels are untested, not negative results. Follow-up n_validated_cells, service-rate deltas and utilization deltas are blank because they were not supplied. cell_results_available=False identifies this limitation. Follow-up policy settings and seed/cell-level outcomes cannot yet be jointly analyzed.

validated_pareto_exists means at least one searched policy setting passed independent-evaluation mean nonharm for both classes and at least 0.005 served-rate gain for one class. This does not itself establish confidence-interval-supported nonharm. Rates are fractions, e.g. 0.556 means 55.6%. class1_share is also a fraction. horizon_days in the background table is the design baseline horizon, not a selected policy horizon.

Files:
- combined_background_policy_universe.csv: full combined background-level dataset with source, design, validation status and available original metrics.
- pareto_by_context.csv: counts and rates by source, policy, demand, share and capacity.
- both_flexible_rho17_20.csv: comparable demand subset for joint share/scale/behavior analysis.
- both_flexible_share_scale_summary.csv: pooled counts across rho 1.7/2.0; each share-capacity point has 18 backgrounds.
- original_cell_results_with_context.csv: original candidate-level validation records with design variables; no follow-up cell records are available.

Checks: unique background/policy and design/policy keys, original rows retained, 54 follow-up rows, 26 follow-up qualifying backgrounds, six reported context counts matched exactly. Do not treat backgrounds as random population samples or pool unmatched rho ranges for share comparisons. Search coverage differs: the follow-up used systematic five-seed discovery; source identifiers preserve this difference.

To complete cell-level integration, obtain the Grid follow-up validated_background_policy_universe.csv and independent_evaluation/validated_meaningful_pareto_cells.csv from served_rate_pareto/pareto_catalog_5seed. Replace the reconstructed follow-up rows by background_id/policy rather than appending them again, and merge follow-up cell records with the follow-up design bank.
