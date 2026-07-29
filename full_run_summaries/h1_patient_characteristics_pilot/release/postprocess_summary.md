# H1 policy-outcome postprocessing: release

Backgrounds processed: 420
Bootstrap draws per paired delta: 2,000
Practical-change tolerance: 0.005

## Output row counts

- selected_policy_seed_outcomes.csv: 16,800
- selected_policy_outcomes.csv: 3,360
- pairwise_group_deltas.csv: 5,040
- objective_switch_deltas.csv: 1,680
- selection_validation.csv: 0

## Interpretation

- Pairwise deltas are first policy minus second policy.
- Class 1 is the priority group only when Q > 0; otherwise both classes use the general pool.
- weighted_utilization gives Class 1 served rate twice Class 2's policy weight.
- objective_switch_deltas are weighted-optimal minus average-optimal.

## Selection validation

Existing condition_optima.csv files were not checked.
