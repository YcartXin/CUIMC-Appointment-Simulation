# H1 policy-outcome postprocessing: release

Backgrounds processed: 840
Bootstrap draws per paired delta: 2,000
Practical-change tolerance: 0.005

## Output row counts

- selected_policy_seed_outcomes.csv: 67,200
- selected_policy_outcomes.csv: 6,720
- pairwise_group_deltas.csv: 10,080
- objective_switch_deltas.csv: 3,360
- selection_validation.csv: 6,720

## Interpretation

- Pairwise deltas are first policy minus second policy.
- Class 1 is the priority group only when Q > 0; otherwise both classes use the general pool.
- weighted_utilization gives Class 1 served rate twice Class 2's policy weight.
- objective_switch_deltas are weighted-optimal minus average-optimal.

## Selection validation

Summary rows found: 6,720/6,720
Exact policy-cell matches: 6,548/6,720
Weighted-objective mismatches can occur if the supplemental average-objective refinement added new cells that were not present when the older weighted summary was classified.
