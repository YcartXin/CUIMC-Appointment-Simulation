# H1 Short-Horizon Reservation (release): Summary

Optimization objective: average_utilization
Practical-equivalence tolerance: 0.005
Weighted-utilization weights: w1=2.0, w2=1.0
Backgrounds classified: 840

This is an auto-generated data summary, not the narrative report.
condition_optima.csv has one row per background with each of the four
conditions' optimal (horizon, Q, window) and both utilization metrics.
condition_deltas.csv has all six paired-seed-bootstrap policy comparisons.
For reservation_only_vs_horizon_only, a positive delta means the
reservation-only policy is higher; a negative delta means horizon-only is higher.

Dominance check: 100.0% of backgrounds had both_flexible's average_utilization at or above all three other conditions (within the 0.005 practical-equivalence tolerance) -- the expected weak-dominance property, since both_flexible's search space contains the other three.
