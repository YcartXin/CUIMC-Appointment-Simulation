# Weight Sensitivity For Strict Reservation

This folder scores the same baseline simulation runs under several class-weight choices.
No new booking policy is introduced here; the weights only change the objective values.
Here, utilization means completed visits divided by available slots.

Baseline slice: `symmetric_baseline`, `lambda1=lambda2=25`, seeds `5101-5105`, Q grid `[0, 4, 8, 12, 16, 20, 24, 28, 32]`.
Weight sets: w1=1, w2=1, w1=1.25, w2=1, w1=1.5, w2=1, w1=2, w2=1, w1=3, w2=1.

Gray vertical bands mark Q values where at least one class has served rate below 50%.
On the served-rate plot, black X markers are placed on the class line that is below 50%.
Dashed horizontal lines in the absolute plots are the pooled FCFS references for the same weight set.
The served-rate plot is not repeated by weight because weights do not change which patients are served.

## Figures
- `figures/weighted_old_utilization_by_weight.png`
- `figures/weighted_offered_wait_by_weight.png`
- `figures/weighted_old_utilization_delta_vs_fcfs_by_weight.png`
- `figures/weighted_wait_delta_vs_fcfs_by_weight.png`
- `figures/served_rate_drop_overall.png`

## Tables
- `tables/weight_sensitivity_by_seed.csv`
- `tables/weight_sensitivity_summary.csv`
