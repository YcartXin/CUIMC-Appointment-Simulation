# H8 Stage 2 Confirmation Summary

Background scenarios classified: **1**

## Scenario classification counts

| classification   |   n_scenarios |
|:-----------------|--------------:|
| inconclusive     |             1 |

## Starting-cell classification counts

|   start_within_step |   start_post_gap |   inconclusive |
|--------------------:|-----------------:|---------------:|
|                 0.1 |              0.1 |              1 |

## Interpretation

- Every background uses three paired configurations: baseline, a 0.10 increase in Class 1's within-class balking step, and a 0.10 increase in the between-class post-threshold gap.
- Class 1's post-threshold balking probability is fixed at 0.50, and Class 2's pre-threshold probability is fixed at 0.00.
- Support requires the between-class gap change to have an absolute Class 1 served-rate effect at least 0.0025 larger than the within-class step change, with a paired 95% confidence interval above zero.
- A scenario is inactive when fewer than 1% of relevant offers reach the Class 1 pre-threshold, Class 1 post-threshold, or Class 2 post-threshold region.
- Active reversed and inconclusive backgrounds are retained as unresolved after Stage 2.
