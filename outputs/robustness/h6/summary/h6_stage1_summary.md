# H6 Stage 1 Robustness Summary

Background scenarios classified: **3**
Adjacent threshold transitions evaluated: **24**

## Scenario classification counts

| classification   |   n_scenarios |
|:-----------------|--------------:|
| inactive         |             1 |
| inconclusive     |             1 |
| supported        |             1 |

## Interpretation

- Each transition from threshold tau to tau + 1 reclassifies the offered-delay bucket at tau + 1.
- Support requires Spearman correlation of at least 0.50 between reclassified bucket mass and absolute served-rate jump.
- The largest served-rate jump must occur in the upper half of the bucket-mass distribution and exceed 0.005 with a paired confidence interval excluding zero.
- A scenario is inactive when there is no within-class balking step, fewer than three usable transitions, or all reclassified bucket masses are below 1%.
- Active reversed and inconclusive backgrounds are exported for Stage 2 confirmation.
