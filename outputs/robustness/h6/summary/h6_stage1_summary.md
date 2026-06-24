# H6 Stage 1 Robustness Summary

Background scenarios classified: **378**
Adjacent threshold transitions evaluated: **4237**

## Scenario classification counts

| classification   |   n_scenarios |
|:-----------------|--------------:|
| inactive         |           164 |
| inconclusive     |            75 |
| reversed         |             5 |
| supported        |           134 |

## Interpretation

- Each transition from threshold tau to tau + 1 reclassifies the offered-delay bucket at tau + 1.
- Support requires Spearman correlation of at least 0.50 between reclassified bucket mass and absolute served-rate jump.
- The largest served-rate jump must occur in the upper half of the bucket-mass distribution and exceed 0.005 with a paired confidence interval excluding zero.
- A scenario is inactive when there is no within-class balking step, fewer than three usable transitions, or all reclassified bucket masses are below 1%.
- Active reversed and inconclusive backgrounds are exported for Stage 2 confirmation.
