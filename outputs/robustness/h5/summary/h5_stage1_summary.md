# H5 Stage 1 Robustness Summary

Background scenarios classified: **381**
Low-to-moderate step comparisons classified: **677**

## Scenario classification counts

| classification   |   n_scenarios |
|:-----------------|--------------:|
| inactive         |           219 |
| inconclusive     |           117 |
| reversed         |            17 |
| supported        |            28 |

## Step-level classification counts

|   balk_step_class1_focal |   inactive |   inconclusive |   reversed |   supported |
|-------------------------:|-----------:|---------------:|-----------:|------------:|
|                      0.1 |        201 |            154 |          8 |           0 |
|                      0.3 |        168 |            101 |         17 |          28 |
|                      0.5 |        122 |             55 |         15 |          49 |

## Interpretation

- Primary inference uses Class 1 balking-step increases of 0.10 and 0.30; the 0.50 step is retained as a diagnostic.
- Support requires accepted delay to fall by at least 0.25 days, the offered-minus-accepted delay contrast to exceed 0.25 days, and Class 1 served rate to fall by at least 0.005.
- A comparison is inactive when the estimated share of Class 1 offers in the post-threshold region is below 1%.
- Supported comparisons where offered delay does not materially fall are flagged as especially strong evidence of selection rather than congestion relief.
- Active reversed and inconclusive backgrounds are exported for Stage 2 confirmation.
