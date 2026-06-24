# H7 Stage 1 Robustness Summary

Background scenarios classified: **370**
Gap-location comparisons classified: **1015**

## Scenario classification counts

| classification   |   n_scenarios |
|:-----------------|--------------:|
| inactive         |           268 |
| inconclusive     |             9 |
| reversed         |            29 |
| supported        |            64 |

## Gap-level classification counts

|   gap_magnitude_focal |   inactive |   inconclusive |   reversed |   supported |
|----------------------:|-----------:|---------------:|-----------:|------------:|
|                  0.05 |        160 |             24 |         25 |          52 |
|                  0.1  |        159 |             10 |         32 |          60 |
|                  0.2  |        143 |              7 |         29 |          56 |
|                  0.3  |         93 |              5 |         14 |          52 |
|                  0.5  |         50 |              0 |         10 |          34 |

## Interpretation

- For each gap magnitude, the Class 2 within-class balking step is identical in the pre-gap and post-gap arms; only the location of the between-class difference changes.
- Support requires the absolute served-rate gap to be at least 0.0025 larger in the pre-threshold-gap arm, with a paired 95% confidence interval above zero.
- Scenario classification uses the paired average effect across all exposure-active valid gap magnitudes.
- A gap comparison requires at least 1% of Class 2 offers in both the pre-threshold and post-threshold regimes.
- Active reversed and inconclusive backgrounds are exported for Stage 2 confirmation.
