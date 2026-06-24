# H9 Stage 1 Robustness Summary

Background scenarios classified: **375**

## Scenario classification counts

| classification   |   n_scenarios |
|:-----------------|--------------:|
| inactive         |           253 |
| inconclusive     |            14 |
| reversed         |             6 |
| supported        |           102 |

## Active component combinations

| utilization_component   | served_gap_component   |   n_scenarios |
|:------------------------|:-----------------------|--------------:|
| inconclusive            | inconclusive           |             1 |
| supported               | inconclusive           |            13 |
| supported               | reversed               |             6 |
| supported               | supported              |           102 |

## Classification by equal baseline probability

|   baseline_equal_noshow_high |   inactive |   inconclusive |   reversed |   supported |
|-----------------------------:|-----------:|---------------:|-----------:|------------:|
|                          0.1 |         25 |              1 |          1 |          16 |
|                          0.3 |         17 |              2 |          1 |          11 |
|                          0.5 |         53 |              4 |          2 |          13 |
|                          0.7 |         77 |              1 |          1 |          26 |
|                          0.8 |         81 |              6 |          1 |          36 |

## Interpretation

- The common arm raises both post-threshold no-show probabilities by 0.10.
- The two gap arms increase the between-class difference by 0.10 while preserving the average probability; both orientations are averaged.
- Support requires the common change to have an aggregate utilization effect at least 0.0025 larger than the gap change.
- Support also requires the gap change to have a served-rate-gap effect at least 0.0025 larger than the common change.
- Both paired 95% confidence intervals must be above zero.
- Active reversed and inconclusive backgrounds are exported for Stage 2 confirmation.
