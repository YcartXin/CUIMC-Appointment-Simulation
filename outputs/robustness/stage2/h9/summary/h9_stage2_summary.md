# H9 Stage 2 Confirmation Summary

Background scenarios classified: **6**

## Scenario classification counts

| classification   |   n_scenarios |
|:-----------------|--------------:|
| reversed         |             6 |

## Active component combinations

| utilization_component   | served_gap_component   |   n_scenarios |
|:------------------------|:-----------------------|--------------:|
| supported               | reversed               |             6 |

## Classification by equal baseline probability

|   baseline_equal_noshow_high |   reversed |
|-----------------------------:|-----------:|
|                          0.1 |          1 |
|                          0.3 |          1 |
|                          0.5 |          2 |
|                          0.7 |          1 |
|                          0.8 |          1 |

## Interpretation

- The common arm raises both post-threshold no-show probabilities by 0.10.
- The two gap arms increase the between-class difference by 0.10 while preserving the average probability; both orientations are averaged.
- Support requires the common change to have an aggregate utilization effect at least 0.0025 larger than the gap change.
- Support also requires the gap change to have a served-rate-gap effect at least 0.0025 larger than the common change.
- Both paired 95% confidence intervals must be above zero.
- Active reversed and inconclusive backgrounds are retained as unresolved after Stage 2.
