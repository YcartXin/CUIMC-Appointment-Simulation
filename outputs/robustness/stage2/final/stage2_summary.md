# Stage 2 Robustness Confirmation Summary

Stage 2 reruns every Stage 1 reversal and only those inconclusive cases whose point estimates already meet all practical-effect criteria but whose confidence intervals are not decisive.

## Final active classifications

| hypothesis   |   inconclusive |   reversed |   supported |
|:-------------|---------------:|-----------:|------------:|
| H1           |            135 |         36 |         167 |
| H2           |            140 |          1 |         218 |
| H3           |             14 |          0 |          87 |
| H4           |            100 |          0 |          26 |
| H5           |            118 |         16 |          28 |
| H6           |             76 |          4 |         134 |
| H7           |              9 |         29 |          64 |
| H8           |             50 |         60 |          13 |
| H9           |             15 |          5 |         102 |

## Stage 2 transitions

| hypothesis   | transition                   |   n_backgrounds |
|:-------------|:-----------------------------|----------------:|
| H1           | not_rerun                    |              36 |
| H1           | reversed_to_inconclusive     |               1 |
| H2           | inconclusive_to_inconclusive |               1 |
| H2           | not_rerun                    |               5 |
| H5           | not_rerun                    |              16 |
| H5           | reversed_to_inconclusive     |               1 |
| H6           | not_rerun                    |               4 |
| H6           | reversed_to_inconclusive     |               1 |
| H7           | not_rerun                    |              29 |
| H7           | reversed_to_reversed         |               1 |
| H8           | not_rerun                    |              66 |
| H8           | reversed_to_inconclusive     |               1 |
| H9           | not_rerun                    |               5 |
| H9           | reversed_to_inconclusive     |               1 |

Inactive configurations excluded from the substantive table: **1699**.

Individual scenarios are retained in the CSV outputs but are not listed one by one in this summary.
