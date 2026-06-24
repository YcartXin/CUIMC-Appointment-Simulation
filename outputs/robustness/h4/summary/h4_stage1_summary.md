# H4 Stage 1 Robustness Summary

Background scenarios classified: **360**
Heavy-oversubscription backgrounds classified: **151**

## Classification counts by demand regime

| demand_regime   |   inactive |   inconclusive |   supported |
|:----------------|-----------:|---------------:|------------:|
| boundary        |         68 |              0 |           0 |
| high            |         25 |            100 |          26 |
| low             |        141 |              0 |           0 |

## Curve-shape counts by demand regime

| demand_regime   | curve_shape   |   n_scenarios |
|:----------------|:--------------|--------------:|
| boundary        | decreasing    |            34 |
| boundary        | flat          |            33 |
| boundary        | irregular     |             1 |
| high            | decreasing    |            57 |
| high            | flat          |            46 |
| high            | hump          |            26 |
| high            | increasing    |             6 |
| high            | irregular     |            16 |
| low             | decreasing    |            23 |
| low             | flat          |           118 |

## Interpretation

- H4 is inferentially evaluated only in heavy-oversubscription backgrounds; lower-demand curves are retained as diagnostics.
- Support requires a statistically reliable interior maximum in mean offered delay, at least 0.25 days above both endpoint levels.
- A statistically reliable interior minimum is classified as a reversal because it is the opposite non-monotone pattern.
- Active high-demand flat, monotone, and irregular curves are inconclusive and exported for Stage 2 confirmation.
