# H2 Stage 1 Robustness Summary

Background scenarios classified: **360**
Target-level comparisons classified: **1080**

## Scenario classification counts

| classification   |   n_scenarios |
|:-----------------|--------------:|
| inactive         |             1 |
| inconclusive     |           140 |
| reversed         |             1 |
| supported        |           218 |

## Target-level classification counts

|   target_loss_share |   inactive |   inconclusive |   reversed |   supported |
|--------------------:|-----------:|---------------:|-----------:|------------:|
|                0.05 |          1 |            169 |          0 |         190 |
|                0.1  |          2 |            130 |          2 |         226 |
|                0.2  |         17 |            113 |          0 |         230 |

## Matching diagnostics

- Target comparisons meeting the realized-loss matching rule: **1015/1080**.
- Matching requires a between-arm loss-share gap no greater than 0.01 and each arm to be within 0.02 of its target.

## Interpretation

- Support requires utilization to be materially higher in the balking arm than in the no-show arm.
- Class 1 served rate must also fall materially in both arms relative to the zero-focal-loss baseline.
- Matched reversed and inconclusive target comparisons are exported for Stage 2 confirmation.
