# Stage 1 Scenario Generation Summary

## Counts

- Deterministic anchors: 32
- Sobol symmetric scenarios: 224
- Total symmetric scenarios: 256
- Asymmetric stress scenarios: 128
- Total Stage 1 background scenarios: 384
- Stage 1 paired seeds: 20
- Stage 2 confirmation seeds: 100

## Design constraints

- `balk_high >= balk_low`
- `noshow_high >= noshow_low`
- `threshold < horizon - 1` for both balking and no-show rules
- `lambda_total = rho * slots_per_day`
- `lambda_class1 = class1_share * lambda_total`
- `lambda_class2 = (1 - class1_share) * lambda_total`

## Scenario types

| Value | Count |
|---:|---:|
| anchor | 32 |
| asymmetric_stress | 128 |
| sobol_symmetric | 224 |

## Demand-to-capacity ratio

| Value | Count |
|---:|---:|
| 0.8 | 75 |
| 1.25 | 77 |
| 2.0 | 70 |
| 3.1 | 84 |
| 4.0 | 78 |

## Class 1 arrival share

| Value | Count |
|---:|---:|
| 0.1 | 73 |
| 0.3 | 76 |
| 0.5 | 87 |
| 0.7 | 73 |
| 0.9 | 75 |

## Daily capacity

| Value | Count |
|---:|---:|
| 16 | 124 |
| 32 | 147 |
| 64 | 113 |

## Booking horizon: Class 1

| Value | Count |
|---:|---:|
| 7 | 73 |
| 14 | 125 |
| 21 | 87 |
| 28 | 99 |

## Validation

- Passed checks: 20
- Failed checks: 0

All validation checks passed.
