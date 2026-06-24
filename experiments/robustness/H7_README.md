# H7 Stage 1 robustness test

## Hypothesis

Holding Class 1's balking rates fixed, an equal between-class balking-rate
difference produces a larger served-rate gap when the difference is placed
below the threshold than when it is placed above the threshold.

## Files

Place these files in the repository:

```text
experiments/robustness/h7_stage1.py
experiments/robustness/H7_README.md
tests/test_h7_stage1.py
```

Keep the `simulation_adapter.py` installed for H6. No adapter replacement is
required.

## Focal experiment

For Class 1 rates `(b0, b1)` and gap magnitude `g`, compare:

```text
pre-threshold gap:  Class 2 = (b0 + g, b1)
post-threshold gap: Class 2 = (b0, b1 - g)
```

The Class 2 step equals `b1 - b0 - g` in both arms, so the comparison changes
only whether the between-class difference is below or above the threshold.

Gap levels are:

```text
0.05, 0.10, 0.20, 0.30, 0.50
```

A gap is valid when `g <= b1 - b0`.

## Classification

For each gap:

```text
D7 = |served-rate gap in pre arm| - |served-rate gap in post arm|
```

Support requires:

- `D7 >= 0.0025`; and
- the paired 95% confidence interval is above zero.

A reversal requires `D7 <= -0.0025` with the confidence interval below zero.

A comparison is exposure-active only when at least 1% of Class 2 offers occur
in each of the pre-threshold and post-threshold regimes.

The scenario classification uses the paired average `D7` across all
exposure-active valid gap magnitudes. At least two active gaps are required.

## Run tests

```powershell
py -3 -m unittest tests.test_h7_stage1
```

## Smoke test

```powershell
py -3 -m experiments.robustness.h7_stage1 all --smoke --workers 1 --no-resume
```

## Full Stage 1 screen

```powershell
py -3 -m experiments.robustness.h7_stage1 all --workers 4
```

Do not include `--smoke` or `--no-resume` for the full run.

## Outputs

```text
outputs/robustness/h7/
├── design/
│   └── h7_background_scenarios.csv
├── raw/
│   └── h7_stage1_raw.csv
└── summary/
    ├── h7_gap_effects.csv
    ├── h7_scenario_effects.csv
    ├── h7_classification_counts.csv
    ├── h7_gap_classification_counts.csv
    ├── h7_failure_candidates.csv
    ├── h7_stage2_candidates.csv
    └── h7_stage1_summary.md
```
