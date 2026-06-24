# H8 Stage 1 robustness test

## Hypothesis

Holding Class 1's post-threshold balking probability fixed, an equal increase
in the between-class post-threshold balking gap has a larger absolute effect on
Class 1 served rate than an equal increase in Class 1's within-class balking
step.

## Files

Place these files in the repository:

```text
experiments/robustness/h8_stage1.py
experiments/robustness/H8_README.md
tests/test_h8_stage1.py
```

Keep the H6/H7 version of:

```text
experiments/robustness/simulation_adapter.py
```

No adapter replacement is required.

## Focal design

Class 1's post-threshold probability is fixed at:

```text
b11 = 0.50
```

Class 2's pre-threshold probability is fixed at:

```text
b02 = 0.00
```

Each background is assigned one balanced starting cell:

```text
S1 = b11 - b01
G1 = b11 - b12
```

where `S1` and `G1` each cycle through:

```text
0.0, 0.1, 0.2, 0.3, 0.4
```

Three paired arms are run:

```text
baseline
step_up: S1 increases by 0.10
gap_up:  G1 increases by 0.10
```

## Classification

For each seed:

```text
delta_step = R1(step_up) - R1(baseline)
delta_gap  = R1(gap_up)  - R1(baseline)

D8 = abs(delta_gap) - abs(delta_step)
```

Support requires:

- mean `D8 >= 0.0025`; and
- the paired 95% confidence interval is above zero.

A reversal requires:

- mean `D8 <= -0.0025`; and
- the confidence interval is below zero.

A scenario is exposure-active only when at least 1% of relevant offers appear
in:

- Class 1's pre-threshold region;
- Class 1's post-threshold region; and
- Class 2's post-threshold region.

## Run tests

```powershell
py -3 -m unittest tests.test_h8_stage1
```

## Smoke test

```powershell
py -3 -m experiments.robustness.h8_stage1 all --smoke --workers 1 --no-resume
```

## Full Stage 1 screen

```powershell
py -3 -m experiments.robustness.h8_stage1 all --workers 4
```

Do not include `--smoke` or `--no-resume` for the full run.

## Outputs

```text
outputs/robustness/h8/
├── design/
│   └── h8_background_scenarios.csv
├── raw/
│   └── h8_stage1_raw.csv
└── summary/
    ├── h8_scenario_effects.csv
    ├── h8_classification_counts.csv
    ├── h8_start_cell_classification_counts.csv
    ├── h8_failure_candidates.csv
    ├── h8_stage2_candidates.csv
    └── h8_stage1_summary.md
```
