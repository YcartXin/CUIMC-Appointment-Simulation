# H9 Stage 1 robustness test

## Hypothesis

A common increase in both classes' post-threshold no-show probabilities has a
larger effect on aggregate utilization, while increasing the between-class
difference at a fixed average has a larger effect on the served-rate gap.

## Files

Place these files in the repository:

```text
experiments/robustness/h9_stage1.py
experiments/robustness/H9_README.md
tests/test_h9_stage1.py
```

Keep the current `simulation_adapter.py`. No adapter replacement is required.

## Focal experiment

For an assigned equal baseline probability `p`, run four paired arms:

```text
baseline:      (p, p)
common_up:     (p + 0.10, p + 0.10)
gap_c1_higher: (p + 0.05, p - 0.05)
gap_c2_higher: (p - 0.05, p + 0.05)
```

The two gap orientations preserve the average `p` and increase the
between-class difference by `0.10`.

The runner assigns a feasible `p` from:

```text
0.10, 0.30, 0.50, 0.70, 0.80
```

subject to all high probabilities remaining at least as large as their
class-specific pre-threshold probabilities.

## Classification

For each seed:

```text
utilization component =
    abs(common utilization change)
    - average abs(gap-orientation utilization changes)

served-gap component =
    average abs(gap-orientation served-gap changes)
    - abs(common served-gap change)
```

Support requires both components to be at least `0.0025`, with both paired 95%
confidence intervals above zero.

A scenario is reversed when either component is materially and precisely
negative. Partial support is inconclusive.

## Exposure rule

The common arm raises each post-threshold no-show probability by `0.10`.
The script estimates the post-threshold accepted-booking share from the paired
increase in no-show counts:

```text
additional no-shows / (0.10 * average booked count)
```

Both classes require at least 1% estimated post-threshold accepted exposure.
Each class must also have at least 100 arrivals over the measured simulation.

## Run tests

```powershell
py -3 -m unittest tests.test_h9_stage1
```

## Smoke test

```powershell
py -3 -m experiments.robustness.h9_stage1 all --smoke --workers 1 --no-resume
```

## Full Stage 1 screen

```powershell
py -3 -m experiments.robustness.h9_stage1 all --workers 4
```

Do not include `--smoke` or `--no-resume` for the full run.

## Outputs

```text
outputs/robustness/h9/
├── design/
│   └── h9_background_scenarios.csv
├── raw/
│   └── h9_stage1_raw.csv
└── summary/
    ├── h9_scenario_effects.csv
    ├── h9_classification_counts.csv
    ├── h9_component_classification_counts.csv
    ├── h9_baseline_probability_counts.csv
    ├── h9_failure_candidates.csv
    ├── h9_stage2_candidates.csv
    └── h9_stage1_summary.md
```
