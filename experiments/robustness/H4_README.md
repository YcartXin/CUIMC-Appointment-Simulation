# H4 Stage 1 robustness test

## Hypothesis

Under heavy oversubscription, moderate common post-threshold balking increases
mean offered delay, while higher balking eventually reduces it.

## Files

Place these files in the repository:

```text
experiments/robustness/h4_stage1.py
experiments/robustness/H4_README.md
tests/test_h4_stage1.py
```

The experiment uses the existing `simulation_adapter.py`. No adapter
replacement is required.

## Focal experiment

Both classes are assigned:

- a pre-threshold balking probability of `0.0`; and
- the same post-threshold balking probability from
  `{0.0, 0.1, 0.3, 0.5, 0.7}`.

Other scenario parameters remain fixed.

H4 is formally evaluated only for `rho >= 3.1`. Curves from lower-demand
backgrounds are still saved as descriptive diagnostics.

## Curve classification

A **hump** requires:

- the maximum mean offered delay to occur at an interior balking level; and
- the peak to exceed both endpoint levels by at least `0.25` days; and
- both paired 95% confidence intervals to exclude zero.

A **U-shaped** curve applies the same criteria to an interior minimum and is
classified as a reversal.

Other active high-demand curves are inconclusive at Stage 1.

## Run tests

```powershell
py -3 -m unittest tests.test_h4_stage1
```

## Smoke test

```powershell
py -3 -m experiments.robustness.h4_stage1 all --smoke --workers 1 --no-resume
```

## Full Stage 1 screen

```powershell
py -3 -m experiments.robustness.h4_stage1 all --workers 4
```

Do not include `--smoke` or `--no-resume` for the full run.

## Outputs

```text
outputs/robustness/h4/
├── design/
│   └── h4_background_scenarios.csv
├── raw/
│   └── h4_stage1_raw.csv
└── summary/
    ├── h4_scenario_effects.csv
    ├── h4_classification_counts.csv
    ├── h4_curve_shape_counts.csv
    ├── h4_failure_candidates.csv
    ├── h4_stage2_candidates.csv
    └── h4_stage1_summary.md
```
