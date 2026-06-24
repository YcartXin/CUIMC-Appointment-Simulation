# H5 Stage 1 robustness test

## Hypothesis

At low-to-moderate Class 1 balking steps, higher post-threshold balking lowers
accepted booking delay mainly through selection rather than congestion relief.

## Files

Place or overwrite these files:

```text
experiments/robustness/h5_stage1.py
experiments/robustness/simulation_adapter.py
experiments/robustness/H5_README.md
tests/test_h5_stage1.py
```

The adapter update adds:

- Class 1 and Class 2 offered counts; and
- class-specific mean offered booking delays.

It only adds output fields and remains compatible with H1 through H4.

## Focal experiment

For each background, Class 1's pre-threshold balking rate remains fixed. The
within-class step is set to:

```text
0.0, 0.1, 0.3, 0.5
```

provided that:

```text
balk_low_class1 + step <= 0.70
```

The 0.10 and 0.30 steps are the primary low-to-moderate comparisons. The 0.50
step is diagnostic.

Each nonzero step is compared with the zero-step arm using the same seeds.

## Support criteria

For a target-level comparison:

1. Class 1 mean accepted delay falls by at least 0.25 days;
2. `delta_offered_delay - delta_accepted_delay` exceeds 0.25 days; and
3. Class 1 served rate falls by at least 0.005.

All three paired 95% confidence intervals must support the predicted direction.

A comparison is inactive when the estimated share of Class 1 offers reaching
the post-threshold region is below 1%.

## Run tests

```powershell
py -3 -m unittest tests.test_h5_stage1
```

## Smoke test

```powershell
py -3 -m experiments.robustness.h5_stage1 all --smoke --workers 1 --no-resume
```

## Full Stage 1 screen

```powershell
py -3 -m experiments.robustness.h5_stage1 all --workers 4
```

Do not include `--smoke` or `--no-resume` for the full run.

## Outputs

```text
outputs/robustness/h5/
├── design/
│   └── h5_background_scenarios.csv
├── raw/
│   └── h5_stage1_raw.csv
└── summary/
    ├── h5_step_effects.csv
    ├── h5_scenario_effects.csv
    ├── h5_classification_counts.csv
    ├── h5_step_classification_counts.csv
    ├── h5_failure_candidates.csv
    ├── h5_stage2_candidates.csv
    └── h5_stage1_summary.md
```
