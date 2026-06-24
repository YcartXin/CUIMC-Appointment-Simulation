# H6 Stage 1 robustness test

## Hypothesis

Changing Class 1's balking threshold can produce nonlinear served-rate effects
because a one-day threshold change reclassifies an entire offered-delay bucket.

## Files

Place or overwrite:

```text
experiments/robustness/h6_stage1.py
experiments/robustness/simulation_adapter.py
experiments/robustness/H6_README.md
tests/test_h6_stage1.py
```

The adapter update adds a separate instrumented simulation function that
records offered-delay counts. The normal `run_scenario()` function is
unchanged in behavior and remains compatible with H1 through H5.

No changes to `simulation/model.py` or `simulation/engine.py` are required.

## Focal experiment

For each background, Class 1's balking threshold is swept through every valid
integer:

```text
0, 1, ..., horizon_class1 - 2
```

For the adjacent transition:

```text
tau -> tau + 1
```

the delay bucket at `tau + 1` changes from the post-threshold regime to the
pre-threshold regime.

The script records:

- the mass of that bucket among all Class 1 offers; and
- the absolute adjacent change in Class 1 served rate.

## Support criteria

Support requires:

1. Spearman correlation between bucket mass and served-rate jump is at least
   `0.50`;
2. the largest jump occurs in the upper half of the bucket-mass distribution;
3. the largest jump is at least `0.005`; and
4. its paired 95% confidence interval excludes zero.

A scenario is inactive when:

- Class 1 has no balking step (`b1 = b0`);
- fewer than three adjacent transitions are usable; or
- every reclassified bucket has less than 1% of Class 1 offers.

## Run tests

```powershell
py -3 -m unittest tests.test_h6_stage1
```

## Smoke test

```powershell
py -3 -m experiments.robustness.h6_stage1 all --smoke --workers 1 --no-resume
```

## Full Stage 1 screen

```powershell
py -3 -m experiments.robustness.h6_stage1 all --workers 4
```

H6 is more computationally intensive than earlier hypotheses because it uses a
dense threshold sweep. The run is resumable; rerun the same full command after
an interruption.

## Outputs

```text
outputs/robustness/h6/
├── design/
│   └── h6_background_scenarios.csv
├── raw/
│   └── h6_stage1_raw.csv
└── summary/
    ├── h6_transition_effects.csv
    ├── h6_scenario_effects.csv
    ├── h6_classification_counts.csv
    ├── h6_failure_candidates.csv
    ├── h6_stage2_candidates.csv
    └── h6_stage1_summary.md
```
