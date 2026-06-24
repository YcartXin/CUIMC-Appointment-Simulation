# H3 Stage 1 robustness test

## Hypothesis

Increasing Class 1's post-threshold no-show probability reduces aggregate
utilization more when the no-show threshold is lower.

## Files

Place these files in the repository:

```text
experiments/robustness/h3_stage1.py
experiments/robustness/H3_README.md
tests/test_h3_stage1.py
```

This experiment uses the `simulation_adapter.py` already installed for H2.
No adapter replacement is required.

## Design

For each H3 background:

- Class 1's pre-threshold no-show probability remains fixed.
- The low arm sets the post-threshold probability equal to the pre-threshold
  probability.
- The high arm sets the post-threshold probability to `0.70`.
- The comparison is repeated at each valid threshold in `{4, 6, 9, 12}`.
- A threshold is valid only when `threshold < horizon_class1 - 1`.

A background is design-inactive when:

- fewer than two proposed thresholds are valid; or
- the pre-threshold no-show probability is already `0.70`, leaving no
  post-threshold increase.

After simulation, a threshold is treated as exposed only when the realized
increase in Class 1 no-shows is at least `0.005` per Class 1 arrival.

## Classification

Support requires:

1. increasing the post-threshold probability materially lowers utilization at
   the lowest exposed threshold;
2. the utilization loss is at least `0.005` greater at the lowest exposed
   threshold than at the highest exposed threshold, with its paired confidence
   interval above zero; and
3. Spearman correlation between threshold and utilization-loss magnitude is at
   most `-0.50`.

## Run tests

From the repository root:

```powershell
py -3 -m unittest tests.test_h3_stage1
```

## Smoke test

```powershell
py -3 -m experiments.robustness.h3_stage1 all --smoke --workers 1 --no-resume
```

## Full Stage 1 screen

```powershell
py -3 -m experiments.robustness.h3_stage1 all --workers 4
```

Do not include `--smoke` or `--no-resume` for the full run.

## Outputs

```text
outputs/robustness/h3/
├── design/
│   └── h3_background_scenarios.csv
├── raw/
│   └── h3_stage1_raw.csv
└── summary/
    ├── h3_threshold_effects.csv
    ├── h3_scenario_effects.csv
    ├── h3_classification_counts.csv
    ├── h3_failure_candidates.csv
    ├── h3_stage2_candidates.csv
    └── h3_stage1_summary.md
```
