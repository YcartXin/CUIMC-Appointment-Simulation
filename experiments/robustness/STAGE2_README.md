# Stage 2 robustness confirmation

## Purpose

Stage 2 uses 100 new paired seeds (`2000` through `2099`) to confirm the cases
that genuinely need additional simulation.

The selector includes:

- every Stage 1 `reversed` background; and
- an `inconclusive` background only when its point estimates already satisfy
  all practical-effect and directional criteria, but one or more confidence
  intervals are not decisive.

It does **not** rerun inactive backgrounds or clearly sub-threshold
inconclusive results. For H4, it reruns only uncertain hump candidates, not
curves that are already clearly flat, decreasing, increasing, or irregular.

## Files

Place these files in the repository:

```text
experiments/robustness/stage2.py
experiments/robustness/STAGE2_README.md
tests/test_stage2.py
```

Keep all H1-H9 Stage 1 scripts, the current `simulation_adapter.py`, and the
existing Stage 1 output directories.

## Test

From the repository root:

```powershell
py -3 -m unittest tests.test_stage2
```

## Select candidates

```powershell
py -3 -m experiments.robustness.stage2 select
```

Review:

```text
outputs/robustness/stage2/design/stage2_candidate_counts.csv
outputs/robustness/stage2/design/stage2_candidates.csv
```

The combined CSV retains individual candidate identifiers for reproducibility,
but the generated Markdown summary does not list cases one by one.

## Smoke test

The smoke test runs the first selected background for each hypothesis with two
new seeds:

```powershell
py -3 -m experiments.robustness.stage2 all --smoke --workers 1 --no-resume
```

This writes to the normal Stage 2 output directory. A later full run will retain
those two completed seeds and run the remaining seeds and backgrounds.

## Full Stage 2 run

```powershell
py -3 -m experiments.robustness.stage2 all --workers 4
```

The full run is resumable. After an interruption, rerun the same command.
Do not use `--no-resume` unless you intend to discard and recompute the Stage 2
raw results.

## Run selected hypotheses only

Examples:

```powershell
py -3 -m experiments.robustness.stage2 all --hypotheses h1,h2,h3 --workers 4
py -3 -m experiments.robustness.stage2 run --hypotheses h7,h8 --workers 4
py -3 -m experiments.robustness.stage2 classify --hypotheses h7,h8
py -3 -m experiments.robustness.stage2 summarize
```

## Commands

```text
select      Build the Stage 2 candidate lists.
run         Run 100 new paired seeds for selected backgrounds.
classify    Apply the original hypothesis-specific classification rules.
summarize   Merge Stage 1 and Stage 2 classifications.
all         Run select, run, classify, and summarize in sequence.
```

## Outputs

```text
outputs/robustness/stage2/
├── design/
│   ├── stage2_candidates.csv
│   ├── stage2_candidate_counts.csv
│   └── h1_stage2_candidates.csv ... h9_stage2_candidates.csv
├── h1/ ... h9/
│   ├── design/
│   ├── raw/
│   │   └── hN_stage2_raw.csv
│   └── summary/
│       ├── hN_scenario_effects.csv
│       ├── hN_stage2_summary.md
│       └── hN_unresolved_after_stage2.csv
└── final/
    ├── stage2_confirmation_results.csv
    ├── stage2_final_status.csv
    ├── confirmed_reversals.csv
    ├── unresolved_after_stage2.csv
    ├── final_active_classification_counts.csv
    ├── stage2_transition_counts.csv
    └── stage2_summary.md
```

`stage2_final_status.csv` uses the Stage 2 classification for every rerun
background and otherwise retains the Stage 1 classification. Inactive
backgrounds remain in the reproducibility CSV but are excluded from the active
summary table.
