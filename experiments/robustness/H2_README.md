# H2 Stage 1 robustness experiment

H2 compares losses occurring at two different points in the booking process:

- **Balking arm:** Class 1 loses patients when they reject an offer, before a slot is occupied.
- **No-show arm:** Class 1 loses patients after accepting and occupying a slot.

The two arms are calibrated to approximately equal realized loss shares among
Class 1 arrivals at targets of 5%, 10%, and 20%.

## Files

Place these files in the repository:

```text
experiments/robustness/h2_stage1.py
experiments/robustness/simulation_adapter.py
tests/test_h2_stage1.py
```

The included `simulation_adapter.py` replaces the existing version. It adds
class-level balking and no-show counts and rates required by H2; it remains
compatible with H1.

## Smoke test

From the repository root:

```powershell
py -3 -m experiments.robustness.h2_stage1 all --smoke --workers 1 --no-resume
```

The smoke test uses two backgrounds, one calibration seed, and two Stage 1
seeds. It verifies calibration, simulation, matching, and classification.

## Full Stage 1 run

```powershell
py -3 -m experiments.robustness.h2_stage1 all --workers 4
```

The full run first calibrates focal probabilities using seeds 900-902, then
runs the matched arms using the 20 Stage 1 seeds. The command is resumable.

## Separate commands

```powershell
py -3 -m experiments.robustness.h2_stage1 calibrate --workers 4
py -3 -m experiments.robustness.h2_stage1 run --workers 4
py -3 -m experiments.robustness.h2_stage1 classify
```

## Outputs

```text
outputs/robustness/h2/
├── design/
│   └── h2_background_scenarios.csv
├── calibration/
│   └── h2_loss_calibration.csv
├── raw/
│   └── h2_stage1_raw.csv
└── summary/
    ├── h2_target_effects.csv
    ├── h2_scenario_effects.csv
    ├── h2_classification_counts.csv
    ├── h2_failure_candidates.csv
    ├── h2_stage2_candidates.csv
    └── h2_stage1_summary.md
```

A target-level comparison is considered matched when the realized loss-share
gap between arms is no more than 0.01 and each arm is within 0.02 of the target.
Scenario-level support requires at least two supported target levels and no
reversal.
