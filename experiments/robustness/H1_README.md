# H1 Stage 1 robustness experiment

This pilot validates the full robustness workflow before the remaining eight
hypotheses are implemented.

## Files

- `simulation_adapter.py`: converts scenario CSV rows to `SimulationConfig` and runs the model.
- `h1_stage1.py`: prepares H1 backgrounds, runs the four cancellation levels with paired seeds, classifies scenarios, and exports Stage 2 candidates.
- `tests/test_h1_stage1.py`: tests H1 background deduplication and classification logic.

## Run from the repository root

First run a small smoke test:

```bash
python experiments/robustness/h1_stage1.py all --smoke --workers 1 --no-resume
```

The smoke test runs 2 backgrounds × 4 focal levels × 2 seeds = 16 simulations.

Then run the full Stage 1 H1 experiment:

```bash
python experiments/robustness/h1_stage1.py all
```

To stop and resume later, rerun the same command. Existing completed
`background_id × focal level × seed` combinations are skipped.

To rerun from scratch:

```bash
python experiments/robustness/h1_stage1.py all --no-resume
```

## Outputs

```text
outputs/robustness/h1/
├── design/
│   └── h1_background_scenarios.csv
├── raw/
│   └── h1_stage1_raw.csv
└── summary/
    ├── h1_scenario_effects.csv
    ├── h1_classification_counts.csv
    ├── h1_failure_candidates.csv
    ├── h1_stage2_candidates.csv
    └── h1_stage1_summary.md
```

`h1_stage2_candidates.csv` contains provisional reversals and high-demand or
boundary inconclusive cases to rerun with the independent 100-seed Stage 2 set.
