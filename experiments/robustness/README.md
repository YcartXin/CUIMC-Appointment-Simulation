# Robustness Scenario Design

This folder contains the Stage 1 background-scenario generator for the nine-hypothesis robustness study.

## Repository placement

Copy this folder to:

```text
experiments/robustness/
```

Copy the accompanying test file to:

```text
tests/test_robustness_scenario_design.py
```

## Dependency

The generator uses `scipy.stats.qmc.Sobol`. Install the local requirement:

```bash
pip install -r experiments/robustness/requirements.txt
```

You should also add `scipy` to the root `requirements.txt` when you next update it.

## Run

From the repository root:

```bash
python experiments/robustness/scenario_design.py
```

Optional custom output directory:

```bash
python experiments/robustness/scenario_design.py \
  --output-dir outputs/robustness/scenarios
```

## Generated files

```text
outputs/robustness/scenarios/
├── symmetric_scenarios.csv
├── asymmetric_scenarios.csv
├── all_stage1_scenarios.csv
├── stage1_seeds.csv
├── stage2_seeds.csv
├── scenario_validation.csv
└── scenario_generation_summary.md
```

The script creates 384 background scenarios:

- 32 deterministic anchors;
- 224 scrambled-Sobol symmetric scenarios; and
- 128 sparse asymmetric stress scenarios.

No clinic simulations are run at this stage.

## Enforced constraints

For each class, the generator enforces:

```text
post-threshold balking probability >= pre-threshold balking probability
post-threshold no-show probability >= pre-threshold no-show probability
threshold < horizon - 1
```

Arrival rates are calculated from the selected congestion and class-share settings:

```text
lambda_total = rho * slots_per_day
lambda_class1 = class1_share * lambda_total
lambda_class2 = (1 - class1_share) * lambda_total
```

## Test

From the repository root:

```bash
python -m unittest tests.test_robustness_scenario_design
```
