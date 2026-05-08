# CUIMC-Appointment-Simulation

This repository contains a discrete-event simulation of a clinic appointment booking system. The model is now a day-level appointment model with first-come-first-served booking, two patient classes, and a rolling booking horizon.

The simulation includes:

- class-specific daily Poisson arrivals
- daily random permutation of arriving patients
- same-day booking allowed
- delay-dependent balking
- delay-dependent no-show
- start-of-day cancellations for future appointments only
- no same-day cancellations
- no rebooking of no-show slots
- a rolling day-level booking calendar
- burn-in, measurement, and cooldown periods

## Repository structure

- `simulation/model.py`: parameter objects, booking objects, metrics, and result containers
- `simulation/engine.py`: simulation engine and event logic
- `simulation/config_loader.py`: loads YAML scenario files into Python objects
- `analysis/`: shared metric and plotting-style helpers for studies
- `scripts/run_simulation.py`: runs one simulation scenario
- `scripts/compare_scenarios.py`: compares two simulation scenarios
- `scripts/generate_metric_analysis_figures.py`: builds metric report figures and manifest
- `experiments/`: parameter sweeps used by reports
- `configs/baseline.yaml`: baseline parameter file
- `configs/scenario_2.yaml`: comparison scenario file
- `notebooks/`: interactive exploration notebooks
- `docs/`: documentation sources and archived notes
- `docs/reference/simulation_documentation.qmd`: simulation documentation source
- `docs/reports/metric_analysis.qmd`: canonical metric-focused study source
- `outputs/`: generated figures, CSVs, manifests, rendered reports, and sweep outputs
- `outputs/reports/metric_analysis/`: metric report figures, data, manifest, and rendered report artifacts

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Reproducible study workflow

Generate all study figures and the metric-analysis manifest:

```bash
make figures
```

Render the canonical metric report when Quarto is installed:

```bash
make report
```

Run the test suite:

```bash
make check
```

Run one-off scenarios:

```bash
python scripts/run_simulation.py
python scripts/compare_scenarios.py
```
