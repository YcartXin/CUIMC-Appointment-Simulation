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

- `engine_files/model.py`: parameter objects, booking objects, metrics, and result containers
- `engine_files/engine.py`: simulation engine and event logic
- `engine_files/config_loader.py`: loads YAML scenario files into Python objects
- `analysis/`: shared metric and plotting-style helpers for studies
- `run_simulation.py`: runs one simulation scenario
- `compare_scenarios.py`: compares two simulation scenarios
- `configs/baseline.yaml`: baseline parameter file
- `configs/scenario_2.yaml`: comparison scenario file
- `docs/simulation_documentation.qmd`: simulation documentation source
- `docs/metric_analysis.qmd`: canonical metric-focused study source
- `docs/metric_analysis_files/`: generated figures, CSVs, and manifest for the metric report

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
