# CUIMC-Appointment-Simulation

This repository contains a discrete-event simulation of a clinic appointment booking system. The model is now a **day-level** appointment model with first-come-first-served booking, two patient classes, and a rolling booking horizon.

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

- `model.py`: parameter objects, booking objects, metrics, and result containers
- `engine.py`: simulation engine and event logic
- `config_loader.py`: loads YAML scenario files into Python objects
- `run_simulation.py`: runs one simulation scenario
- `compare_scenarios.py`: compares two simulation scenarios
- `configs/baseline.yaml`: baseline parameter file
- `configs/scenario_2.yaml`: comparison scenario file
- `simulation_documentation.qmd`: simulation documentation source
- `simulation_documentation.html`: rendered documentation output

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt