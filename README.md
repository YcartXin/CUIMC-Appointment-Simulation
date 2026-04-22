# CUIMC-Appointment-Simulation

This repository contains a discrete-event simulation of a clinic appointment booking system. The model represents slot-by-slot scheduling under a first-come-first-served booking rule with two patient classes.

The simulation includes:

- class-specific Poisson arrivals
- delay-dependent balking
- delay-dependent no-show
- end-of-day cancellations
- a rolling booking calendar
- burn-in, measurement, and cooldown periods

## Repository structure

- `model.py`: parameter objects, booking objects, metrics, and result containers
- `engine.py`: simulation engine and event logic
- `config_loader.py`: loads YAML scenario files into Python objects
- `run_simulation.py`: runs one simulation scenario
- `compare_scenarios.py`: compares two simulation scenarios
- `configs/baseline.yaml`: baseline parameter file
- `configs/scenario_2.yaml`: comparison scenario file
- `docs/simulation_documentation.qmd`: short documentation of model logic and setup
- `docs/simulation_documentation.html`: rendered documentation output

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt