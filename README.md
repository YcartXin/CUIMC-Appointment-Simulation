# CUIMC-Appointment-Simulation

This repository contains a discrete-event clinic appointment simulation with:

- first-come-first-served booking
- two patient classes
- delay-dependent balking
- delay-dependent no-show
- end-of-day cancellations
- a rolling appointment calendar
- burn-in, measurement, and cooldown periods

## Repository structure

- `FCFS Model Simulation.ipynb`: working notebook version
- `model.py`: model objects, parameters, metrics, and result containers
- `engine.py`: simulation engine
- `config_loader.py`: loads YAML configuration into Python objects
- `baseline.yaml`: editable baseline parameter file
- `run_simulation.py`: standalone script to run the simulation

## How to run

1. Install dependencies:
pip install numpy pyyaml
