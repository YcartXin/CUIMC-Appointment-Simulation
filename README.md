# CUIMC-Appointment-Simulation

This repository contains a discrete-event clinic appointment simulation with:

- first-come-first-serve booking
- two patient classes
- delay-dependent balking
- delay-dependent no-show
- end-of-day cancellations
- rolling appointment calendar
- burn-in, measurement, and cooldown periods

## Files

- `FCFS_Model_Simulation.ipynb`: original working notebook
- Additional Python files will be added as the notebook is modularized

## Purpose

This project was developed to simulate clinic appointment scheduling behavior under a slot-by-slot booking system and evaluate metrics such as:

- average booking delay by class
- attended utilization by class
- overall attended utilization
- percent of customers serviced
- total value generated
