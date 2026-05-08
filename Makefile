PYTHON ?= python
QUARTO ?= quarto

.PHONY: figures report docs test check

figures:
	$(PYTHON) experiments/sweep_class_1_balking.py
	$(PYTHON) experiments/sweep_class_1_balking_threshold.py
	$(PYTHON) docs/generate_metric_analysis_figures.py

report:
	$(QUARTO) render docs/metric_analysis.qmd

docs:
	$(QUARTO) render docs/metric_analysis.qmd
	$(QUARTO) render "docs/Class 1 Balking Sensitivity Analysis.qmd"
	$(QUARTO) render "docs/Class 1 Balking Threshold Analysis.qmd"
	$(QUARTO) render docs/simulation_documentation.qmd

test:
	$(PYTHON) -m unittest discover

check: test

