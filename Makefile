PYTHON ?= python
QUARTO ?= quarto

.PHONY: figures report docs test check

figures:
	$(PYTHON) experiments/sweep_class_1_balking.py
	$(PYTHON) experiments/sweep_class_1_balking_threshold.py
	$(PYTHON) scripts/generate_metric_analysis_figures.py

report:
	mkdir -p outputs/reports/metric_analysis/rendered
	cd docs/reports && $(QUARTO) render metric_analysis.qmd --output-dir ../../outputs/reports/metric_analysis/rendered

docs:
	mkdir -p outputs/reports/metric_analysis/rendered outputs/companions outputs/reference
	cd docs/reports && $(QUARTO) render metric_analysis.qmd --output-dir ../../outputs/reports/metric_analysis/rendered
	cd docs/companions && $(QUARTO) render "Class 1 Balking Sensitivity Analysis.qmd" --output-dir ../../outputs/companions
	cd docs/companions && $(QUARTO) render "Class 1 Balking Threshold Analysis.qmd" --output-dir ../../outputs/companions
	cd docs/reference && $(QUARTO) render simulation_documentation.qmd --output-dir ../../outputs/reference

test:
	$(PYTHON) -m unittest discover

check: test
