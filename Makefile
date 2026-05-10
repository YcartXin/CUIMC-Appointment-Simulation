PYTHON ?= python
QUARTO ?= quarto
REPORT_SOURCE_DIR := docs/reports/metric_analysis
REPORT_DIR := $(REPORT_SOURCE_DIR)/thorough
REPORT_BUILD_DIR := $(REPORT_DIR)/build
REPORT_BUILD_EXTS := aux fdb_latexmk fls log out synctex.gz toc

.PHONY: figures report docs test check

figures:
	$(PYTHON) experiments/sweep_class_1_balking.py
	$(PYTHON) experiments/sweep_class_1_balking_threshold.py
	$(PYTHON) scripts/generate_metric_analysis_figures.py

report:
	mkdir -p $(REPORT_DIR) $(REPORT_BUILD_DIR)
	cd $(REPORT_SOURCE_DIR) && $(QUARTO) render metric_analysis.qmd --output-dir thorough
	@for ext in $(REPORT_BUILD_EXTS); do \
		if [ -e "$(REPORT_DIR)/metric_analysis.$$ext" ]; then \
			mv "$(REPORT_DIR)/metric_analysis.$$ext" "$(REPORT_BUILD_DIR)/"; \
		fi; \
	done

docs:
	mkdir -p $(REPORT_DIR) $(REPORT_BUILD_DIR) outputs/companions docs/reference/simulation_explanation
	cd $(REPORT_SOURCE_DIR) && $(QUARTO) render metric_analysis.qmd --output-dir thorough
	@for ext in $(REPORT_BUILD_EXTS); do \
		if [ -e "$(REPORT_DIR)/metric_analysis.$$ext" ]; then \
			mv "$(REPORT_DIR)/metric_analysis.$$ext" "$(REPORT_BUILD_DIR)/"; \
		fi; \
	done
	cd docs/companions && $(QUARTO) render "Class 1 Balking Sensitivity Analysis.qmd" --output-dir ../../outputs/companions
	cd docs/companions && $(QUARTO) render "Class 1 Balking Threshold Analysis.qmd" --output-dir ../../outputs/companions
	cd docs/reference && $(QUARTO) render simulation_documentation.qmd --output-dir simulation_explanation

test:
	$(PYTHON) -m unittest discover

check: test
