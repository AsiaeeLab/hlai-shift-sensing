PYTHON ?= python3
RSCRIPT ?= Rscript
DATA_DIR ?= ./data
PROCESSED_DIR ?= ./data/processed
UPSTREAM_DIR ?= .
RESULTS_DIR ?= ./results
FIG_DIR ?= ./figures

export DATA_DIR PROCESSED_DIR UPSTREAM_DIR RESULTS_DIR FIG_DIR OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0

.PHONY: all test download preprocess analyze falsification figures verify report clean

all: test preprocess analyze falsification figures verify report

test:
	$(PYTHON) -m unittest discover -s tests -v

download:
	$(PYTHON) scripts/00_download_data.py --data-dir "$(DATA_DIR)"

preprocess:
	$(PYTHON) scripts/01_preprocess.py --data-dir "$(DATA_DIR)" --processed-dir "$(PROCESSED_DIR)"

analyze:
	$(PYTHON) scripts/02_analyze_paired_shifts.py --data-dir "$(DATA_DIR)" --processed-dir "$(PROCESSED_DIR)" --upstream-dir "$(UPSTREAM_DIR)" --results-dir "$(RESULTS_DIR)"

falsification:
	$(PYTHON) scripts/03_falsification_controls.py --data-dir "$(DATA_DIR)" --processed-dir "$(PROCESSED_DIR)" --results-dir "$(RESULTS_DIR)"

figures:
	$(PYTHON) scripts/04_make_figures_tables.py --results-dir "$(RESULTS_DIR)" --fig-dir "$(FIG_DIR)"

verify:
	$(PYTHON) scripts/05_verify_results.py --data-dir "$(DATA_DIR)" --processed-dir "$(PROCESSED_DIR)" --results-dir "$(RESULTS_DIR)" --fig-dir "$(FIG_DIR)"

report:
	for rmd in notebooks/0*.Rmd; do DATA_CACHE="$(DATA_DIR)" PROCESSED_CACHE="$(PROCESSED_DIR)" RESULTS_CACHE="$(RESULTS_DIR)" FIGURES_CACHE="$(FIG_DIR)" $(RSCRIPT) -e "rmarkdown::render('$$rmd', quiet=TRUE)"; done

clean:
	rm -rf "$(RESULTS_DIR)"/* "$(FIG_DIR)"/*.pdf "$(FIG_DIR)"/*.png "$(FIG_DIR)"/*.svg
