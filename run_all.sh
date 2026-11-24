#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
export PROCESSED_DIR="${PROCESSED_DIR:-${DATA_DIR}/processed}"
export UPSTREAM_DIR="${UPSTREAM_DIR:-${ROOT_DIR}}"
export RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results}"
export FIG_DIR="${FIG_DIR:-${ROOT_DIR}/figures}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

cd "${ROOT_DIR}"

if [ "${SKIP_DOWNLOAD}" != "1" ]; then
  python3 scripts/00_download_data.py --data-dir "${DATA_DIR}"
fi

python3 scripts/01_preprocess.py --data-dir "${DATA_DIR}" --processed-dir "${PROCESSED_DIR}"
python3 scripts/02_analyze_paired_shifts.py --data-dir "${DATA_DIR}" --processed-dir "${PROCESSED_DIR}" --upstream-dir "${UPSTREAM_DIR}" --results-dir "${RESULTS_DIR}"
python3 scripts/03_falsification_controls.py --data-dir "${DATA_DIR}" --processed-dir "${PROCESSED_DIR}" --results-dir "${RESULTS_DIR}"
python3 scripts/04_make_figures_tables.py --results-dir "${RESULTS_DIR}" --fig-dir "${FIG_DIR}"
python3 scripts/05_verify_results.py --data-dir "${DATA_DIR}" --processed-dir "${PROCESSED_DIR}" --results-dir "${RESULTS_DIR}" --fig-dir "${FIG_DIR}"

DATA_CACHE="${DATA_DIR}" PROCESSED_CACHE="${PROCESSED_DIR}" RESULTS_CACHE="${RESULTS_DIR}" \
  Rscript -e "rmarkdown::render('notebooks/eda_walkthrough.Rmd', output_file='eda_walkthrough.html', quiet=TRUE)"
