#!/usr/bin/env bash
set -euo pipefail

set -a
source .env
set +a

RUN_ID="${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"

mkdir -p "${RUN_DIR}/generations" "${RUN_DIR}/judgments" "${RUN_DIR}/summaries"

export RUN_DIR

python -m scripts.02_run_generation
python -m scripts.03_run_judge
python -m scripts.04_aggregate_results