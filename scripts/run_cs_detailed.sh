#!/usr/bin/env bash
set -euo pipefail

: "${SAGN_ROOT:?Set SAGN_ROOT to the SAGN_with_SLE checkout}"
: "${DATA_DIR:?Set DATA_DIR to the OGB dataset directory}"
: "${PROBS_DIR:?Set PROBS_DIR to the intermediate_outputs directory}"

PYTHON_BIN=${PYTHON_BIN:-python}
RESULTS_DIR=${RESULTS_DIR:-"${PWD}/results"}
OUTPUT_DIR="${PROBS_DIR}/ogbn-products/sagn"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
HELPER="${SCRIPT_DIR}/post_process_detailed.py"

export DGLBACKEND=pytorch
export SAGN_ROOT
export SAGN_SRC_DIR="${SAGN_ROOT}/src"

checkpoint_count=$(
    find "${OUTPUT_DIR}" -maxdepth 1 -type f \
        -name '*seed_*_stage_3.pt' -size +0c | wc -l
)
if [[ "${checkpoint_count}" -ne 10 ]]; then
    echo "ERROR: expected 10 stage-3 checkpoints, found ${checkpoint_count}."
    exit 1
fi

mkdir -p "${RESULTS_DIR}"
cd "${SAGN_ROOT}"

exec "${PYTHON_BIN}" -u "${HELPER}" \
    --dataset ogbn-products \
    --data-dir "${DATA_DIR}" \
    --probs_dir "${PROBS_DIR}" \
    --gpu -1 \
    --model sagn \
    --K 2 \
    --label-K 9 \
    --use-labels \
    --stage 3 \
    --num-correction-layers 50 \
    --num-smoothing-layers 50 \
    --correction-alpha 0.3 \
    --smoothing-alpha 0.65 \
    --correction-adj DAD \
    --smoothing-adj DAD \
    --scale 25 \
    --operations correction smoothing \
    --expected-runs 10 \
    --results-tsv "${RESULTS_DIR}/cs_per_seed.tsv"
