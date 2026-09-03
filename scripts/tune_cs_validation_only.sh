#!/usr/bin/env bash
set -euo pipefail

: "${SAGN_ROOT:?Set SAGN_ROOT to the patched SAGN_with_SLE checkout}"
: "${DATA_DIR:?Set DATA_DIR to the OGB dataset directory}"
: "${PROBS_DIR:?Set PROBS_DIR to the intermediate_outputs directory}"

PYTHON_BIN=${PYTHON_BIN:-python}
RESULTS_DIR=${RESULTS_DIR:-"${PWD}/validation_grid"}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
HELPER="${SCRIPT_DIR}/post_process_validation_only.py"
SEED_START=${SEED_START:-0}
expected_seeds=()
for ((offset = 0; offset < 10; offset++)); do
    expected_seeds+=("$((SEED_START + offset))")
done
mkdir -p "${RESULTS_DIR}"

export DGLBACKEND=pytorch
export SAGN_ROOT
export SAGN_SRC_DIR="${SAGN_ROOT}/src"

# This grid is frozen before execution. Test accuracy is neither computed nor
# printed by the helper. Expand the grid only before running it, and commit the
# change so that the candidate set is auditable.
for correction_alpha in 0.30 0.35 0.40; do
    for smoothing_alpha in 0.60 0.65 0.70; do
        for scale in 20 25 30; do
            config_id="c${correction_alpha}_s${smoothing_alpha}_z${scale}"
            output_json="${RESULTS_DIR}/${config_id}.json"
            if [[ -s "${output_json}" ]]; then
                echo "Skipping completed ${config_id}"
                continue
            fi
            "${PYTHON_BIN}" -u "${HELPER}" \
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
                --correction-alpha "${correction_alpha}" \
                --smoothing-alpha "${smoothing_alpha}" \
                --correction-adj DAD \
                --smoothing-adj DAD \
                --scale "${scale}" \
                --operations correction smoothing \
                --expected-runs 10 \
                --expected-seeds "${expected_seeds[@]}" \
                --output-json "${output_json}"
        done
    done
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/select_cs_by_validation.py" \
    --grid-dir "${RESULTS_DIR}" \
    --output "${RESULTS_DIR}/locked_cs_config.json"
