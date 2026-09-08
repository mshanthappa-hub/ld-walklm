#!/usr/bin/env bash
set -euo pipefail

: "${SAGN_ROOT:?Set SAGN_ROOT to the patched SAGN_with_SLE checkout}"
: "${DATA_DIR:?Set DATA_DIR to the parent OGB dataset directory}"

PYTHON_BIN=${PYTHON_BIN:-python}
GH_REPO=${GH_REPO:-MeghanaShanthappa/ld-walklm}
RELEASE_TAG=${RELEASE_TAG:-ogbn-products-walklm-v2-10seed-20260904}
REPRO_ROOT=${REPRO_ROOT:-"${PWD}/v2_release_reproduction"}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
PROBS_DIR="${REPRO_ROOT}/intermediate_outputs"
OUTPUT_DIR="${PROBS_DIR}/ogbn-products/sagn"
GRID_DIR="${REPRO_ROOT}/validation_grid"
FINAL_DIR="${REPRO_ROOT}/final_test"
MANIFEST_DIR="${REPRO_ROOT}/manifests"
COMMITTED_GRID="${REPO_ROOT}/results/v2/validation_grid"
COMMITTED_FINAL="${REPO_ROOT}/results/v2/final_test/final_test.json"
COMMITTED_MANIFEST="${REPO_ROOT}/results/v2/archive/v2_stage3_sha256.txt"

command -v gh >/dev/null || { echo "ERROR: GitHub CLI (gh) is required."; exit 1; }
mkdir -p "${OUTPUT_DIR}" "${GRID_DIR}" "${FINAL_DIR}" "${MANIFEST_DIR}"

gh release download "${RELEASE_TAG}" \
    --repo "${GH_REPO}" \
    --dir "${OUTPUT_DIR}" \
    --pattern '*seed_1?_stage_3.pt' \
    --clobber
gh release download "${RELEASE_TAG}" \
    --repo "${GH_REPO}" \
    --dir "${MANIFEST_DIR}" \
    --pattern v2_stage3_sha256.txt \
    --clobber

cmp "${COMMITTED_MANIFEST}" "${MANIFEST_DIR}/v2_stage3_sha256.txt"
checkpoint_count=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f \
    -name '*seed_1?_stage_3.pt' -size +0c | wc -l)
if [[ "${checkpoint_count}" -ne 10 ]]; then
    echo "ERROR: expected 10 stage-3 tensors; found ${checkpoint_count}."
    exit 1
fi
(cd "${OUTPUT_DIR}" && sha256sum -c "${COMMITTED_MANIFEST}")

SAGN_ROOT="${SAGN_ROOT}" DATA_DIR="${DATA_DIR}" \
PROBS_DIR="${PROBS_DIR}" RESULTS_DIR="${GRID_DIR}" \
SEED_START=10 PYTHON_BIN="${PYTHON_BIN}" \
bash "${SCRIPT_DIR}/tune_cs_validation_only.sh"

diff -ru "${COMMITTED_GRID}" "${GRID_DIR}"

MARKER="${FINAL_DIR}/REPRODUCTION_TEST_ACCESS_STARTED"
OUTPUT_JSON="${FINAL_DIR}/final_test.json"
LOG_FILE="${FINAL_DIR}/final_test.log"
if [[ -e "${MARKER}" || -e "${OUTPUT_JSON}" ]]; then
    echo "ERROR: reproduction test access already started in ${FINAL_DIR}."
    exit 1
fi
{
    echo "status=started"
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "lock_sha256=$(sha256sum "${COMMITTED_GRID}/locked_cs_config.json" | awk '{print $1}')"
} > "${MARKER}"

export DGLBACKEND=pytorch
export SAGN_ROOT
export SAGN_SRC_DIR="${SAGN_ROOT}/src"
set +e
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/post_process_final_once.py" \
    --dataset ogbn-products \
    --data-dir "${DATA_DIR}" \
    --probs_dir "${PROBS_DIR}" \
    --gpu -1 \
    --model sagn \
    --K 2 \
    --label-K 9 \
    --use-labels \
    --stage 3 \
    --operations correction smoothing \
    --expected-seeds 10 11 12 13 14 15 16 17 18 19 \
    --lock-file "${COMMITTED_GRID}/locked_cs_config.json" \
    --output-json "${OUTPUT_JSON}" \
    2>&1 | tee "${LOG_FILE}"
exit_code=${PIPESTATUS[0]}
set -e
if [[ "${exit_code}" -ne 0 ]]; then
    echo "status=failed_exit_${exit_code}" >> "${MARKER}"
    exit "${exit_code}"
fi
echo "status=complete" >> "${MARKER}"
cmp "${COMMITTED_FINAL}" "${OUTPUT_JSON}"
echo "Verified released v2 tensors, validation grid, lock, and final result."
