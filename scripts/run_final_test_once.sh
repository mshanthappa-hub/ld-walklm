#!/usr/bin/env bash
set -euo pipefail

: "${SAGN_ROOT:?Set SAGN_ROOT to the patched SAGN_with_SLE checkout}"
: "${DATA_DIR:?Set DATA_DIR to the OGB dataset directory}"
: "${V2_PROBS_DIR:?Set V2_PROBS_DIR to the dedicated v2 output directory}"

PYTHON_BIN=${PYTHON_BIN:-python}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
LOCK_FILE=${LOCK_FILE:-"${REPO_ROOT}/results/v2/validation_grid/locked_cs_config.json"}
FINAL_DIR=${FINAL_DIR:-"${REPO_ROOT}/results/v2/final_test"}
MARKER="${FINAL_DIR}/TEST_ACCESS_STARTED"
OUTPUT_JSON="${FINAL_DIR}/final_test.json"
LOG_FILE="${FINAL_DIR}/final_test.log"

case "${LOCK_FILE}" in
    "${REPO_ROOT}"/*) ;;
    *)
        echo "ERROR: LOCK_FILE must be inside the evidence repository."
        exit 1
        ;;
esac
lock_relative=${LOCK_FILE#"${REPO_ROOT}/"}
lock_directory=${lock_relative%/*}

git -C "${REPO_ROOT}" ls-files --error-unmatch "${lock_relative}" >/dev/null
if ! git -C "${REPO_ROOT}" diff --quiet HEAD -- "${lock_directory}"; then
    echo "ERROR: commit the validation records and lock before final test access."
    exit 1
fi
candidate_count=$(
    git -C "${REPO_ROOT}" ls-files "${lock_directory}/*.json" | wc -l
)
if [[ "${candidate_count}" -ne 28 ]]; then
    echo "ERROR: expected 27 candidates plus one lock; found ${candidate_count}."
    exit 1
fi
if [[ -e "${MARKER}" ]]; then
    echo "ERROR: final test access has already started; inspect ${MARKER}."
    exit 1
fi

mkdir -p "${FINAL_DIR}"
{
    echo "status=started"
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "lock_sha256=$(sha256sum "${LOCK_FILE}" | awk '{print $1}')"
} > "${MARKER}"

export DGLBACKEND=pytorch
export SAGN_ROOT
export SAGN_SRC_DIR="${SAGN_ROOT}/src"
cd "${SAGN_ROOT}"

set +e
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/post_process_final_once.py" \
    --dataset ogbn-products \
    --data-dir "${DATA_DIR}" \
    --probs_dir "${V2_PROBS_DIR}" \
    --gpu -1 \
    --model sagn \
    --K 2 \
    --label-K 9 \
    --use-labels \
    --stage 3 \
    --operations correction smoothing \
    --expected-seeds 10 11 12 13 14 15 16 17 18 19 \
    --lock-file "${LOCK_FILE}" \
    --output-json "${OUTPUT_JSON}" \
    2>&1 | tee "${LOG_FILE}"
exit_code=${PIPESTATUS[0]}
set -e

if [[ "${exit_code}" -eq 0 ]]; then
    echo "status=complete" >> "${MARKER}"
else
    echo "status=failed_exit_${exit_code}" >> "${MARKER}"
fi
exit "${exit_code}"
