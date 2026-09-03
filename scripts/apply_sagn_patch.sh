#!/usr/bin/env bash
set -euo pipefail

BASE_COMMIT=08fe607ed7f31ca39740a44ce2f139565de3c595
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
PATCH_FILE="${REPO_ROOT}/integration/0001-Add-LD-and-EmbeddingGemma-feature-integration.patch"
TARGET=${1:-${SAGN_ROOT:-}}

if [[ -z "${TARGET}" ]]; then
    echo "Usage: $0 /path/to/SAGN_with_SLE"
    echo "Alternatively set SAGN_ROOT."
    exit 2
fi

if ! git -C "${TARGET}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: not a Git checkout: ${TARGET}"
    exit 1
fi

actual_commit=$(git -C "${TARGET}" rev-parse HEAD)
if [[ "${actual_commit}" != "${BASE_COMMIT}" ]]; then
    echo "ERROR: SAGN HEAD is ${actual_commit}; expected ${BASE_COMMIT}."
    exit 1
fi

if git -C "${TARGET}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "Integration patch is already applied."
    exit 0
fi

if [[ -n "$(git -C "${TARGET}" status --porcelain)" ]]; then
    echo "ERROR: SAGN checkout has unrelated changes; refusing to patch."
    exit 1
fi

git -C "${TARGET}" apply --check "${PATCH_FILE}"
git -C "${TARGET}" apply "${PATCH_FILE}"

echo "Applied LD + EmbeddingGemma-WalkLM integration to ${TARGET}."
echo "Set LD_FEATURE_PATH and WALKLM_FEATURE_PATH before training."
