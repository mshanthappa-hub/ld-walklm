#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# See ../THIRD_PARTY_NOTICES.md for dependency and third-party code notices.

set -euo pipefail

: "${SAGN_ROOT:?Set SAGN_ROOT to the feature-integrated SAGN checkout}"
: "${DATA_DIR:?Set DATA_DIR to the OGB dataset directory}"
: "${V2_PROBS_DIR:?Set V2_PROBS_DIR to a new, dedicated output directory}"
: "${LD_FEATURE_PATH:?Set LD_FEATURE_PATH to products_sagn/hidden_state.pt}"
: "${WALKLM_FEATURE_PATH:?Set WALKLM_FEATURE_PATH to the EmbeddingGemma tensor}"

PYTHON_BIN=${PYTHON_BIN:-python}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
PRE_REGISTRATION="${REPO_ROOT}/protocol/v2_pre_registration.yaml"
V2_RESULTS_DIR=${V2_RESULTS_DIR:-"${REPO_ROOT}/results/v2"}
OUTPUT_DIR="${V2_PROBS_DIR}/ogbn-products/sagn"
LOG_DIR=${LOG_DIR:-"${V2_RESULTS_DIR}/logs"}
HELPER="${SCRIPT_DIR}/sagn_validation_only.py"

git -C "${REPO_ROOT}" ls-files --error-unmatch \
    protocol/v2_pre_registration.yaml >/dev/null
if ! git -C "${REPO_ROOT}" diff --quiet HEAD -- \
    protocol/v2_pre_registration.yaml; then
    echo "ERROR: commit the pre-registration before starting v2."
    exit 1
fi

if find "${OUTPUT_DIR}" -maxdepth 1 -type f \
    -name '*seed_[0-9]_stage_*.pt' -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: V2_PROBS_DIR contains v1 seed outputs; use a dedicated directory."
    exit 1
fi

export DGLBACKEND=pytorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_FEATURE_PATH WALKLM_FEATURE_PATH SAGN_ROOT
export SAGN_SRC_DIR="${SAGN_ROOT}/src"
mkdir -p "${LOG_DIR}"
cd "${SAGN_ROOT}"

common_args=(
    --dataset ogbn-products
    --data_dir "${DATA_DIR}"
    --probs_dir "${V2_PROBS_DIR}"
    --gpu 0
    --aggr-gpu -1
    --model sagn
    --num-runs 1
    --epoch-setting 300 100 100 100
    --threshold 0.95 0.95 0.999
    --eval-every 20
    --lr 0.001
    --zero-inits
    --batch-size 50000
    --eval-batch-size 100000
    --num-hidden 512
    --num-heads 1
    --dropout 0.5
    --attn-drop 0.4
    --input-drop 0.2
    --label-drop 0.5
    --K 2
    --label-K 9
    --use-labels
    --weight-decay 0
    --chunks 4
    --memory-efficient
)

for seed in {10..19}; do
    checkpoint="${OUTPUT_DIR}/use_labels_True_use_feats_True_K_2_label_K_9_probs_seed_${seed}_stage_3.pt"
    if [[ -s "${checkpoint}" ]]; then
        echo "Seed ${seed} is complete; skipping."
        continue
    fi

    echo "Starting validation-only v2 seed ${seed}"
    "${PYTHON_BIN}" -u "${HELPER}" \
        "${common_args[@]}" \
        --seed "${seed}" \
        --warmup-stage -1 \
        2>&1 | tee "${LOG_DIR}/seed_${seed}.log"

    if [[ ! -s "${checkpoint}" ]]; then
        echo "ERROR: seed ${seed} did not produce stage 3."
        exit 1
    fi
done

cd "${REPO_ROOT}"
SAGN_ROOT="${SAGN_ROOT}" \
DATA_DIR="${DATA_DIR}" \
PROBS_DIR="${V2_PROBS_DIR}" \
PYTHON_BIN="${PYTHON_BIN}" \
RESULTS_DIR="${V2_RESULTS_DIR}/validation_grid" \
SEED_START=10 \
    bash "${SCRIPT_DIR}/tune_cs_validation_only.sh"

echo "Validation-only v2 is locked. Do not run the test evaluator yet."
echo "Review and commit ${V2_RESULTS_DIR}/validation_grid before final evaluation."
