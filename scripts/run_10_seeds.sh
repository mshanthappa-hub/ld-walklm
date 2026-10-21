#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# See ../THIRD_PARTY_NOTICES.md for dependency and third-party code notices.

set -euo pipefail

: "${SAGN_ROOT:?Set SAGN_ROOT to the feature-integrated SAGN checkout}"
: "${DATA_DIR:?Set DATA_DIR to the OGB dataset directory}"
: "${PROBS_DIR:?Set PROBS_DIR to the intermediate_outputs directory}"
: "${LD_FEATURE_PATH:?Set LD_FEATURE_PATH to products_sagn/hidden_state.pt}"
: "${WALKLM_FEATURE_PATH:?Set WALKLM_FEATURE_PATH to the EmbeddingGemma tensor}"
export LD_FEATURE_PATH WALKLM_FEATURE_PATH

PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_DIR="${PROBS_DIR}/ogbn-products/sagn"
LOG_DIR=${LOG_DIR:-"${PWD}/logs"}

export DGLBACKEND=pytorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${LOG_DIR}"
cd "${SAGN_ROOT}"

common_args=(
    --dataset ogbn-products
    --data_dir "${DATA_DIR}"
    --probs_dir "${PROBS_DIR}"
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

for seed in {0..9}; do
    checkpoint="${OUTPUT_DIR}/use_labels_True_use_feats_True_K_2_label_K_9_probs_seed_${seed}_stage_3.pt"
    if [[ -s "${checkpoint}" ]]; then
        echo "Seed ${seed} is complete; skipping."
        continue
    fi

    echo "Starting seed ${seed}"
    "${PYTHON_BIN}" -u src/sagn.py \
        "${common_args[@]}" \
        --seed "${seed}" \
        --warmup-stage -1 \
        2>&1 | tee "${LOG_DIR}/seed_${seed}.log"

    if [[ ! -s "${checkpoint}" ]]; then
        echo "ERROR: seed ${seed} did not produce stage 3."
        exit 1
    fi
done
