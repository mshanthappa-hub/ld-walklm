#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# See ../THIRD_PARTY_NOTICES.md for dependency and third-party code notices.

set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to the parent OGB dataset directory}"
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT=${OUTPUT:-"${PWD}/walklm_products_2449029_embeddinggemma300m_walks4_len4_mean_norm_embeddings.pt"}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
EXTRA_ARGS=()
if [[ -n "${MAX_NODES:-}" ]]; then
    EXTRA_ARGS+=(--max-nodes "${MAX_NODES}")
fi

exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/generate_embeddinggemma_features.py" \
    --data-dir "${DATA_DIR}" \
    --output "${OUTPUT}" \
    --model google/embeddinggemma-300m \
    --model-revision 57c266a7 \
    --seed 42 \
    --num-walks 4 \
    --walk-length 4 \
    --max-length 160 \
    --max-text-chars 160 \
    --batch-size 4 \
    "${EXTRA_ARGS[@]}"
