#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# See ../THIRD_PARTY_NOTICES.md for dependency and third-party code notices.

set -euo pipefail

: "${INPUT_TENSOR:?Set INPUT_TENSOR to the complete archived EmbeddingGemma tensor}"
: "${UPLOAD_ROOT:?Set UPLOAD_ROOT to a directory with at least 2 GB free}"
GH_REPO=${GH_REPO:-MeghanaShanthappa/ld-walklm}
RELEASE_TAG=${RELEASE_TAG:-ogbn-products-walklm-v2-10seed-20260904}
BASENAME=walklm_products_2449029_embeddinggemma300m_walks4_len4_mean_norm_embeddings.pt
EXPECTED_SHA256=fef45b647df5155bf17505508cb08ee5a164ad9b33b69910ad1b437109721c90
PART_DIR="${UPLOAD_ROOT}/embeddinggemma_release_parts"

command -v gh >/dev/null || { echo "ERROR: GitHub CLI (gh) is required."; exit 1; }
actual=$(sha256sum "${INPUT_TENSOR}" | awk '{print $1}')
if [[ "${actual}" != "${EXPECTED_SHA256}" ]]; then
    echo "ERROR: input SHA-256 ${actual}; expected ${EXPECTED_SHA256}."
    exit 1
fi
mkdir -p "${PART_DIR}"
find "${PART_DIR}" -maxdepth 1 -type f -name "${BASENAME}.part-*" -delete
checksum_file="${PART_DIR}/walklm_embedding_parts.sha256"
: > "${checksum_file}"

file_size=$(stat -c "%s" "${INPUT_TENSOR}")
chunk_mib=1900
chunk_bytes=$((chunk_mib * 1024 * 1024))
offset=0
index=0

while [[ "${offset}" -lt "${file_size}" ]]; do
    suffix=$(printf "%02d" "${index}")
    name="${BASENAME}.part-${suffix}"
    part="${PART_DIR}/${name}"
    dd if="${INPUT_TENSOR}" of="${part}" bs=1M \
        skip="$((index * chunk_mib))" count="${chunk_mib}" status=progress
    local_size=$(stat -c "%s" "${part}")
    if [[ "${local_size}" -ge 2147483648 ]]; then
        echo "ERROR: ${part} is not smaller than 2 GiB."
        exit 1
    fi
    (cd "${PART_DIR}" && sha256sum "${name}") >> "${checksum_file}"
    gh release upload "${RELEASE_TAG}" "${part}" \
        --repo "${GH_REPO}" --clobber
    remote_size=$(gh release view "${RELEASE_TAG}" --repo "${GH_REPO}" \
        --json assets --jq ".assets[] | select(.name == \"${name}\") | .size")
    if [[ "${remote_size}" != "${local_size}" ]]; then
        echo "ERROR: release asset ${name} has size ${remote_size:-missing}; expected ${local_size}."
        exit 1
    fi
    rm -f "${part}"
    offset=$((offset + chunk_bytes))
    index=$((index + 1))
done

gh release upload "${RELEASE_TAG}" "${checksum_file}" \
    --repo "${GH_REPO}" --clobber
echo "Uploaded and size-verified ${index} sub-2-GiB parts to ${GH_REPO}:${RELEASE_TAG}."
