#!/usr/bin/env bash
set -euo pipefail

: "${INPUT_TENSOR:?Set INPUT_TENSOR to the complete archived EmbeddingGemma tensor}"
: "${UPLOAD_ROOT:?Set UPLOAD_ROOT to a persistent directory with at least 9 GB free}"
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
if find "${PART_DIR}" -maxdepth 1 -type f -name "${BASENAME}.part-*" | grep -q .; then
    echo "ERROR: split parts already exist in ${PART_DIR}; use a fresh UPLOAD_ROOT."
    exit 1
fi

split -b 1900M -a 2 "${INPUT_TENSOR}" "${PART_DIR}/${BASENAME}.part-"
mapfile -t parts < <(find "${PART_DIR}" -maxdepth 1 -type f \
    -name "${BASENAME}.part-*" -print | LC_ALL=C sort)
for part in "${parts[@]}"; do
    size=$(stat -c '%s' "${part}")
    if [[ "${size}" -ge 2147483648 ]]; then
        echo "ERROR: ${part} is not smaller than 2 GiB."
        exit 1
    fi
done
(cd "${PART_DIR}" && sha256sum "${BASENAME}.part-"*) \
    > "${PART_DIR}/walklm_embedding_parts.sha256"

for asset in "${parts[@]}" "${PART_DIR}/walklm_embedding_parts.sha256"; do
    gh release upload "${RELEASE_TAG}" "${asset}" \
        --repo "${GH_REPO}" --clobber
done
release_assets="${PART_DIR}/release_assets.tsv"
gh release view "${RELEASE_TAG}" --repo "${GH_REPO}" --json assets \
    --jq '.assets[] | [.name, .size] | @tsv' > "${release_assets}"
for asset in "${parts[@]}" "${PART_DIR}/walklm_embedding_parts.sha256"; do
    name=$(basename "${asset}")
    local_size=$(stat -c '%s' "${asset}")
    remote_size=$(awk -F '\t' -v name="${name}" '$1 == name {print $2}' "${release_assets}")
    if [[ "${remote_size}" != "${local_size}" ]]; then
        echo "ERROR: release asset ${name} has size ${remote_size:-missing}; expected ${local_size}."
        exit 1
    fi
done

echo "Uploaded ${#parts[@]} verified sub-2-GiB parts to ${GH_REPO}:${RELEASE_TAG}."
