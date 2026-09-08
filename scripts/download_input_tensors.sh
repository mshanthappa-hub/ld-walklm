#!/usr/bin/env bash
set -euo pipefail

GH_REPO=${GH_REPO:-MeghanaShanthappa/ld-walklm}
RELEASE_TAG=${RELEASE_TAG:-ogbn-products-walklm-v2-10seed-20260904}
OUTPUT_DIR=${OUTPUT_DIR:-"${PWD}/input_tensors"}
BASENAME=walklm_products_2449029_embeddinggemma300m_walks4_len4_mean_norm_embeddings.pt
EXPECTED_SHA256=fef45b647df5155bf17505508cb08ee5a164ad9b33b69910ad1b437109721c90
PART_DIR="${OUTPUT_DIR}/parts"
OUTPUT="${OUTPUT_DIR}/${BASENAME}"

command -v gh >/dev/null || { echo "ERROR: GitHub CLI (gh) is required."; exit 1; }
command -v sha256sum >/dev/null || { echo "ERROR: sha256sum is required."; exit 1; }
mkdir -p "${PART_DIR}"

if [[ -f "${OUTPUT}" ]]; then
    actual=$(sha256sum "${OUTPUT}" | awk '{print $1}')
    if [[ "${actual}" == "${EXPECTED_SHA256}" ]]; then
        echo "Verified existing ${OUTPUT}"
        exit 0
    fi
    echo "ERROR: existing output has unexpected SHA-256: ${actual}"
    exit 1
fi

gh release download "${RELEASE_TAG}" \
    --repo "${GH_REPO}" \
    --dir "${PART_DIR}" \
    --pattern "${BASENAME}.part-*" \
    --clobber
gh release download "${RELEASE_TAG}" \
    --repo "${GH_REPO}" \
    --dir "${PART_DIR}" \
    --pattern walklm_embedding_parts.sha256 \
    --clobber

mapfile -t parts < <(find "${PART_DIR}" -maxdepth 1 -type f \
    -name "${BASENAME}.part-*" -print | LC_ALL=C sort)
if [[ "${#parts[@]}" -eq 0 ]]; then
    echo "ERROR: no split tensor assets were downloaded."
    exit 1
fi

(cd "${PART_DIR}" && sha256sum -c walklm_embedding_parts.sha256)

temporary="${OUTPUT}.tmp.$$"
: > "${temporary}"
for part in "${parts[@]}"; do
    cat "${part}" >> "${temporary}"
done

actual=$(sha256sum "${temporary}" | awk '{print $1}')
if [[ "${actual}" != "${EXPECTED_SHA256}" ]]; then
    echo "ERROR: joined tensor SHA-256 ${actual}; expected ${EXPECTED_SHA256}."
    exit 1
fi
mv "${temporary}" "${OUTPUT}"
echo "Verified ${OUTPUT}"
