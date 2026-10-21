#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# See ../THIRD_PARTY_NOTICES.md for dependency and third-party code notices.

set -euo pipefail

: "${SOURCE_REPO:?Set SOURCE_REPO to the superseded repository}"
SOURCE_TAG=${SOURCE_TAG:-ogbn-products-sagn-10seed-20260903}
TARGET_REPO=${TARGET_REPO:-MeghanaShanthappa/ld-walklm}
TARGET_TAG=${TARGET_TAG:-ogbn-products-walklm-v1-exploratory-20260903}
: "${MIGRATION_ROOT:?Set MIGRATION_ROOT to a persistent directory with enough free space}"
ASSET_DIR="${MIGRATION_ROOT}/v1_release_assets"

command -v gh >/dev/null || { echo "ERROR: GitHub CLI (gh) is required."; exit 1; }
mkdir -p "${ASSET_DIR}"
gh auth status

gh release edit "${SOURCE_TAG}" \
    --repo "${SOURCE_REPO}" \
    --title 'Archived v1: LD + frozen EmbeddingGemma walk-text + SAGN/SLE + C&S (10 seeds)'
gh release download "${SOURCE_TAG}" \
    --repo "${SOURCE_REPO}" \
    --dir "${ASSET_DIR}" \
    --clobber

if ! gh release view "${TARGET_TAG}" --repo "${TARGET_REPO}" >/dev/null 2>&1; then
    gh release create "${TARGET_TAG}" \
        --repo "${TARGET_REPO}" \
        --draft \
        --title 'Archived v1: LD + frozen EmbeddingGemma walk-text + SAGN/SLE + C&S (10 seeds)' \
        --notes 'Archived test-aware v1 artifacts. Not the primary validation-locked v2 result.'
fi

while IFS= read -r asset; do
    gh release upload "${TARGET_TAG}" "${asset}" \
        --repo "${TARGET_REPO}" --clobber
done < <(find "${ASSET_DIR}" -maxdepth 1 -type f -print | LC_ALL=C sort)

source_json="${MIGRATION_ROOT}/source_assets.json"
target_json="${MIGRATION_ROOT}/target_assets.json"
gh release view "${SOURCE_TAG}" --repo "${SOURCE_REPO}" \
    --json assets --jq '[.assets[] | {name: .name, size: .size}] | sort_by(.name)' > "${source_json}"
gh release view "${TARGET_TAG}" --repo "${TARGET_REPO}" \
    --json assets --jq '[.assets[] | {name: .name, size: .size}] | sort_by(.name)' > "${target_json}"
cmp "${source_json}" "${target_json}"
gh release edit "${TARGET_TAG}" --repo "${TARGET_REPO}" --draft=false

echo "Verified and published ${TARGET_REPO} release ${TARGET_TAG}."
echo "After checking the release in a browser, retire public access to ${SOURCE_REPO}."
