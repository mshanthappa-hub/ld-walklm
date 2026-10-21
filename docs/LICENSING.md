<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Licensing scope and release checks

## Project source and documentation

NVIDIA-authored source and documentation use the root [MIT license](../LICENSE).
The README author list remains an authorship credit; it is not a separate
assignment or legal ownership determination. This project is being prepared
as an NVIDIA project, subject to internal ownership, IP, and release review.

Third-party code retains its own copyright and terms. The SAGN integration
and adapted wrappers retain the original MIT notice. The root license does
not relicense third-party software, models, datasets, text, or generated
artifacts whose rights depend on those inputs.

## External contributions

External code and documentation contributions are not currently accepted.
Accordingly no CLA file is included. Revisit this decision and publish the
approved NVIDIA agreement and signing procedure before accepting external
contributions. See [CONTRIBUTING.md](../CONTRIBUTING.md).

## Models, inputs, and generated artifacts

Review these separately from the source-code MIT license:

- **OGB `ogbn-products`:** the official
  [dataset page](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products)
  links an [Amazon license](https://s3.amazonaws.com/amazon-reviews-pds/license.txt).
  Do not describe this dataset as CC0. The linked license could not be retrieved
  during this preparation; applicable terms and redistribution scope still
  need confirmation.
- **GLEM product titles/descriptions:** downloaded separately through the
  TAGDataset path. Availability via a loader does not establish rights to
  redistribute the text or derived artifacts.
- **LD BERT hidden-state tensors:** external inputs, not included as source
  here. Confirm applicable terms with the source/input providers.
- **EmbeddingGemma:** weights are obtained separately. The official
  [model card](https://huggingface.co/google/embeddinggemma-300m) identifies
  [Gemma terms](https://ai.google.dev/gemma/terms), not the project's MIT
  license. Review applicable model, use, and output terms.
- **Generated features, predictions, checkpoints, and logs:** an MIT source
  license does not itself clear their distribution. Review their inputs,
  contents, confidentiality, and applicable restrictions separately.

[PROVENANCE.md](../PROVENANCE.md) records software and input origins.
It is not a complete permission or license clearance.

## Before distribution

Include the project license, source notices, required third-party notices,
and all applicable license texts with each distribution. The
[third-party inventory](../THIRD_PARTY_NOTICES.md) includes reference snapshots
for dependencies whose exact installed versions were not preserved; it is
not a complete runtime license audit.

Complete team self/peer IP review, resolve model/data permissions and dependency
version gaps, and obtain required release approvals. Keep confidential internal
review records in the internal review system rather than this repository.
Repository housekeeping alone does not establish approval to release.
