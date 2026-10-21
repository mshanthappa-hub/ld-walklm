<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Third-party notices and dependency inventory

This file identifies third-party source and dependencies used by the project.
It does not establish complete license/IP clearance for every proposed release
artifact. See [licensing scope](docs/LICENSING.md).

## SAGN_with_SLE — patch context and adapted code

- Upstream: [skepsun/SAGN_with_SLE](https://github.com/skepsun/SAGN_with_SLE)
- Revision: `08fe607ed7f31ca39740a44ce2f139565de3c595`
- Copyright (c) 2021 skepsun
- License: MIT; retained full text:
  [LICENSES/SAGN_with_SLE-LICENSE.txt](LICENSES/SAGN_with_SLE-LICENSE.txt)

Affected files:

- `integration/0001-Add-LD-and-EmbeddingGemma-feature-integration.patch`:
  source context and modifications to `src/dataset.py`, `src/pre_process.py`,
  `src/sagn.py`, and `src/train_process.py`; introduces NVIDIA-authored
  `src/feature_stack.py`.
- `scripts/sagn_validation_only.py`: evaluator adapted from upstream
  `src/train_process.py`.
- `scripts/post_process_detailed.py`,
  `scripts/post_process_validation_only.py`, and
  `scripts/post_process_final_once.py`: checkpoint-loading and metric
  routines adapted from upstream `src/post_process.py`.

NVIDIA notices apply to NVIDIA-authored modifications. The upstream copyright,
MIT permission notice, and disclaimer are retained independently. Keep the
upstream SAGN LICENSE when applying the patch; do not replace it with the
project LICENSE.

## Separately installed dependencies

Tracked source does not vendor these packages, wheels, or runtime libraries.
They are imported, used by SAGN, or listed in `requirements-reproduction.txt`.
License texts are included for review/attribution; inclusion does not mean
the corresponding package source or binaries are redistributed.

| Package | Top-level license | Version evidence |
| --- | --- | --- |
| [torch](LICENSES/torch-LICENSE.txt) | BSD-3-Clause-style | reference-only: recorded development build commit unknown |
| [dgl](LICENSES/dgl-LICENSE.txt) | Apache-2.0 | recorded and required 1.1.3 |
| [ogb](LICENSES/ogb-LICENSE.txt) | MIT | recorded and required 1.3.6 |
| [torch-geometric](LICENSES/torch-geometric-LICENSE.txt) | MIT | required 2.8.0.post1; full runtime not captured |
| [transformers](LICENSES/transformers-LICENSE.txt) | Apache-2.0 | required 5.15.0; original embedding runtime not preserved |
| [numpy](LICENSES/numpy-LICENSE.txt) | BSD-3-Clause | reference-only: unpinned |
| [pandas](LICENSES/pandas-LICENSE.txt) | BSD-3-Clause | reference-only: unpinned |
| [scipy](LICENSES/scipy-LICENSE.txt) | BSD-3-Clause | reference-only: unpinned |
| [scikit-learn](LICENSES/scikit-learn-LICENSE.txt) | BSD-3-Clause | reference-only: unpinned |
| [networkx](LICENSES/networkx-LICENSE.txt) | BSD-3-Clause | reference-only: unpinned |
| [matplotlib](LICENSES/matplotlib-LICENSE.txt) | Matplotlib license (PSF-based) | reference-only: unpinned; additional bundled-component notices not included |
| [seaborn](LICENSES/seaborn-LICENSE.txt) | BSD-3-Clause | reference-only: unpinned |
| [safetensors](LICENSES/safetensors-LICENSE.txt) | Apache-2.0 | reference-only: unpinned |
| [tqdm](LICENSES/tqdm-LICENSE.txt) | MPL-2.0 and MIT components | reference-only: unpinned; coverage is file-specific |

Exact source URLs and immutable revisions:
[LICENSES/README.md](LICENSES/README.md).

For unpinned packages and the development PyTorch build, these are
**reference snapshots**, not verified license texts from the original
installed artifacts. The recorded environment is not a complete package lock
or software bill of materials. A wheel/container/runtime distribution needs
an inventory of actual artifacts, transitive packages, file-level licenses,
bundled components, and required notices.

Matplotlib has additional component notices; NumPy, SciPy, PyTorch and binary
wheels may bundle other libraries. tqdm uses MIT and MPL-2.0 for different
components, not an unrestricted choice between them. The full
[MPL-2.0 text](LICENSES/MPL-2.0.txt) is also included.

## Models and datasets

EmbeddingGemma weights, GLEM product text, LD feature tensors, OGB data, and
derived artifacts are not covered by the project MIT license. Distribution
permissions are not established by this file. See
[docs/LICENSING.md](docs/LICENSING.md) and [PROVENANCE.md](PROVENANCE.md).
