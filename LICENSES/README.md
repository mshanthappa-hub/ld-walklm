<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# License-copy sources

Third-party texts retain their upstream terms and copyright notices.
The NVIDIA notice on this index applies only to this index.

## Included/adapted third-party source

`SAGN_with_SLE-LICENSE.txt` is copied from
[the pinned upstream LICENSE](https://github.com/skepsun/SAGN_with_SLE/blob/08fe607ed7f31ca39740a44ce2f139565de3c595/LICENSE).
It covers SAGN context/adapted code identified in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Dependency reference texts

Fetched on 2026-10-21. Each URL identifies an immutable source revision.
These are top-level texts, not complete wheel/runtime notice bundles.
**Reference-only** means the revision is not established as the original
installed dependency revision. This index does not pin or change dependencies.

| Copy | Official source revision | Binding to the recorded run |
| --- | --- | --- |
| [torch-LICENSE.txt](torch-LICENSE.txt) | [pytorch/pytorch `6dbae335eb88`](https://github.com/pytorch/pytorch/blob/6dbae335eb883eb25b41e06a0723895f9e20f7bf/LICENSE) | reference-only: recorded development build commit unknown |
| [dgl-LICENSE.txt](dgl-LICENSE.txt) | [dmlc/dgl `6e2c0f489ba4`](https://github.com/dmlc/dgl/blob/6e2c0f489ba4cfae4614d6f07da41e0ecd36b226/LICENSE) | recorded and required 1.1.3 |
| [ogb-LICENSE.txt](ogb-LICENSE.txt) | [snap-stanford/ogb `a47b716f7e97`](https://github.com/snap-stanford/ogb/blob/a47b716f7e972f666eae9909ee0f922cd0f9d966/LICENSE) | recorded and required 1.3.6 |
| [torch-geometric-LICENSE.txt](torch-geometric-LICENSE.txt) | [pyg-team/pytorch_geometric `d4e1c4568996`](https://github.com/pyg-team/pytorch_geometric/blob/d4e1c45689967a614a7f3d873ad4ba83c4a1f3bd/LICENSE) | required 2.8.0.post1; full runtime not captured |
| [transformers-LICENSE.txt](transformers-LICENSE.txt) | [huggingface/transformers `5eddc12edfaf`](https://github.com/huggingface/transformers/blob/5eddc12edfaf8cafde8c9bae4ccb12f8a139b4f9/LICENSE) | required 5.15.0; original embedding runtime not preserved |
| [numpy-LICENSE.txt](numpy-LICENSE.txt) | [numpy/numpy `cf5aa60b7e79`](https://github.com/numpy/numpy/blob/cf5aa60b7e795efc0be703cace2821fe8fd44e54/LICENSE.txt) | reference-only: unpinned |
| [pandas-LICENSE.txt](pandas-LICENSE.txt) | [pandas-dev/pandas `4d8ee2e42a58`](https://github.com/pandas-dev/pandas/blob/4d8ee2e42a5864125ffbb9afe036f7b298110964/LICENSE) | reference-only: unpinned |
| [scipy-LICENSE.txt](scipy-LICENSE.txt) | [scipy/scipy `d5d45654187c`](https://github.com/scipy/scipy/blob/d5d45654187cdbf0109f8761c2c99b6c5a2a74fd/LICENSE.txt) | reference-only: unpinned |
| [scikit-learn-LICENSE.txt](scikit-learn-LICENSE.txt) | [scikit-learn/scikit-learn `ffbf5196e423`](https://github.com/scikit-learn/scikit-learn/blob/ffbf5196e423409f354d11544522d316b825188e/COPYING) | reference-only: unpinned |
| [networkx-LICENSE.txt](networkx-LICENSE.txt) | [networkx/networkx `abf2feda2252`](https://github.com/networkx/networkx/blob/abf2feda2252ee4e0d5a66964addf179225dfe76/LICENSE.txt) | reference-only: unpinned |
| [matplotlib-LICENSE.txt](matplotlib-LICENSE.txt) | [matplotlib/matplotlib `98dd0357aa1e`](https://github.com/matplotlib/matplotlib/blob/98dd0357aa1e6b9fd87148234d34c129167bb1cb/LICENSE/LICENSE) | reference-only: unpinned; additional bundled-component notices not included |
| [seaborn-LICENSE.txt](seaborn-LICENSE.txt) | [mwaskom/seaborn `c1cb28783585`](https://github.com/mwaskom/seaborn/blob/c1cb287835852406691aed60ea0c62710f2c6b18/LICENSE.md) | reference-only: unpinned |
| [safetensors-LICENSE.txt](safetensors-LICENSE.txt) | [safetensors/safetensors `e246a2560645`](https://github.com/safetensors/safetensors/blob/e246a2560645b7525f5775669ed816eb57c5bcc8/LICENSE) | reference-only: unpinned |
| [tqdm-LICENSE.txt](tqdm-LICENSE.txt) | [tqdm/tqdm `9cf5a12b1f95`](https://github.com/tqdm/tqdm/blob/9cf5a12b1f955468a17f0ba3c59092b23e4258ac/LICENCE) | reference-only: unpinned; coverage is file-specific |

`MPL-2.0.txt` is the full Mozilla Public License 2.0 from
[Mozilla](https://www.mozilla.org/media/MPL/2.0/index.txt), included alongside
tqdm's component notices. No package source is relicensed by this directory.

Confirm exact installed artifacts and retain any additional LICENSE,
NOTICE, bundled-library, or file-level notices before distributing an
environment or executable. See [licensing scope](../docs/LICENSING.md).
