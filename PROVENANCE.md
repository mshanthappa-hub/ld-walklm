# Source and feature provenance

This document distinguishes what is archived and verified from what remains
external or uncertain; it does not override the protocol audit.

## Source revisions

| Component | Revision | Role |
|---|---|---|
| SAGN_with_SLE | `08fe607ed7f31ca39740a44ce2f139565de3c595` | Clean upstream base for the integration patch |
| WalkLM code base | `bff59aae01855a21cbf10fde932ccf0ac7ff9aab` | Base repository before recovered runtime modifications |
| WalkLM artifact commit | `505f635` | Original LFS chunks, checksum sidecar, and seed-0 log |
| LD | `cf2804b08c1cb4453ca7950d9f20b3d3bc7a43e4` | Pinned upstream source snapshot |

The patch in `integration/` applies to the exact SAGN revision above. It
contains the recovered feature loader and the compatibility changes used by
the successful high-memory run.

## Input feature artifacts

The 1,636-dimensional input was concatenated in this exact column order:

1. LD released BERT hidden state: `[2449029, 768]`, `float32`.
2. EmbeddingGemma-300M WalkLM tensor: `[2449029, 768]`, `float32`.
3. Raw OGB feature matrix: `[2449029, 100]`, converted to `float32`.

The LD tensor came from the official released `products_sagn/hidden_state.pt`
inside `bert-base-uncased.zip` (Google Drive file ID
`10HHIgaRPgWyVZAX9_FDeOk5cHiOHO2Mv`). The WalkLM artifact filename was
`walklm_products_2449029_embeddinggemma300m_walks4_len4_mean_norm_embeddings.pt`.

The original WalkLM SHA-256 sidecar was recovered from commit `505f635`:
`fef45b647df5155bf17505508cb08ee5a164ad9b33b69910ad1b437109721c90`.
The serialized dictionary is 8,503,030,973 bytes and contains both `walklm_x`
and `raw_x`. The LD tensor checksum was not captured and remains `PENDING` in
`artifacts/input_manifest.tsv`. On a host that still has both tensors, generate
the completed manifest with:

```bash
python scripts/verify_input_artifacts.py \
  --ld /path/to/products_sagn/hidden_state.pt \
  --walklm /path/to/walklm_products_2449029_embeddinggemma300m_walks4_len4_mean_norm_embeddings.pt \
  --output artifacts/input_manifest.tsv
```

Only run that command on trusted tensor files. PyTorch tensor containers may
use pickle-backed deserialization.

## Recovered EmbeddingGemma generation

The successful full-graph run used the following semantic configuration:

```text
model: google/embeddinggemma-300m
encoder: AutoModel, no fine-tuning
seed: 42
walks per node: 4
walk length: 4
node text truncation: 160 characters
tokenizer maximum length: 160
pooling: attention-masked mean of the final hidden state
normalization: L2
batch size: 4
```

The exact recovered command is included in `report/technical_report.tex` and
the behavior is preserved without label access in
`scripts/generate_embeddinggemma_features.py`. The base WalkLM commit did not
contain the runtime streaming modifications, and the Hugging Face model
revision was not pinned. Therefore the archived checksum is authoritative;
a future generation can verify semantic reproducibility but may not produce a
byte-identical PyTorch container across model or library revisions.


## Recovered SAGN integration

The patch adds:

- an environment-variable-driven, shape-checked feature loader;
- concatenation in `[LD, WalkLM, raw]` order;
- CPU-safe indexing for minibatches when propagated features live in RAM;
- DGL `copy_src`/`copy_edge` compatibility aliases;
- fixes for chunk numbering, NumPy conversion, output allocation, and
  temporary `.npy` cleanup.

Apply it with `scripts/apply_sagn_patch.sh` to a clean checkout of the pinned
SAGN commit.

## Reproducibility boundary

The ten final prediction tensors, their SHA-256 hashes, logs, evaluator
outputs, training arguments, post-processing arguments, feature integration,
EmbeddingGemma settings, and EmbeddingGemma artifact checksum are preserved.

The remaining artifact gap is the LD `hidden_state.pt` byte size and SHA-256.
The source is identified by its official release file and Google Drive ID.

Separately, `TUNING_AND_PROTOCOL.md` records test-aware selection discovered in
the archived history. That protocol issue prevents the current result from
being represented as submission-ready even after the LD checksum is filled.

External data must be declared as **Yes** for any future leaderboard submission.
