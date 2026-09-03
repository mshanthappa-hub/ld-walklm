# Source and feature provenance

This document distinguishes what is archived and verified from what still
needs to be recovered before claiming end-to-end reproducibility.

## Source revisions

| Component | Revision | Role |
|---|---|---|
| SAGN_with_SLE | `08fe607ed7f31ca39740a44ce2f139565de3c595` | Clean upstream base for the integration patch |
| WalkLM | `bff59aae01855a21cbf10fde932ccf0ac7ff9aab` | Repository snapshot available when this evidence package was assembled |
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

The two external tensor checksums were not captured in the shared archive.
Their expected metadata is recorded in `artifacts/input_manifest.tsv` as
`PENDING`. On a host that still has the tensors, generate the completed
manifest with:

```bash
python scripts/verify_input_artifacts.py \
  --ld /path/to/products_sagn/hidden_state.pt \
  --walklm /path/to/walklm_products_2449029_embeddinggemma300m_walks4_len4_mean_norm_embeddings.pt \
  --output artifacts/input_manifest.tsv
```

Only run that command on trusted tensor files. PyTorch tensor containers may
use pickle-backed deserialization.

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
outputs, training arguments, and post-processing arguments are archived and
verified. The integration needed to consume the two external feature tensors
is now preserved here.

Two items remain before an end-to-end feature-generation claim is warranted:

- record the SHA-256 hashes of the LD and EmbeddingGemma tensors;
- recover and preserve the exact EmbeddingGemma feature-generation command
  or script (the repository snapshot alone does not prove that command).

External data must be declared as **Yes** for any leaderboard submission.
