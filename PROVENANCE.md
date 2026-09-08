# Source and feature provenance

This document separates directly preserved facts from reconstructed behavior and
unknown original-runtime details. It does not override the protocol audit in
`TUNING_AND_PROTOCOL.md`.

## Source revisions

| Component | Revision | Role |
|---|---|---|
| SAGN_with_SLE | `08fe607ed7f31ca39740a44ce2f139565de3c595` | Clean upstream base for the integration patch |
| LD | `cf2804b08c1cb4453ca7950d9f20b3d3bc7a43e4` | Pinned upstream source snapshot |
| EmbeddingGemma reconstruction | `google/embeddinggemma-300m`, revision `57c266a7` | Frozen encoder used by the reconstruction script |
| This repository | See Git history and release tag | Integration, reconstruction, protocol, and result evidence |

The former personal WalkLM packaging repository is not a source dependency.
The small checksum sidecar and seed-0 evidence needed from it have been migrated
to `artifacts/v1/`; all 36 large release assets were mirrored byte-for-byte into the public
`ld-walklm` archived v1 release. The superseded packaging repository is now
private.

## Input feature artifacts

The 1,636-dimensional input was concatenated in this exact column order:

1. LD released BERT hidden state: `[2449029, 768]`, `float32`.
2. Frozen EmbeddingGemma walk-text tensor: `[2449029, 768]`, `float32`.
3. Raw OGB feature matrix: `[2449029, 100]`, converted to `float32`.

The LD tensor is `products_sagn/hidden_state.pt` inside the official
`bert-base-uncased.zip` feature release from the LD authors (Google Drive file
ID `10HHIgaRPgWyVZAX9_FDeOk5cHiOHO2Mv`). It is 7,523,417,875 bytes and has
SHA-256:

```text
fb1309cb73d4b1199b3eef342d6d3dc7ce2d2e38c57d50543fd8da0c8bbd6ec7
```

The EmbeddingGemma artifact is named
`walklm_products_2449029_embeddinggemma300m_walks4_len4_mean_norm_embeddings.pt`.
It is a serialized dictionary containing `walklm_x` and `raw_x`, is
8,503,030,973 bytes, and has SHA-256:

```text
fef45b647df5155bf17505508cb08ee5a164ad9b33b69910ad1b437109721c90
```

The original sidecar is retained verbatim at
`artifacts/v1/walklm_embedding.sha256`. The tensor is mirrored as five
sub-2-GiB release assets and can be downloaded, joined, and verified with
`scripts/download_input_tensors.sh`.

Reverify trusted copies and regenerate the local manifest with:

```bash
python scripts/verify_input_artifacts.py \
  --ld /path/to/products_sagn/hidden_state.pt \
  --walklm /path/to/walklm_products_2449029_embeddinggemma300m_walks4_len4_mean_norm_embeddings.pt \
  --output artifacts/input_manifest.tsv
```

PyTorch tensor containers may use pickle-backed deserialization. Do not load
untrusted files.

## Walk-text source

`node-text.csv.gz` is the same product text downloaded by the PyG `TAGDataset`
path. It contains the GLEM title-and-description representation for each
product, not an unspecified product raw-text corpus. The recorded Google Drive
file ID is `1I-S176-W4Bm1iPDjQv3hYwQBtxE0v8mt`.

Directed walks use the original `ogbn-products` edge orientation. For every
start node, four walks of length four are sampled with a CPU PyTorch generator
seeded to 42. Each visited node contributes at most 160 characters and nodes
are separated by the string ` cites `. The concatenated text is tokenized to a
maximum of 160 tokens. In the archived behavior, the first rendered walk fills
the encoder input, so walks two through four are generated but truncated before
they reach EmbeddingGemma.

## Original recovered command

The command recorded for the successful 2026-08-21 generation was:

```bash
PYTHONPATH=. python -u walklm.py \
  --dataset ogbn-products \
  --hf_model google/embeddinggemma-300m \
  --encoder_type auto_model --pooling mean \
  --normalize_embeddings --mlm_epochs 0 --epochs 1 \
  --batch_size 4 --num_walks 4 --walk_length 4 \
  --max_length 160 --max_text_chars 160 \
  --use_raw_features --save_embeddings
```

The `walklm.py` runtime modifications that accepted `--encoder_type`,
`--pooling`, and `--normalize_embeddings` were not committed. Therefore this
command is historical evidence, not an executable command against a pinned
source file. `--mlm_epochs 0` means that EmbeddingGemma was not fine-tuned.
`--epochs 1` trained each downstream diagnostic classifier for one epoch; it
did not update EmbeddingGemma and did not affect the saved tensor.

## Reconstruction status

`scripts/generate_embeddinggemma_features.py` was written on 2026-09-03, after
the original generation. It is explicitly a label-free reconstruction of the
recorded behavior, not the preserved original program. It uses `AutoModel`,
attention-masked mean pooling of the final hidden state, L2 normalization, and
float32 output. The reconstruction pins model revision `57c266a7` and writes
its software, model, dtype, and device metadata next to the tensor.

The exact original `transformers` version, PyTorch version, internal model
compute dtype, GPU model, and resolved Hugging Face revision were not captured.
They must be reported as unknown rather than inferred from the later SAGN
environment. The archived tensor checksum is therefore authoritative. A future
reconstruction can test semantic similarity but need not reproduce a
byte-identical PyTorch container.

A deterministic prefix can be regenerated with `--max-nodes` and compared to
the matching archived rows using `scripts/compare_embeddinggemma_subset.py`.
The resulting cosine-similarity JSON will be committed after the GPU check is
completed.

## Recovered SAGN integration

The patch in `integration/` applies to the pinned SAGN revision. It adds:

- an environment-variable-driven, shape-checked feature loader;
- concatenation in `[LD, frozen EmbeddingGemma walk-text, raw]` order;
- CPU-safe indexing for minibatches when propagated features live in RAM;
- DGL `copy_src` and `copy_edge` compatibility aliases;
- fixes for chunk numbering, NumPy conversion, output allocation, and
  temporary `.npy` cleanup.

Apply it with `scripts/apply_sagn_patch.sh` to a clean checkout of the pinned
SAGN commit.

## Reproducibility boundary

Preserved and verified artifacts include the ten final prediction tensors,
their SHA-256 hashes, training and post-processing logs, evaluator outputs,
training arguments, integration patch, two external tensor hashes and sizes,
validation candidate records, selected lock, and guarded final result.

The original modified EmbeddingGemma runtime file and exact generation
environment were not preserved. The provided script reconstructs the recorded
semantics and openly records this limitation.

`TUNING_AND_PROTOCOL.md` documents the test-aware v1 method selection. V2
creates a forward validation-only boundary but does not erase that history.
External data must be declared as **Yes** for any leaderboard submission.
