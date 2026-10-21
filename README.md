<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# LD + frozen EmbeddingGemma walk-text embeddings (WalkLM-style) + SAGN/SLE + C&S on ogbn-products

Reproducibility and submission evidence for a 10-seed `ogbn-products`
node-classification experiment using language-derived features with SAGN/SLE
and Correct & Smooth (C&S).

## Authors

- Meghana Shanthappa (NVIDIA)
- Junhao Shen (NVIDIA)
- Rishi Puri (NVIDIA)

## License and contribution policy

NVIDIA-authored source and documentation use the [MIT license](LICENSE).
Third-party components retain their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES/](LICENSES/README.md).
Models, datasets, text, and generated feature/checkpoint artifacts are not
automatically licensed under MIT; see [licensing scope](docs/LICENSING.md).
External contributions are not currently accepted; see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Walk-text representation

For every product node, the generator samples four directed random walks of
length four and renders visited nodes with the GLEM title-and-description text
distributed through the PyG TAGDataset download path. The text is encoded with
the pretrained `google/embeddinggemma-300m` model. Attention-mask mean pooling
and L2 normalization produce one 768-dimensional vector per product.

EmbeddingGemma is frozen: it is **not fine-tuned on the walk corpus**, unlike
the language-model fine-tuning proposed by WalkLM. The phrase *WalkLM-style*
refers only to rendering graph walks as text before encoding. The 160-token
limit means that the first rendered walk consumes the available encoder input;
walks two through four do not reach the encoder. Four walks are still generated
to reconstruct the recorded artifact exactly.

The walk-text vector is concatenated with 768-dimensional LD BERT features and
100-dimensional raw OGB features, yielding a 1,636-dimensional SAGN/SLE input.
The complete pipeline obtains **87.4984% ± 0.0490%** test accuracy. V2 did not
include a walk-text-only or feature-stack ablation, so the difference from
prior leaderboard entries cannot be attributed to this component alone.

> [!IMPORTANT]
>
> The validation-locked v2 result is **87.4984% ± 0.0490% test accuracy**.
> V2 prevents any test-driven choices after its
> committed protocol boundary; the earlier v1 exploration remains disclosed in
> [TUNING_AND_PROTOCOL.md](TUNING_AND_PROTOCOL.md).


## Primary v2 result

Official OGB split, new seeds `10..19`. Values are accuracy mean ± unbiased
sample standard deviation (`ddof=1`) across 10 independent runs.

| Evaluation | Validation | Test |
|---|---:|---:|
| Stage 3, before C&S | 0.939204 ± 0.000522 | Not evaluated |
| Validation-selected C&S | **0.940584 ± 0.000433** | **0.874984 ± 0.000490** |

Percentage form:

- Validation: **94.0584% ± 0.0433%**
- Test: **87.4984% ± 0.0490%**

The test point estimate is 0.001284 (0.1284 percentage points) above the
`GLEM+GIANT+SAGN+SCR` public OGB entry of 0.8737 ± 0.0006 recorded on
2026-09-04. The other directly discussed external-data comparator is
`LD+GIANT+SAGN+SCR`, reported at 0.8718 ± 0.0004 with 110,636,896
parameters. These are numerical comparisons; official status is determined by
OGB. The final test was evaluated only after the 27 validation records and lock
were committed at `8320228`; the guarded result was recorded at `c195821`.

The selected C&S setting beat the runner-up by 0.0000762912 mean validation
accuracy, exactly three validation nodes per seed on average (30 additional
correct predictions summed over ten seeds). The historical v1 experiment
informed the frozen v2 method and remains disclosed. See the
[official leaderboard](https://ogb.stanford.edu/docs/leader_nodeprop/),
[submission requirements](https://ogb.stanford.edu/docs/leader_overview/), and
[experimental protocol](https://ogb.stanford.edu/docs/leader_rules/).

### Archived v1 exploratory result

| Evaluation | Train | Validation | Test |
|---|---:|---:|---:|
| Stage 3, before C&S | 0.958059 ± 0.000949 | 0.938901 ± 0.000271 | 0.872624 ± 0.000492 |
| After test-aware C&S exploration | 0.984537 ± 0.000081 | 0.940010 ± 0.000299 | 0.874753 ± 0.000487 |

## Method

```text
LD released BERT hidden states       768 dimensions
Frozen EmbeddingGemma walk-text features (WalkLM-style)  768 dimensions
Raw OGB node features                100 dimensions
                                      ----------------
Combined input                      1636 dimensions
        |
        v
SAGN/SLE multi-stage training (stages 0-3)
        |
        v
Correct & Smooth
```

External data: **Yes**. The feature stack includes released LD language-model
features, GLEM title-and-description text, and frozen EmbeddingGemma
walk-text features.

Implementations used:

- [WalkLM paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2ac879d1865475a7abc8dfc7a9c15c27-Abstract-Conference.html)
- [LD](https://github.com/MIRALab-USTC/LD)
- [SAGN_with_SLE](https://github.com/skepsun/SAGN_with_SLE)
- [Correct and Smooth](https://arxiv.org/abs/2010.13993)

Exact source revisions, the recovered integration patch and generation command,
the artifact boundary, and the protocol audit are documented in
[`PROVENANCE.md`](PROVENANCE.md) and [`TUNING_AND_PROTOCOL.md`](TUNING_AND_PROTOCOL.md).

## Dataset

| Item | Value |
|---|---:|
| Dataset | `ogbn-products` |
| Nodes | 2,449,029 |
| Edges after preprocessing | 123,718,280 |
| Train nodes | 196,615 |
| Validation nodes | 39,323 |
| Test nodes | 2,213,091 |
| Classes | 47 |
| OGB version | 1.3.6 |

## Training configuration

| Parameter | Value |
|---|---:|
| Seeds | 10 through 19 (v2); 0 through 9 (archived v1) |
| SAGN hops (`K`) | 2 |
| Label hops (`label-K`) | 9 |
| Hidden size | 512 |
| Attention heads | 1 |
| Epochs by stage | 300, 100, 100, 100 |
| Pseudo-label thresholds | 0.95, 0.95, 0.999 |
| Learning rate | 0.001 |
| Batch size | 50,000 |
| Evaluation batch size | 100,000 |
| Dropout / attention / input / label | 0.5 / 0.4 / 0.2 / 0.5 |
| Weight decay | 0 |
| Aggregation chunks | 4 |

Validation-selected v2 C&S parameters:

| Parameter | Value |
|---|---:|
| Correction layers | 50 |
| Smoothing layers | 50 |
| Correction alpha | 0.35 |
| Smoothing alpha | 0.60 |
| Correction adjacency | DAD |
| Smoothing adjacency | DAD |
| Scale | 20 |

The primary machine-readable result is the committed validation lock and final
JSON under [`results/v2/`](results/v2/). The archived v1 configuration remains
in [`configs/experiment.yaml`](configs/experiment.yaml).

## Repository contents

- [`PROVENANCE.md`](PROVENANCE.md): pinned revisions, feature origin, and the
  explicitly documented reproducibility boundary.
- [`integration/0001-Add-LD-and-EmbeddingGemma-feature-integration.patch`](integration/0001-Add-LD-and-EmbeddingGemma-feature-integration.patch):
  validated patch for the pinned SAGN source tree.
- [`artifacts/input_manifest.tsv`](artifacts/input_manifest.tsv): expected
  external input tensor metadata and checksum status.
- [`scripts/apply_sagn_patch.sh`](scripts/apply_sagn_patch.sh): guarded,
  idempotent patch installer.
- [`scripts/verify_input_artifacts.py`](scripts/verify_input_artifacts.py):
  shape, dtype, byte-size, and SHA-256 verifier for the two input tensors.
- [`scripts/generate_embeddinggemma_features.py`](scripts/generate_embeddinggemma_features.py):
  label-free reconstruction of the recorded EmbeddingGemma walk-text settings.
- [`results/parameter_count.tsv`](results/parameter_count.tsv): conservative
  OGB multi-stage parameter accounting.
- [`requirements-reproduction.txt`](requirements-reproduction.txt): Python
  dependencies, with the recorded PyTorch build documented separately.
- [`results/v1/cs_per_seed.tsv`](results/v1/cs_per_seed.tsv): evaluator values for
  every seed before and after C&S.
- [`results/v1/summary.tsv`](results/v1/summary.tsv): aggregate means and unbiased
  sample standard deviations.
- [`results/v1/environment.txt`](results/v1/environment.txt): recorded machine and
  package versions.
- [`results/v1/stage3_manifest.tsv`](results/v1/stage3_manifest.tsv): final
  checkpoint sizes and timestamps.
- [`results/v1/stage3_sha256.txt`](results/v1/stage3_sha256.txt): SHA-256 checksums
  for the 10 final checkpoints.
- [`scripts/post_process_detailed.py`](scripts/post_process_detailed.py):
  evaluator wrapper that sorts checkpoints numerically and reports unbiased
  standard deviation.
- [`scripts/run_10_seeds.sh`](scripts/run_10_seeds.sh): restart-safe
  multi-seed training driver.
- [`scripts/run_cs_detailed.sh`](scripts/run_cs_detailed.sh): archived-v1 C&S
  invocation matching the recorded exploratory experiment.
- [`scripts/download_input_tensors.sh`](scripts/download_input_tensors.sh):
  download, join, and checksum-verify the split EmbeddingGemma input.
- [`scripts/reproduce_v2_from_release.sh`](scripts/reproduce_v2_from_release.sh):
  download and verify the released predictions, reproduce the 27-point
  validation grid, and compare the locked validation and final-test records.
- [`scripts/compare_embeddinggemma_subset.py`](scripts/compare_embeddinggemma_subset.py):
  deterministic prefix comparison for the reconstruction script.
- [`results/reconstruction/`](results/reconstruction/): exact 128-row prefix
  comparison/environment plus the successful v2-from-release log and checksum.
- [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md): remaining steps for an
  official OGB submission.
- [`report/technical_report.pdf`](report/technical_report.pdf): concise method,
  result, reproducibility, and protocol report.
- [`TUNING_AND_PROTOCOL.md`](TUNING_AND_PROTOCOL.md): full tuning disclosure
  and the controlling leaderboard-eligibility status.
- [`SUBMISSION.md`](SUBMISSION.md): form-ready metadata and remaining actions.
- [`scripts/tune_cs_validation_only.sh`](scripts/tune_cs_validation_only.sh):
  validation-only C&S grid and auditable configuration lock for future work.
- [`protocol/v2_pre_registration.yaml`](protocol/v2_pre_registration.yaml):
  committed seeds, training settings, C&S grid, selection rule, and test policy.
- [`scripts/run_v2_validation_only.sh`](scripts/run_v2_validation_only.sh):
  end-to-end v2 training and C&S selection without test-label evaluation.
- [`scripts/run_final_test_once.sh`](scripts/run_final_test_once.sh): guarded
  final evaluator that requires the validation records and lock to be committed.
- [`results/v2/validation_grid/locked_cs_config.json`](results/v2/validation_grid/locked_cs_config.json): validation-selected C&S configuration and per-seed validation scores.
- [`results/v2/final_test/final_test.json`](results/v2/final_test/final_test.json): guarded one-time final test result.
- [`results/v2/final_test/TEST_ACCESS_STARTED`](results/v2/final_test/TEST_ACCESS_STARTED): durable test-access audit marker.

## Checkpoints and logs

The primary v2 artifact set is preserved in the
[validation-locked 10-seed v2 release](https://github.com/MeghanaShanthappa/ld-walklm/releases/tag/ogbn-products-walklm-v2-10seed-20260904).
It contains all ten seed `10..19` stage-3 tensors (460,419,782 bytes each),
portable SHA-256 checksums, a size manifest, the recorded environment, and a
compressed bundle of the per-seed, validation-grid, and final-test logs. The
approximately 4.6 GB tensors are linked rather than duplicated in Git.

The archived v1 small provenance artifacts are stored under
[`artifacts/v1/`](artifacts/v1/). All 36 large v1 release assets were mirrored
byte-for-byte into the [archived v1 release](https://github.com/MeghanaShanthappa/ld-walklm/releases/tag/ogbn-products-walklm-v1-exploratory-20260903).
The superseded packaging repository is now private.

## Obtaining the input tensors

The LD tensor is `products_sagn/hidden_state.pt` inside
`bert-base-uncased.zip` from the [LD feature release](https://drive.google.com/drive/folders/109GHk4dtDf0hX_kQLDeMfnvz0Bo5aq_k?usp=drive_link)
(Google Drive file ID `10HHIgaRPgWyVZAX9_FDeOk5cHiOHO2Mv`). Its expected
shape is `2449029 x 768`, dtype is `float32`, size is 7,523,417,875 bytes,
and SHA-256 is
`fb1309cb73d4b1199b3eef342d6d3dc7ce2d2e38c57d50543fd8da0c8bbd6ec7`.

The 8.5 GB EmbeddingGemma tensor is mirrored in five sub-2-GiB parts in the
[v2 artifact release](https://github.com/MeghanaShanthappa/ld-walklm/releases/tag/ogbn-products-walklm-v2-10seed-20260904).
`scripts/download_input_tensors.sh` downloads and verifies each part, joins
them, and verifies the complete reconstructed file.
Its complete-file SHA-256 is
`fef45b647df5155bf17505508cb08ee5a164ad9b33b69910ad1b437109721c90`.
`google/embeddinggemma-300m` is gated; reconstruction requires accepting its
license and authenticating with Hugging Face. The reconstruction pins revision
`57c266a7`. Only load tensors obtained from trusted sources.

## Reconstructing EmbeddingGemma features

The label-free script is a reconstruction written after the original run, not
the original runtime file. It implements the recovered semantic settings.
On a high-memory host with the OGB data and product text available:

```bash
export DATA_DIR=/path/to/OGB
export PYTHON_BIN=/path/to/python
export OUTPUT=/path/to/walklm_products_2449029_embeddinggemma300m_walks4_len4_mean_norm_embeddings.pt
bash scripts/run_embeddinggemma_generation.sh
```

A deterministic 128-node prefix reconstructed with the pinned settings is
exactly equal to the corresponding archived rows: mean cosine similarity 1.0,
mean L2 difference 0.0, and maximum absolute difference 0.0. The comparison
and environment metadata are committed under `results/reconstruction/`.
The script prints a SHA-256 and whether the serialized output matches the
archived container checksum. See `PROVENANCE.md` for version limitations.

## Reproducing training

Start from the exact SAGN base and apply the packaged integration patch:

```bash
git clone https://github.com/skepsun/SAGN_with_SLE.git
git -C SAGN_with_SLE checkout 08fe607ed7f31ca39740a44ce2f139565de3c595
bash scripts/apply_sagn_patch.sh /path/to/SAGN_with_SLE
```

Install `requirements-reproduction.txt` plus a PyTorch build suitable for
the target GPU, then point the runner at the two trusted feature tensors:

```bash
export SAGN_ROOT=/path/to/SAGN_with_SLE
export PYTHON_BIN=/path/to/python
export DATA_DIR=/path/to/OGB
export PROBS_DIR=/path/to/intermediate_outputs
export LD_FEATURE_PATH=/path/to/products_sagn/hidden_state.pt
export WALKLM_FEATURE_PATH=/path/to/walklm_products_2449029_embeddinggemma300m_walks4_len4_mean_norm_embeddings.pt

bash scripts/run_10_seeds.sh
```

Validate and hash those tensors with `scripts/verify_input_artifacts.py`
before launching training.

## Archived v1 post-processing

`scripts/run_cs_detailed.sh`, `results/v1/cs_per_seed.tsv`, and
`results/v1/summary.tsv` are test-aware v1 artifacts. They do not reproduce the
submitted v2 number.

## Reproducing v2 from the released predictions

The v2 aggregate can be reproduced without retraining. This workflow downloads
the ten released stage-3 tensors, verifies the checked-in SHA-256 values,
recomputes all 27 validation-only C&S candidates, verifies that the new lock
matches the committed lock, and writes a separate reproduction of the final
metric:

```bash
export SAGN_ROOT=/path/to/patched/SAGN_with_SLE
export PYTHON_BIN=/path/to/python
export DATA_DIR=/path/to/OGB
export REPRO_ROOT=/path/to/fresh/reproduction

bash scripts/reproduce_v2_from_release.sh
```

## Validation-locked v2 workflow

The v2 protocol was committed before execution and uses new seeds `10..19`, a
dedicated prediction directory, validation-only training evaluation, and a
frozen 27-candidate C&S grid. All candidate records report
`test_metrics_accessed: false`. The selection rule chose correction alpha
0.35, smoothing alpha 0.60, and scale 20.

The complete validation grid and lock were committed at `8320228` before test
access. The guarded final evaluator wrote its durable access marker before
loading the dataset and recorded the one-time final result at `c195821`. The
lock digest matches the digest in both the marker and final JSON.

To reproduce the validation-only run from a fresh prediction directory:

```bash
export SAGN_ROOT=/path/to/SAGN_with_SLE
export PYTHON_BIN=/path/to/python
export DATA_DIR=/path/to/OGB
export V2_PROBS_DIR=/path/to/new_v2_intermediate_outputs
export LD_FEATURE_PATH=/path/to/products_sagn/hidden_state.pt
export WALKLM_FEATURE_PATH=/path/to/walklm_embeddinggemma.pt

bash scripts/run_v2_validation_only.sh
```

Do not rerun the guarded final evaluator as part of model development. The
committed `final_test.json` is the authoritative one-time v2 test record.

## Reproducibility and submission status

The v2 validation grid, selected lock, access marker, final result, checkpoint
hashes, size manifest, and environment record are preserved in Git. The public
v2 release preserves all ten final tensors and the complete log bundle. The
submission record includes a monitored contact email.

A fresh execution from the public release verified all ten checkpoint hashes,
recomputed all 27 validation candidates and the lock exactly, and reproduced
the final test result `0.8749843545 ± 0.0004904255`. Its compressed log and
SHA-256 are committed under `results/reconstruction/` at `c2dd7fa`.

V2 establishes a forward test-isolated selection boundary. It does not erase
that EmbeddingGemma and the fixed pipeline were informed by the earlier v1
test-aware exploration. The repository reports the measured result, preserves
the history, and leaves official ranking to OGB's submission review.
