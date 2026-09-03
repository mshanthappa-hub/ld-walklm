# ogbn-products leaderboard experiment

Reproducibility and submission evidence for a 10-seed `ogbn-products`
node-classification experiment using language-derived features with SAGN/SLE
and Correct & Smooth (C&S).


> [!IMPORTANT]
>
> The recorded 87.4753% test result is exploratory, not currently eligible
> for OGB submission, because test scores were viewed during method and C&S
> selection. See [TUNING_AND_PROTOCOL.md](TUNING_AND_PROTOCOL.md) and the
> [submission record](SUBMISSION.md).


## Final result

Official OGB split, seeds `0..9`. Values are accuracy mean ± unbiased sample
standard deviation (`ddof=1`) across 10 independent runs.

| Evaluation | Train | Validation | Test |
|---|---:|---:|---:|
| Stage 3, before C&S | 0.958059 ± 0.000949 | 0.938901 ± 0.000271 | 0.872624 ± 0.000492 |
| After selected C&S (exploratory) | **0.984537 ± 0.000081** | **0.940010 ± 0.000299** | **0.874753 ± 0.000487** |

Percentage form:

- Validation: **94.0010% ± 0.0299%**
- Test: **87.4753% ± 0.0487%**

The point test mean is 0.001053 (0.1053 percentage points) above the 0.8737
reference result recorded from the OGB leaderboard on 2026-09-03. It is not an
official rank. More importantly, the recovered history shows test-aware method
selection, so this result must not be submitted as protocol-compliant without
an explicit OGB ruling or a clean experiment. See the
[official leaderboard](https://ogb.stanford.edu/docs/leader_nodeprop/),
[submission requirements](https://ogb.stanford.edu/docs/leader_overview/), and
[experimental protocol](https://ogb.stanford.edu/docs/leader_rules/).

## Method

```text
LD released BERT hidden states       768 dimensions
EmbeddingGemma-300M WalkLM features  768 dimensions
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
features and WalkLM features generated from product text.

Implementations used:

- [WalkLM experiment code](https://github.com/MeghanaShanthappa/WalkLM)
- [LD](https://github.com/MIRALab-USTC/LD)
- [SAGN_with_SLE](https://github.com/skepsun/SAGN_with_SLE)

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
| Seeds | 0 through 9 |
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

Tuned C&S parameters:

| Parameter | Value |
|---|---:|
| Correction layers | 50 |
| Smoothing layers | 50 |
| Correction alpha | 0.30 |
| Smoothing alpha | 0.65 |
| Correction adjacency | DAD |
| Smoothing adjacency | DAD |
| Scale | 25 |

The machine-readable configuration is in
[`configs/experiment.yaml`](configs/experiment.yaml).

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
  preserved label-free generator for the exact EmbeddingGemma WalkLM settings.
- [`results/parameter_count.tsv`](results/parameter_count.tsv): conservative
  OGB multi-stage parameter accounting.
- [`requirements-reproduction.txt`](requirements-reproduction.txt): Python
  dependencies, with the recorded PyTorch build documented separately.
- [`results/cs_per_seed.tsv`](results/cs_per_seed.tsv): evaluator values for
  every seed before and after C&S.
- [`results/summary.tsv`](results/summary.tsv): aggregate means and unbiased
  sample standard deviations.
- [`results/environment.txt`](results/environment.txt): recorded machine and
  package versions.
- [`results/stage3_manifest.tsv`](results/stage3_manifest.tsv): final
  checkpoint sizes and timestamps.
- [`results/stage3_sha256.txt`](results/stage3_sha256.txt): SHA-256 checksums
  for the 10 final checkpoints.
- [`scripts/post_process_detailed.py`](scripts/post_process_detailed.py):
  evaluator wrapper that sorts checkpoints numerically and reports unbiased
  standard deviation.
- [`scripts/run_10_seeds.sh`](scripts/run_10_seeds.sh): restart-safe
  multi-seed training driver.
- [`scripts/run_cs_detailed.sh`](scripts/run_cs_detailed.sh): portable C&S
  invocation matching the recorded experiment.
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

## Checkpoints and logs

The verified artifact set is preserved in the
[10-seed WalkLM release](https://github.com/MeghanaShanthappa/WalkLM/releases/tag/ogbn-products-sagn-10seed-20260903).

The release contains all 10 stage-3 tensors (460,419,711 bytes each), their
SHA-256 hashes, per-seed training logs, the raw result table, environment
record, and checkpoint manifest. It is linked instead of duplicated here to
avoid storing another copy of approximately 4.6 GB.

## Reproducing EmbeddingGemma features

The exact recovered feature settings are preserved in a standalone script.
On a high-memory host with the OGB data and product text available:

```bash
export DATA_DIR=/path/to/OGB
export PYTHON_BIN=/path/to/python
export OUTPUT=/path/to/walklm_products_2449029_embeddinggemma300m_walks4_len4_mean_norm_embeddings.pt
bash scripts/run_embeddinggemma_generation.sh
```

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

## Reproducing C&S

Install the versions recorded in `results/environment.txt`, clone
`SAGN_with_SLE`, download the stage-3 checkpoint assets, and set:

```bash
export SAGN_ROOT=/path/to/SAGN_with_SLE
export PYTHON_BIN=/path/to/python
export DATA_DIR=/path/to/OGB
export PROBS_DIR=/path/to/intermediate_outputs
export RESULTS_DIR=/path/to/output-results

bash scripts/run_cs_detailed.sh
```

The script requires exactly 10 non-empty stage-3 checkpoints and writes a
fresh per-seed TSV before calculating the mean and unbiased sample standard
deviation.

## Forward validation-only v2 workflow

The committed v2 protocol uses new seeds `10..19`, a dedicated prediction
directory, validation-only training evaluation, and the frozen 27-candidate
C&S grid. It intentionally stops after writing the selected configuration:

```bash
export SAGN_ROOT=/path/to/SAGN_with_SLE
export PYTHON_BIN=/path/to/python
export DATA_DIR=/path/to/OGB
export V2_PROBS_DIR=/path/to/new_v2_intermediate_outputs
export LD_FEATURE_PATH=/path/to/products_sagn/hidden_state.pt
export WALKLM_FEATURE_PATH=/path/to/walklm_embeddinggemma.pt

bash scripts/run_v2_validation_only.sh
```

Review and commit all files under `results/v2/validation_grid/` before running
`scripts/run_final_test_once.sh`. The final script refuses to proceed unless
the 27 candidate records and selected lock are tracked and unchanged, and it
creates a durable marker as soon as test access begins.

This prevents further test-driven choices from the v2 lock forward. It does
not erase the historical v1 test exposure described in
`TUNING_AND_PROTOCOL.md`.

## Reproducibility and submission status

The final checkpoints, logs, checkpoint hashes, environment, evaluator
outputs, training and C&S arguments, pinned revisions, SAGN integration, exact
EmbeddingGemma command, and both external input artifact checksums are
preserved. The verified tensor metadata is recorded in
`artifacts/input_manifest.tsv`.

The 423,228,606-parameter multi-stage count and technical report are included.
However, the recovered test-aware tuning history is a substantive OGB protocol
blocker. The current score is an exploratory research result, not a
submission-ready leaderboard claim. Follow `SUBMISSION.md` and
`TUNING_AND_PROTOCOL.md`; do not omit that disclosure.
