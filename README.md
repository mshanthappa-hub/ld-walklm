# ogbn-products leaderboard experiment

Reproducibility and submission evidence for a 10-seed `ogbn-products`
node-classification experiment using language-derived features with SAGN/SLE
and Correct & Smooth (C&S).

## Acknowledgements

Special thanks to Junhao Shen and Rishi Puri for their guidance on this project.


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

The point test mean is 0.001284 (0.1284 percentage points) above the 0.8737
reference result recorded from the OGB leaderboard on 2026-09-04. It is not an
official rank. The final test was evaluated after the 27 validation-only C&S
records and selected configuration were committed at `8320228`; the guarded
result was then recorded at `c195821`. The historical v1 experiment informed
the fixed v2 method, so that history must accompany any eligibility discussion.
See the
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
- [`results/v2/validation_grid/locked_cs_config.json`](results/v2/validation_grid/locked_cs_config.json): validation-selected C&S configuration and per-seed validation scores.
- [`results/v2/final_test/final_test.json`](results/v2/final_test/final_test.json): guarded one-time final test result.
- [`results/v2/final_test/TEST_ACCESS_STARTED`](results/v2/final_test/TEST_ACCESS_STARTED): durable test-access audit marker.

## Checkpoints and logs

The primary v2 artifact set is preserved in the
[validation-locked 10-seed v2 release](https://github.com/MeghanaShanthappa/ogbn-products-leaderboard_walklm/releases/tag/ogbn-products-walklm-v2-10seed-20260904).
It contains all ten seed `10..19` stage-3 tensors (460,419,782 bytes each),
portable SHA-256 checksums, a size manifest, the recorded environment, and a
compressed bundle of the per-seed, validation-grid, and final-test logs. The
approximately 4.6 GB tensors are linked rather than duplicated in Git.

The archived v1 tensors and logs remain available in the
[exploratory v1 release](https://github.com/MeghanaShanthappa/WalkLM/releases/tag/ogbn-products-sagn-10seed-20260903).

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

V2 establishes a forward test-isolated selection boundary. It does not erase
that EmbeddingGemma and the fixed pipeline were informed by the earlier v1
test-aware exploration. The repository therefore presents 87.4984% as a
prospective Top-1 result, preserves the history, and does not claim an official
rank before OGB accepts the submission.
