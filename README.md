# ogbn-products leaderboard experiment

Reproducibility and submission evidence for a 10-seed `ogbn-products`
node-classification experiment using language-derived features with SAGN/SLE
and Correct & Smooth (C&S).

## Final result

Official OGB split, seeds `0..9`. Values are accuracy mean ± unbiased sample
standard deviation (`ddof=1`) across 10 independent runs.

| Evaluation | Train | Validation | Test |
|---|---:|---:|---:|
| Stage 3, before C&S | 0.958059 ± 0.000949 | 0.938901 ± 0.000271 | 0.872624 ± 0.000492 |
| After tuned C&S | **0.984537 ± 0.000081** | **0.940010 ± 0.000299** | **0.874753 ± 0.000487** |

Percentage form:

- Validation: **94.0010% ± 0.0299%**
- Test: **87.4753% ± 0.0487%**

The point test mean is 0.001053 (0.1053 percentage points) above the 0.8737
reference result recorded from the OGB leaderboard on 2026-09-03. This is an
experimental result, not an official rank, until it is submitted and accepted
by OGB. See the
[official leaderboard](https://ogb.stanford.edu/docs/leader_nodeprop/) and
[submission rules](https://ogb.stanford.edu/docs/leader_overview/).

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

## Checkpoints and logs

The verified artifact set is preserved in the
[10-seed WalkLM release](https://github.com/MeghanaShanthappa/WalkLM/releases/tag/ogbn-products-sagn-10seed-20260903).

The release contains all 10 stage-3 tensors (460,419,711 bytes each), their
SHA-256 hashes, per-seed training logs, the raw result table, environment
record, and checkpoint manifest. It is linked instead of duplicated here to
avoid storing another copy of approximately 4.6 GB.

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

## Reproducibility status

The final checkpoints, logs, hashes, environment, evaluator outputs, training
arguments, and C&S arguments are archived. Before an official submission, the
exact Git commit IDs for WalkLM, LD, SAGN_with_SLE, and the feature-generation
code used to build the 1,636-dimensional input should also be recorded.
