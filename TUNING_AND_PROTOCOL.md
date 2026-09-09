# Tuning and protocol disclosure

## Status

The primary v2 result is `0.8749843545 ± 0.0004904255` test accuracy. It was
produced with new seeds `10..19`, validation-only training/C&S selection, a
committed configuration lock, and one guarded final test evaluation. It is a
prospective leaderboard entry, not an official rank until OGB accepts it.

The earlier v1 experiment inspected test accuracy while selecting the language
model and C&S settings. V2 creates a forward test-isolated boundary but retains
the already informed EmbeddingGemma pipeline. This history must remain
disclosed in any eligibility discussion. This file is controlling if another
repository file is ambiguous.

## Recovered model exploration

The product-text embedding experiments compared at least the following
encoders/configurations before the full EmbeddingGemma run:

- TinyBERT, DistilBERT, and BERT-base masked-language-model variants;
- `BAAI/bge-base-en-v1.5` with masked mean pooling and normalization;
- `intfloat/e5-base-v2` with masked mean pooling and normalization;
- `Qwen/Qwen3-Embedding-0.6B`;
- `google/embeddinggemma-300m`.

The 725,745-node screening runs reported OGB test-split accuracy and were used
to decide which encoder to scale up. This is test-aware method selection.

The downstream exploration also included different feature combinations and
at least these SAGN values:

| Hyperparameter | Values observed in the experiment history |
|---|---|
| Feature hops (`K`) | 2, 3 |
| Label hops (`label-K`) | 9, 14 |
| Hidden channels | 200, 256, 512 |
| Stage-0 epochs | 300, 500 |
| Later-stage epochs | 100 |
| Batch size | 25,000, 50,000 |

These ranges are disclosed conservatively. Some trials were smoke tests or
memory workarounds rather than completed candidates. No principled reason for
`K=2` was preserved: both `K=2` and `K=3` were explored, and `K=2` was carried
forward from the test-visible v1 work. It is therefore classified as a
test-aware choice rather than a validation-only v2 choice.

Walk construction was also varied during early subset experiments. The
recovered chat transcript and surviving artifact names distinguish completed,
partial, and failed trials as follows:

| Encoder / status | Walks | Walk length | Token limit | Node-text limit | MLM epochs | Preserved outcome |
|---|---:|---:|---:|---:|---:|---|
| TinyBERT, completed subset baseline | 2 | 3 | 256 (then-default) | 128 | 1 | WalkLM-only 0.7284; WalkLM+raw 0.7823 |
| DistilBERT, completed subset screen | 4 | 4 | 160 | 160 | 1 | WalkLM-only 0.8032; WalkLM+raw 0.8283 |
| DistilBERT, completed subset screen | 4 | 4 | 160 | 160 | 2 | WalkLM-only 0.8101; WalkLM+raw 0.8345 |
| BERT-base, completed subset screen | 4 | 4 | 160 | 160 | 1 | WalkLM-only 0.8015; WalkLM+raw 0.8282 |
| DistilBERT, partial subset screen | 4 | 5 | 160 | 128 | 1 | WalkLM-only 0.7893; process ended before WalkLM+raw evaluation |
| DistilBERT, failed memory trial | 4 | 5 | 192 | 160 | 1 | ended after model loading / corpus preparation |
| DistilBERT, failed memory trial | 6 | 4 | 192 | 192 | 1 | ended after model loading / corpus preparation |
| Frozen EmbeddingGemma, full artifact | 4 | 4 | 160 | 160 | 0 | configuration used by the submitted feature tensor |

The nominal `--max_nodes 500000` subset construction produced 725,745 nodes.
These runs exposed test accuracy and informed the final method, so the choices
are classified as v1 test-aware. They were not a systematic controlled grid.
A 4-walk/8-step smoke command failed on missing dependencies before an
experiment ran; memory-safer 6-walk retries and a 256-character variant were
suggested in chat but lack evidence of completed execution, so they are not
listed as tuned candidates.

## Recovered C&S exploration

The upstream starting point was approximately:

```text
correction_alpha = 0.4553513529697225
smoothing_alpha  = 0.7153479868942039
scale            = 15
layers           = 50 / 50
adjacency        = DAD / DAD
```

The archived scripts proposed or executed these grids:

```text
initial grid:
  correction_alpha: [0.35, 0.4553513529697225, 0.55]
  smoothing_alpha:  [0.65, 0.7153479868942039, 0.80]
  scale:            [10, 15, 20]

focused grid:
  correction_alpha: [0.30, 0.35, 0.40]
  smoothing_alpha:  [0.60, 0.65, 0.70]
  scale:            [20, 25, 30]

small grid:
  correction_alpha: [0.30, 0.35, 0.40]
  smoothing_alpha:  [0.65]
  scale:            [20, 25, 30]
```

The preserved partial small-grid log contains complete results for
`(0.30, 0.65, 20)`, `(0.30, 0.65, 25)`, and `(0.30, 0.65, 30)`. The selected
setting was `(0.30, 0.65, 25)`. Contemporary notes explicitly compared and
sorted candidates by test accuracy, so it cannot be represented as a purely
validation-selected setting.

The v2 validation grid is identical to the v1 focused grid. V2 selected
`(0.35, 0.60, 20)` using validation only. The v1 test-selected setting
`(0.30, 0.65, 25)` ranks 11th among the 27 v2 candidates by mean validation
accuracy.

## Label use audit

- Training labels are used for SAGN/SLE and as the only labels passed to C&S.
- Validation labels are used for early stopping/model selection.
- The code path does not pass validation or test labels into SAGN/SLE features
  or C&S propagation.
- The protocol problem is hyperparameter/model selection after viewing test
  accuracy, not direct test-label input to the predictor.

## Validation-only tooling

`scripts/tune_cs_validation_only.sh` runs the frozen C&S grid through
`scripts/post_process_validation_only.py`, which scores and prints only the
validation split. `scripts/select_cs_by_validation.py` applies the pre-committed
selection rule and writes an auditable configuration lock.

The v2 training wrapper masks all 2,213,091 test labels before preprocessing
and reports `Test: nan`. Predictions are still generated for all nodes because
C&S requires graph-wide probabilities, but no test metric is computed during
training or validation selection.

## Completed v2 execution

Seeds `0..9` had already been consumed by test-aware v1 exploration, so v2
preregistered fresh seeds `10..19` and reported all ten without selection.
`protocol/v2_pre_registration.yaml`, committed at `827e8fc`, froze those seeds, the full training configuration, the 27-member C&S grid, the
validation selection rule, and final-test policy before v2 execution. A later
wrapper fix at `24f6978` only handled an upstream post-save NumPy summary error;
it did not change the method or locked hyperparameters.

All 27 candidate records have `test_metrics_accessed: false`. Their committed
content digest is
`7aece8d8b28481cd60cf05ce847ff5759b849c638416b1a4be170a842abf14bf`. The
selection rule chose `(correction_alpha=0.35, smoothing_alpha=0.60, scale=20)`
with 50/50 layers and DAD/DAD adjacency. Across seeds `10..19`:

```text
Before C&S validation: 0.9392035196 ± 0.0005224146
Selected C&S validation: 0.9405843908 ± 0.0004332633
```

The grid and lock were committed at
`8320228eca9e66ff8d7b57080e240d312c0729db`. The guarded final-test invocation
wrote `TEST_ACCESS_STARTED` before dataset loading. Its marker, lock, and final
JSON share lock SHA-256
`3d5f7ed1a2096dff5de4aeb45bbb83b1378c60aa43eeec2e926f206cca299f8d`. The
one-time post-C&S result, committed at
`c195821768703cb1b72132b2906795e8003edf7d`, is:

```text
Final test: 0.8749843545 ± 0.0004904255
```

Independent recalculation reproduces both validation and test aggregates and
confirms that the locked candidate is the actual validation-grid winner. No
v2 test accuracy was computed outside the guarded final invocation according to
the preserved commands, logs, marker, and repository history.

The selected setting exceeds the runner-up, `(0.30, 0.60, 25)`, by
0.0000762912 mean validation accuracy. That is exactly three validation nodes
per seed on average, or 30 additional correct predictions across ten seeds.

A conservative lower bound is 300 v1 test evaluations from the final ten-seed
SAGN training alone: 30 evaluation points per seed across four stages. Encoder
screening, feature-stack trials, and C&S comparisons added further test looks.
The exact total is not recoverable because some exploratory logs were not
retained.

## Artifact release

All ten v2 stage-3 tensors, portable checkpoint hashes, environment metadata,
and logs are published at:
<https://github.com/MeghanaShanthappa/ld-walklm/releases/tag/ogbn-products-walklm-v2-10seed-20260904>

## Interpretation

V2 prevents any further test-driven choice from its committed boundary onward.
It does not erase the historical fact that v1 test results informed the fixed
EmbeddingGemma method. The result should therefore be submitted with this
disclosure and described as prospective until OGB accepts it.
