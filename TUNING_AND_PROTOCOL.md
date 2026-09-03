# Tuning and protocol disclosure

## Status

The recorded `0.8747530490 ± 0.0004873863` test result is **exploratory and
must not be submitted as a protocol-compliant OGB result**. The recovered
experiment history shows that OGB test accuracy was inspected while selecting
the language model and C&S settings. OGB permits test labels only for final
evaluation.

This disclosure is intentionally stricter than the result README. It is the
controlling document if any other file is ambiguous.

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
memory workarounds rather than completed candidates.

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

## Label use audit

- Training labels are used for SAGN/SLE and as the only labels passed to C&S.
- Validation labels are used for early stopping/model selection.
- The code path does not pass validation or test labels into SAGN/SLE features
  or C&S propagation.
- The protocol problem is hyperparameter/model selection after viewing test
  accuracy, not direct test-label input to the predictor.

## Validation-only tooling

`scripts/tune_cs_validation_only.sh` runs a frozen C&S grid through
`scripts/post_process_validation_only.py`, which evaluates and prints only the
validation split. `scripts/select_cs_by_validation.py` records an auditable
lock before any final test evaluation.

This workflow is useful for future experiments. It does not retroactively
make the current experiment untouched. Before submitting this result, disclose
the history to OGB and request a ruling, or develop a new method/configuration
whose selection follows a clean, pre-registered validation-only protocol.

## Locked v2 execution

`protocol/v2_pre_registration.yaml` freezes seeds `10..19`, the full training
configuration, the 27-member C&S grid, its validation-only selection rule, and
the final-test policy before execution. `scripts/run_v2_validation_only.sh`
uses a replacement training evaluator that masks test labels before
preprocessing and never evaluates them. It
generates predictions for all nodes, which is required for later propagation,
but train/validation are the only scored splits.

After all 10 seeds finish, the driver runs the validation-only C&S grid and
writes `results/v2/validation_grid/locked_cs_config.json`. The candidate JSON
records and lock must be reviewed and committed. Only then may
`scripts/run_final_test_once.sh` be invoked. That script evaluates only the
post-C&S test predictions and creates `TEST_ACCESS_STARTED` before loading the
dataset, so a failed or interrupted attempt cannot be silently repeated.

This is a forward-clean execution boundary, not a claim that the historical
choice of EmbeddingGemma was uninformed by v1 test results. That caveat remains
mandatory in any OGB eligibility discussion.