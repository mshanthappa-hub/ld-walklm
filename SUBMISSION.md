# OGB submission record

## Current disposition

**V2 IS THE PRIMARY SUBMISSION CANDIDATE; IT IS NOT YET AN OFFICIAL RESULT.**

V2 prevents test-driven changes after its committed protocol boundary. Earlier
v1 test results informed the frozen encoder and SAGN choices, so the disclosure
below must accompany the submission.

## Form-ready fields

| OGB field | Value |
|---|---|
| OGB version | `1.3.6` |
| Dataset | `ogbn-products` |
| Method (30-character form field) | `LD+EG-Walk+SAGN/SLE+C&S` |
| Full method name | LD + frozen EmbeddingGemma walk-text embeddings (WalkLM-style) + SAGN/SLE + C&S |
| External data | Yes: released LD BERT features, GLEM title-and-description text, and `google/embeddinggemma-300m` revision `57c266a7` |
| Test performance | `0.8749843545,0.0004904255` |
| Validation performance | `0.9405843908,0.0004332633` |
| Runs | 10 independent runs, seeds 10 through 19; none selected or discarded |
| Standard deviation | Unbiased sample standard deviation (`ddof=1`) |
| Primary contact | Meghana Shanthappa (NVIDIA) |
| Primary contact email | `shanthappa.meghana@gmail.com` |
| Coauthors | Junhao Shen (NVIDIA); Rishi Puri (NVIDIA) |
| Code | <https://github.com/MeghanaShanthappa/ld-walklm> |
| Technical report | <https://github.com/MeghanaShanthappa/ld-walklm/blob/main/report/technical_report.pdf> |
| Parameters | 423,228,606 conservative multi-stage total; see `results/parameter_count.tsv` |
| Hardware | NVIDIA RTX PRO 4500 Blackwell, 32,623 MiB GPU memory |
| Officiality | Official implementation of this combination; upstream components are cited third-party implementations |

## Tuned hyperparameters

The asterisk marks the value used for the v2 result. Values labelled
`v1 test-aware` were considered while test accuracy was visible and were
frozen before the v2 boundary.

```text
encoder: [TinyBERT, DistilBERT, BERT-base, BGE-base, E5-base,
          Qwen3-Embedding-0.6B, EmbeddingGemma-300M* (v1 test-aware)]
feature_stack: [raw, WalkLM+raw, LD+WalkLM+raw* (v1 test-aware)]
K: [2* (v1 test-aware), 3]
label_K: [9* (v1 test-aware), 14]
hidden_channels: [200, 256, 512* (v1 test-aware)]
stage_0_epochs: [300* (v1 test-aware), 500]
later_stage_epochs: [100* (v1 test-aware)]
batch_size: [25000, 50000* (v1 test-aware)]
correction_layers: [50*]
smoothing_layers: [50*]
correction_alpha: [0.30 (v1 test-selected), 0.35* (v2 validation), 0.40,
                   0.4553513529697225, 0.55]
smoothing_alpha: [0.60* (v2 validation), 0.65 (v1 test-selected), 0.70,
                  0.7153479868942039, 0.80]
scale: [10, 15, 20* (v2 validation), 25 (v1 test-selected), 30]
C&S adjacency: [DAD/DAD*]
walks_per_node: [2 (completed earlier subset), 4* (completed; v1 test-aware),
                 6 (attempted; OOM)]
walk_length: [3 (completed earlier subset), 4* (completed; v1 test-aware),
              5 (partially evaluated / OOM)]
tokenizer_max_length: [160*, 192 (attempted; OOM),
                       256 (earlier subset default)]
node_text_char_limit: [128 (earlier subset / partial longer-walk run),
                       160*, 192 (attempted; OOM)]
walk_encoder_mlm_epochs: [0* (frozen EmbeddingGemma),
                          1, 2 (test-aware DistilBERT screening)]
epoch_selection: validation early stopping*
```

The initial, focused, and small C&S grids and their execution status are
itemized in `TUNING_AND_PROTOCOL.md`.

## Concise method description

We concatenate released 768-dimensional LD BERT hidden states,
768-dimensional frozen EmbeddingGemma representations of directed graph-walk
text, and 100-dimensional OGB node features. A four-stage SAGN/SLE predictor
uses two feature-propagation hops and nine label-propagation hops. C&S is
applied independently to each final seed prediction. Scores use the official
split and OGB evaluator.

EmbeddingGemma was not fine-tuned. The 160-token limit means that only the
first of the four generated walks reaches the encoder. No v2 component ablation
was run, so the complete-pipeline difference cannot be assigned specifically
to the walk-text feature.

## Locked v2 selection

The 27-member validation-only C&S grid selected:

```text
correction layers / alpha / adjacency: 50 / 0.35 / DAD
smoothing layers / alpha / adjacency:  50 / 0.60 / DAD
scale:                                  20
```

The winner exceeded the runner-up by 0.0000762912 mean validation accuracy,
equivalent to three validation nodes per seed on average. The grid and lock
were committed at `8320228eca9e66ff8d7b57080e240d312c0729db` before final test
access. The guarded result and durable marker were committed at
`c195821768703cb1b72132b2906795e8003edf7d`.

## Protocol disclosure for the form

During exploratory v1 development, test accuracy was visible while comparing
text encoders, feature stacks, SAGN configurations, and C&S settings. In
particular, the frozen v2 SAGN configuration was chosen among alternatives
with test visible. The v2 C&S grid was identical to the v1 focused grid. On v2
validation, the v1 test-selected setting `(0.30, 0.65, 25)` ranked 11th among
27 candidates.

Seeds 0 through 9 were consumed by the test-aware v1 exploration. V2 therefore
preregistered fresh seeds 10 through 19, reported all ten without selection,
masked test labels during training and validation selection, committed the
validation lock, and then performed one guarded final test invocation. V2
creates a forward test-isolated boundary but cannot erase the earlier method
selection. Full details are in `TUNING_AND_PROTOCOL.md`.

A conservative lower bound is 300 v1 test evaluations from the final ten-seed
SAGN training alone: 30 evaluation points per seed. Encoder screening,
feature-stack trials, and C&S comparisons added more test looks, but the exact
total cannot be recovered because some exploratory logs were not retained.

## Artifact release

The ten v2 stage-3 predictions, logs, hashes, environment metadata, and the
five-part, checksum-verified EmbeddingGemma input tensor are published at:
<https://github.com/MeghanaShanthappa/ld-walklm/releases/tag/ogbn-products-walklm-v2-10seed-20260904>

## Remaining actions

1. Send the updated repository and disclosure to OGB.
2. Record the submission date and OGB decision.
