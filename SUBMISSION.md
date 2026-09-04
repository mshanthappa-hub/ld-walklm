# OGB submission record

## Current disposition

**V2 IS THE PRIMARY SUBMISSION CANDIDATE; IT IS NOT YET AN OFFICIAL RESULT.**

The validation-selected configuration, one-time final evaluation, and audit
evidence are complete. V2 prevents test-driven changes after the committed
protocol boundary. Because the earlier v1 test results informed the fixed
EmbeddingGemma pipeline, preserve the concise disclosure below when asking OGB
to accept the entry.

## Form-ready fields

| OGB field | Value |
|---|---|
| OGB version | 1.3.6 |
| Dataset | `ogbn-products` |
| Method | `LD + EmbeddingGemma-WalkLM + Raw + SAGN/SLE + C&S (external data)` |
| External data | Yes: released LD BERT features, product raw text, and `google/embeddinggemma-300m` |
| Test performance | `0.8749843545 ± 0.0004904255` |
| Validation performance | `0.9405843908 ± 0.0004332633` |
| Runs | 10 independent seeds, 10 through 19 |
| Standard deviation | Unbiased sample standard deviation (`ddof=1`) |
| Contact | Meghana Shanthappa — **REQUIRED: add a monitored email address** |
| Code | <https://github.com/MeghanaShanthappa/ogbn-products-leaderboard_walklm> |
| Technical report | `report/technical_report.pdf` in the code repository |
| Parameters | 423,228,606 conservative multi-stage total; see `results/parameter_count.tsv` |
| Hardware | NVIDIA RTX PRO 4500 Blackwell, 32,623 MiB GPU memory |
| Officiality | Official implementation of this combination; upstream components are cited third-party implementations |

## Concise method description

We concatenate released 768-dimensional LD BERT hidden states,
768-dimensional EmbeddingGemma representations of four length-four directed
product-graph random walks, and the 100-dimensional OGB node features. A
four-stage SAGN/SLE model uses two feature-propagation hops and nine
label-propagation hops. Parameter-free Correct & Smooth is applied to each
seed final prediction. Scores use the official split and OGB evaluator.

## Locked v2 selection

The 27-member validation-only C&S grid selected:

```text
correction layers / alpha / adjacency: 50 / 0.35 / DAD
smoothing layers / alpha / adjacency:  50 / 0.60 / DAD
scale:                                  20
```

The grid and lock were committed at `8320228eca9e66ff8d7b57080e240d312c0729db`
before final test access. The guarded result and durable access marker were
committed at `c195821768703cb1b72132b2906795e8003edf7d`. The final test mean is
0.0012843545 (0.12843545 percentage points) above the public 0.8737 reference
visible on 2026-09-04; this is only a prospective rank until accepted.

## Protocol disclosure

V2 used new seeds and validation-only training/C&S selection, with final test
evaluation after the configuration lock was committed. Earlier exploratory
runs had shown test scores while comparing encoders and C&S settings; those
runs informed the fixed EmbeddingGemma pipeline but no v2 choice was changed
after its committed boundary. Full details are in `TUNING_AND_PROTOCOL.md`.

## Artifact release

All ten v2 stage-3 checkpoints, logs, hashes, and environment metadata are in:
<https://github.com/MeghanaShanthappa/ogbn-products-leaderboard_walklm/releases/tag/ogbn-products-walklm-v2-10seed-20260904>

## Remaining actions

1. Add a monitored contact email.
2. Send the repository/report and the protocol disclosure to OGB with the submission.
3. Record the submission date and official status in this repository.
