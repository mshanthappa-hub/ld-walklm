# OGB submission record

## Current disposition

**DO NOT SUBMIT THE CURRENT RESULT AS PROTOCOL-COMPLIANT.**

The artifacts and 10-seed calculation are complete, but the recovered history
shows test-aware encoder and C&S selection. See `TUNING_AND_PROTOCOL.md`.
The following form record is preserved so it can be completed after OGB gives
guidance or after a clean, pre-registered experiment is run.

## Form-ready fields

| OGB field | Value |
|---|---|
| OGB version | 1.3.6 |
| Dataset | `ogbn-products` |
| Method | `LD + EmbeddingGemma-WalkLM + Raw + SAGN/SLE + C&S (external data)` |
| External data | Yes: released LD BERT features, product raw text, and `google/embeddinggemma-300m` |
| Test performance | `0.8747530490 ± 0.0004873863` (exploratory; not currently eligible) |
| Validation performance | `0.9400096636 ± 0.0002987397` |
| Runs | 10 independent seeds, 0 through 9 |
| Standard deviation | Unbiased sample standard deviation (`torch.std`, correction 1) |
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
seed's final predictions. Scores use the official split and OGB evaluator.

## Tuned hyperparameters

Use the complete disclosure in `TUNING_AND_PROTOCOL.md`, not a shortened claim.
The final recorded SAGN configuration is in `configs/experiment.yaml`. The C&S
candidate ranges included:

```text
correction_alpha: [0.30*, 0.35, 0.40, 0.4553513529697225, 0.55]
smoothing_alpha:  [0.60, 0.65*, 0.70, 0.7153479868942039, 0.80]
scale:            [10, 15, 20, 25*, 30]
```

The asterisks identify the recorded configuration, not a claim that it was
selected without test feedback.

## Remaining user/OGB actions

1. Add a monitored contact email.
2. Send OGB the protocol disclosure and ask whether the exploratory history
   disqualifies this entry; do not omit it.
3. If required, run a new pre-registered validation-only experiment and use
   test accuracy exactly once after the configuration is locked. The committed
   v2 workflow uses seeds `10..19` and stops before final test access.
4. Submit through the official OGB Google form only after the above is clear.
