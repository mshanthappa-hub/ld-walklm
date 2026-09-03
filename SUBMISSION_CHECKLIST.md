# OGB submission checklist

## Completed

- [x] Use the official `ogbn-products` split.
- [x] Run 10 independent seeds (`0..9`).
- [x] Preserve per-seed stage-3 predictions.
- [x] Preserve per-seed metrics before and after C&S.
- [x] Calculate mean and unbiased sample standard deviation across 10 runs.
- [x] Record OGB version 1.3.6.
- [x] Record training and post-processing hyperparameters.
- [x] Declare external data: **Yes**.
- [x] Archive all 10 final tensors, logs, hashes, and environment details.
- [x] Verify the 10 release assets against SHA-256 checksums.
- [x] Record pinned source revisions and their provenance boundary.
- [x] Preserve and validate the recovered 1,636-dimensional SAGN integration.

## Before submitting

- [ ] Record SHA-256 hashes for the LD and EmbeddingGemma input tensors.
- [ ] Preserve or link the two external input tensors outside Git.
- [ ] Recover the exact EmbeddingGemma feature-generation command or script.
- [ ] Re-run the evaluator from downloaded release assets and compare its TSV
      byte-for-byte or numerically with `results/cs_per_seed.tsv`.
- [ ] Choose the final method name and author list.
- [ ] Prepare the concise method description and external-data disclosure.
- [ ] Submit through the official OGB form.
- [ ] Add the submission date and official leaderboard URL or status here.

## Proposed method name

`LD + EmbeddingGemma-WalkLM + Raw + SAGN/SLE + C&S`

## Result to submit

```text
Validation: 0.9400096636 ± 0.0002987397
Test:       0.8747530490 ± 0.0004873863
Runs:       10
Std:        unbiased sample standard deviation (ddof=1)
External data: Yes
OGB version: 1.3.6
```
