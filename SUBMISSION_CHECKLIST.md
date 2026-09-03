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
- [x] Recover the exact EmbeddingGemma command and semantic generation code.
- [x] Recover the EmbeddingGemma artifact byte size and SHA-256.
- [x] Record the LD input tensor byte size and SHA-256.
- [x] Calculate the conservative multi-stage parameter count.
- [x] Write a technical report and form-ready submission record.
- [x] Preserve the tuning history and protocol audit.
- [x] Add a validation-only C&S selection and configuration-locking workflow.

## Before submitting

- [ ] Preserve or link the LD tensor outside Git; the EmbeddingGemma artifact is
      in the original WalkLM LFS commit.
- [ ] Re-run the evaluator from downloaded release assets and compare its TSV
      byte-for-byte or numerically with `results/cs_per_seed.tsv`.
- [ ] Add a monitored contact email to `SUBMISSION.md`.
- [ ] Disclose the test-aware selection history to OGB and obtain a ruling.
- [ ] If required, run a clean, pre-registered validation-only experiment and
      evaluate the test split once after locking the method.
- [ ] For v2, commit all 27 validation candidate records and
      `locked_cs_config.json` before invoking the guarded final-test script.
- [ ] Submit through the official OGB form only when protocol eligibility is
      resolved.
- [ ] Add the submission date and official leaderboard URL or status here.

## Proposed future method name

`LD + EmbeddingGemma-WalkLM + Raw + SAGN/SLE + C&S`

## Exploratory result (not currently eligible to submit)

```text
Validation: 0.9400096636 ± 0.0002987397
Test:       0.8747530490 ± 0.0004873863
Runs:       10
Std:        unbiased sample standard deviation (ddof=1)
External data: Yes
OGB version: 1.3.6
```
