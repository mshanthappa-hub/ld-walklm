<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# OGB submission checklist

## Completed

- [x] Use the official `ogbn-products` split and OGB 1.3.6 evaluator.
- [x] Declare external data: **Yes**.
- [x] Preserve the v1 exploratory run and complete tuning disclosure.
- [x] Commit the v2 protocol before execution (`827e8fc`).
- [x] Run 10 new v2 seeds (`10..19`) without test metric evaluation during training.
- [x] Evaluate all 27 C&S candidates using validation metrics only.
- [x] Select maximum mean validation accuracy with the committed tie breakers.
- [x] Commit every validation record and the selected lock before test access (`8320228`).
- [x] Run the guarded final-test evaluator once.
- [x] Preserve the durable access marker and final JSON (`c195821`).
- [x] Independently reproduce the v2 mean and unbiased sample standard deviation.
- [x] Verify that the selected candidate is the actual validation-grid winner.
- [x] Verify matching lock digests in the lock, marker, and final JSON.
- [x] Record pinned sources, exact input tensor checksums, integration patch, parameter count, and environment.
- [x] Write the technical report and form-ready submission record.
- [x] Publish all ten v2 checkpoints, logs, hashes, and environment metadata in the v2 GitHub release.
- [x] Add the monitored submission contact email.
- [x] Use the full predictor name and remove SCR from this method.
- [x] Add author NVIDIA affiliations, a license, and the C&S citation.
- [x] Add OGB-format tuned hyperparameters and complete v1/v2 disclosure.
- [x] Add a v2-from-release reproduction script and mark v1 outputs archived.

## Before submitting

- [x] Mirror the EmbeddingGemma input in five sub-2-GiB v2 release parts.
- [x] Mirror all 36 v1 release assets into `ld-walklm` and make the old repository private.
- [x] Run and commit the 128-node EmbeddingGemma prefix comparison (exact equality; commit `7c09ca5`).

- [x] Reproduce all 27 validation candidates, lock, and final test from downloaded v2 assets (commit `c2dd7fa`).
- [ ] Include the concise historical protocol disclosure in the OGB submission.
- [ ] Submit through the official OGB process.
- [ ] Add the submission date and official leaderboard URL or decision here.

## Primary v2 result

```text
Validation: 0.9405843908 ± 0.0004332633
Test:       0.8749843545 ± 0.0004904255
Runs:       10 (seeds 10–19)
Std:        unbiased sample standard deviation (ddof=1)
External data: Yes
OGB version: 1.3.6
Method:     LD+EG-Walk+SAGN/SLE+C&S
Status:     not official until accepted
```
