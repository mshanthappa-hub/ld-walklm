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

## Before submitting

- [ ] Re-run validation post-processing from downloaded v2 assets and compare it with the committed records.
- [ ] Add a monitored contact email to `SUBMISSION.md`.
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
Status:     prospective Top-1; not official until accepted
```
