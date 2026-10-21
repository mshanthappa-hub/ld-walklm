<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Archived v1 exploratory results

These files belong to the test-aware v1 experiment over seeds `0..9`. They are
retained for auditability and **must not be presented as the submitted v2
result**.

- `cs_per_seed.tsv`: v1 per-seed before/after C&S metrics.
- `summary.tsv`: v1 aggregate metrics.
- `environment.txt`: environment captured after the v1 run.
- `stage3_manifest.tsv`: v1 stage-3 tensor manifest.
- `stage3_sha256.txt`: v1 stage-3 tensor checksums.

The v1 C&S configuration `(0.30, 0.65, 25)` was selected while test accuracy
was visible. `scripts/run_cs_detailed.sh` reconstructs that archived evaluation
only. The primary validation-locked result is under `results/v2/`.

See `TUNING_AND_PROTOCOL.md` for the full disclosure.
