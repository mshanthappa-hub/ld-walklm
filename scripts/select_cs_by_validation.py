#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# See ../THIRD_PARTY_NOTICES.md for dependency and third-party code notices.

"""Lock one C&S configuration using validation results only."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(
        path
        for path in args.grid_dir.glob("*.json")
        if path.resolve() != args.output.resolve()
    )
    if not files:
        raise SystemExit("No validation result JSON files found")

    candidates = []
    digest = hashlib.sha256()
    for path in files:
        raw = path.read_bytes()
        digest.update(path.name.encode("utf-8") + b"\0" + raw)
        record = json.loads(raw)
        if record.get("test_metrics_accessed") is not False:
            raise RuntimeError(f"Unsafe tuning record: {path}")
        result = record["correct_and_smooth"]
        candidates.append(
            (
                -float(result["mean"]),
                float(result["std_unbiased"]),
                path.name,
                record,
            )
        )

    candidates.sort(key=lambda item: item[:3])
    selected = candidates[0][3]
    payload = {
        "selection_rule": (
            "maximum mean validation accuracy; ties broken by minimum "
            "unbiased validation standard deviation, then filename"
        ),
        "test_metrics_accessed": False,
        "candidate_count": len(candidates),
        "candidate_records_sha256": digest.hexdigest(),
        "selected_parameters": selected["parameters"],
        "selected_validation": selected["correct_and_smooth"],
        "warning": (
            "This lock makes future selection auditable but cannot erase any "
            "test-aware exploration performed before it was created."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
