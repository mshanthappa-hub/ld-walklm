#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# See ../THIRD_PARTY_NOTICES.md for dependency and third-party code notices.

# Portions adapted from SAGN_with_SLE.
# NVIDIA SPDX notice above applies to NVIDIA-authored modifications only.
# MIT License
#
# Copyright (c) 2021 skepsun
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# End of third-party notice.

"""Run SAGN Correct & Smooth and persist per-seed evaluator results."""

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np


SAGN_SRC_DIR = Path(
    os.environ.get(
        "SAGN_SRC_DIR",
        str(Path(os.environ.get("SAGN_ROOT", ".")).resolve() / "src"),
    )
).resolve()
if not (SAGN_SRC_DIR / "post_process.py").is_file():
    raise SystemExit(f"SAGN post_process.py not found in {SAGN_SRC_DIR}")

sys.path.insert(0, str(SAGN_SRC_DIR))
import post_process as upstream  # noqa: E402


SEED_PATTERN = re.compile(r"_seed_(\d+)_stage_(\d+)\.pt$")
seed_ids = []
checkpoint_files = []
metric_rows = []
metric_pass = 0


def load_output_files(output_path):
    """Load checkpoints in numeric seed order and retain their provenance."""
    global seed_ids, checkpoint_files

    matched = []
    for filename in glob.glob(output_path):
        match = SEED_PATTERN.search(filename)
        if match is not None:
            matched.append((int(match.group(1)), filename))
    matched.sort(key=lambda item: item[0])

    if len(matched) != detailed_args.expected_runs:
        raise RuntimeError(
            f"Expected {detailed_args.expected_runs} model outputs, "
            f"detected {len(matched)}: {[seed for seed, _ in matched]}"
        )

    seed_ids = [seed for seed, _ in matched]
    checkpoint_files = [filename for _, filename in matched]
    print(f"Detect {len(matched)} model output files for seeds {seed_ids}")
    return [
        upstream.torch.load(filename, map_location="cpu")
        for filename in checkpoint_files
    ]


def write_tsv():
    destination = Path(detailed_args.results_tsv).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("phase\tseed\ttrain\tvalid\ttest\tcheckpoint\n")
        for row in metric_rows:
            handle.write(
                f"{row['phase']}\t{row['seed']}\t{row['train']:.10f}\t"
                f"{row['valid']:.10f}\t{row['test']:.10f}\t"
                f"{Path(row['checkpoint']).name}\n"
            )
    os.replace(temporary, destination)
    print(f"Saved raw per-seed evaluator results to {destination}")


def calculate_metrics(
    probs_list,
    labels,
    train_nid,
    val_nid,
    test_nid,
    evaluator,
    args,
):
    """Print every run plus mean and unbiased sample standard deviation."""
    global metric_pass

    phase = "before" if metric_pass == 0 else "correct_and_smooth"
    metric_pass += 1

    train_positions = upstream.torch.arange(len(train_nid))
    valid_positions = upstream.torch.arange(
        len(train_nid), len(train_nid) + len(val_nid)
    )
    test_positions = upstream.torch.arange(
        len(train_nid) + len(val_nid),
        len(train_nid) + len(val_nid) + len(test_nid),
    )

    split_results = {"train": [], "valid": [], "test": []}
    for seed, checkpoint, probs in zip(seed_ids, checkpoint_files, probs_list):
        if args.dataset in ["ppi", "yelp"]:
            preds = (probs > 0).float()
        else:
            preds = upstream.torch.argmax(probs, dim=-1)

        scores = {
            "train": float(evaluator(preds[train_positions], labels[train_nid])),
            "valid": float(evaluator(preds[valid_positions], labels[val_nid])),
            "test": float(evaluator(preds[test_positions], labels[test_nid])),
        }
        for split, score in scores.items():
            split_results[split].append(score)

        metric_rows.append(
            {
                "phase": phase,
                "seed": seed,
                **scores,
                "checkpoint": checkpoint,
            }
        )
        print(
            f"{phase} seed {seed}: train={scores['train']:.10f} "
            f"valid={scores['valid']:.10f} test={scores['test']:.10f}"
        )

    ddof = 1 if len(probs_list) > 1 else 0
    print("Mean ± unbiased sample standard deviation:")
    for label, key in [("Train", "train"), ("Valid", "valid"), ("Test", "test")]:
        values = split_results[key]
        print(
            f"{label} score: {np.mean(values):.10f}±"
            f"{np.std(values, ddof=ddof):.10f}"
        )
    write_tsv()


parser = upstream.define_parser()
parser.add_argument("--expected-runs", type=int, default=10)
parser.add_argument("--results-tsv", required=True)
detailed_args = parser.parse_args()

upstream.load_output_files = load_output_files
upstream.calculate_metrics = calculate_metrics
upstream.main(detailed_args)
