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

"""Apply a committed C&S lock and evaluate only the final test predictions."""

import glob
import hashlib
import json
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
metric_pass = 0
parsed_train_nid = []
parsed_val_nid = []
test_result = None


def load_output_files(output_path):
    global seed_ids
    matched = []
    for filename in glob.glob(output_path):
        match = SEED_PATTERN.search(filename)
        if match is not None:
            matched.append((int(match.group(1)), filename))
    matched.sort(key=lambda item: item[0])
    seed_ids = [seed for seed, _ in matched]
    if seed_ids != args.expected_seeds:
        raise RuntimeError(
            f"Expected seeds {args.expected_seeds}, found {seed_ids}"
        )
    print(f"Detected locked final outputs for seeds {seed_ids}")
    return [
        upstream.torch.load(filename, map_location="cpu")
        for _, filename in matched
    ]


def calculate_final_test_only(
    probs_list,
    labels,
    train_nid,
    val_nid,
    test_nid,
    evaluator,
    parsed_args,
):
    del train_nid, val_nid, parsed_args
    global metric_pass, test_result
    metric_pass += 1
    if metric_pass == 1:
        print("Before-C&S metrics intentionally skipped")
        return
    if metric_pass != 2:
        raise RuntimeError("Unexpected additional metric pass")

    first_test_position = len(parsed_train_nid) + len(parsed_val_nid)
    test_positions = upstream.torch.arange(
        first_test_position, first_test_position + len(test_nid)
    )
    scores = []
    for seed, probabilities in zip(seed_ids, probs_list):
        predictions = upstream.torch.argmax(probabilities, dim=-1)
        score = float(evaluator(predictions[test_positions], labels[test_nid]))
        scores.append(score)
        print(f"final test seed {seed}: test={score:.10f}")

    test_result = {
        "per_seed": dict(zip(map(str, seed_ids), scores)),
        "mean": float(np.mean(scores)),
        "std_unbiased": float(np.std(scores, ddof=1)),
    }
    print(
        f"final test: {test_result['mean']:.10f} ± "
        f"{test_result['std_unbiased']:.10f}"
    )


def load_dataset_and_capture(*load_args, **load_kwargs):
    global parsed_train_nid, parsed_val_nid
    data = original_load_dataset(*load_args, **load_kwargs)
    parsed_train_nid = data[3]
    parsed_val_nid = data[4]
    return data


parser = upstream.define_parser()
parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
parser.add_argument("--lock-file", type=Path, required=True)
parser.add_argument("--output-json", type=Path, required=True)
args = parser.parse_args()

lock_bytes = args.lock_file.read_bytes()
lock = json.loads(lock_bytes)
if lock.get("test_metrics_accessed") is not False:
    raise RuntimeError("The selected configuration is not a validation-only lock")
if lock.get("candidate_count") != 27:
    raise RuntimeError("Expected a lock selected from exactly 27 candidates")

parameters = lock["selected_parameters"]
args.num_correction_layers = int(parameters["correction_layers"])
args.num_smoothing_layers = int(parameters["smoothing_layers"])
args.correction_alpha = float(parameters["correction_alpha"])
args.smoothing_alpha = float(parameters["smoothing_alpha"])
args.correction_adj = parameters["correction_adjacency"]
args.smoothing_adj = parameters["smoothing_adjacency"]
args.scale = float(parameters["scale"])

original_load_dataset = upstream.load_dataset
upstream.load_dataset = load_dataset_and_capture
upstream.load_output_files = load_output_files
upstream.calculate_metrics = calculate_final_test_only
upstream.main(args)

if test_result is None:
    raise RuntimeError("Final test metric was not produced")
payload = {
    "dataset": args.dataset,
    "seeds": seed_ids,
    "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
    "selected_parameters": parameters,
    "test_metrics_accessed": True,
    "evaluation_scope": "post_correct_and_smooth_only",
    "test": test_result,
}
destination = args.output_json.expanduser().resolve()
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, destination)
print(f"Saved one-time final test result to {destination}")
