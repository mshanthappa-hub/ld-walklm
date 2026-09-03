#!/usr/bin/env python3
"""Run C&S while exposing only validation accuracy for model selection.

This wrapper deliberately never evaluates or prints test labels. The upstream
C&S implementation receives split indices because predictions are stored in
train/validation/test order, but C&S itself receives labels only at train_nid.
"""

import glob
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
validation_before = None
validation_after = None


def load_output_files(output_path):
    global seed_ids
    matched = []
    for filename in glob.glob(output_path):
        match = SEED_PATTERN.search(filename)
        if match is not None:
            matched.append((int(match.group(1)), filename))
    matched.sort(key=lambda item: item[0])
    if len(matched) != args.expected_runs:
        raise RuntimeError(
            f"Expected {args.expected_runs} outputs, found "
            f"{len(matched)}: {[seed for seed, _ in matched]}"
        )
    seed_ids = [seed for seed, _ in matched]
    print(f"Detected model outputs for seeds {seed_ids}")
    return [
        upstream.torch.load(filename, map_location="cpu")
        for _, filename in matched
    ]


def calculate_validation_only(
    probs_list,
    labels,
    train_nid,
    val_nid,
    test_nid,
    evaluator,
    parsed_args,
):
    del train_nid, test_nid
    global metric_pass, validation_before, validation_after
    phase = "before" if metric_pass == 0 else "correct_and_smooth"
    metric_pass += 1

    # Checkpoints are ordered [train, validation, test]. Only the validation
    # slice and validation labels are touched here.
    validation_positions = upstream.torch.arange(
        len(parsed_train_nid), len(parsed_train_nid) + len(val_nid)
    )
    scores = []
    for seed, probabilities in zip(seed_ids, probs_list):
        predictions = upstream.torch.argmax(probabilities, dim=-1)
        score = float(
            evaluator(predictions[validation_positions], labels[val_nid])
        )
        scores.append(score)
        print(f"{phase} seed {seed}: valid={score:.10f}")

    payload = {
        "per_seed": dict(zip(map(str, seed_ids), scores)),
        "mean": float(np.mean(scores)),
        "std_unbiased": float(np.std(scores, ddof=1)),
    }
    if phase == "before":
        validation_before = payload
    else:
        validation_after = payload
    print(
        f"{phase} validation: {payload['mean']:.10f} ± "
        f"{payload['std_unbiased']:.10f}"
    )


parser = upstream.define_parser()
parser.add_argument("--expected-runs", type=int, default=10)
parser.add_argument("--output-json", required=True)
args = parser.parse_args()

# Save the length needed to locate the validation slice without ever scoring
# the test slice. It is populated by a small load_dataset wrapper.
parsed_train_nid = []
original_load_dataset = upstream.load_dataset


def load_dataset_and_capture(*load_args, **load_kwargs):
    global parsed_train_nid
    data = original_load_dataset(*load_args, **load_kwargs)
    parsed_train_nid = data[3]
    return data


upstream.load_dataset = load_dataset_and_capture
upstream.load_output_files = load_output_files
upstream.calculate_metrics = calculate_validation_only
upstream.main(args)

destination = Path(args.output_json).expanduser().resolve()
destination.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "dataset": args.dataset,
    "seeds": seed_ids,
    "test_metrics_accessed": False,
    "parameters": {
        "correction_layers": args.num_correction_layers,
        "smoothing_layers": args.num_smoothing_layers,
        "correction_alpha": args.correction_alpha,
        "smoothing_alpha": args.smoothing_alpha,
        "correction_adjacency": args.correction_adj,
        "smoothing_adjacency": args.smoothing_adj,
        "scale": args.scale,
    },
    "before": validation_before,
    "correct_and_smooth": validation_after,
}
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, destination)
print(f"Saved validation-only result to {destination}")
