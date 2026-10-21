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

"""Train SAGN while evaluating only train and validation labels.

The pinned upstream trainer evaluates the test split at every evaluation
interval. This wrapper replaces that evaluator before calling upstream.main.
It still writes probabilities for all nodes because later SLE stages and the
eventual locked final evaluation need them, but it never evaluates test-label values.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np


SAGN_SRC_DIR = Path(
    os.environ.get(
        "SAGN_SRC_DIR",
        str(Path(os.environ.get("SAGN_ROOT", ".")).resolve() / "src"),
    )
).resolve()
if not (SAGN_SRC_DIR / "sagn.py").is_file():
    raise SystemExit(f"SAGN sagn.py not found in {SAGN_SRC_DIR}")

sys.path.insert(0, str(SAGN_SRC_DIR))
import sagn as upstream  # noqa: E402
import pre_process as preprocessing  # noqa: E402


original_load_dataset = preprocessing.load_dataset


def load_dataset_with_masked_test_labels(*load_args, **load_kwargs):
    """Replace test labels with an invalid sentinel before preprocessing."""
    data = list(original_load_dataset(*load_args, **load_kwargs))
    labels = data[1].clone()
    test_nid = data[5]
    labels[test_nid] = -1
    data[1] = labels
    print(f"VALIDATION_ONLY_PROTOCOL: masked {len(test_nid)} test labels")
    return tuple(data)


def validation_only_test(
    model,
    feats,
    label_emb,
    teacher_probs,
    labels,
    loss_fcn,
    val_loader,
    all_split_loader,
    evaluator,
    train_nid,
    val_nid,
    test_nid,
    args,
):
    """Return train/validation metrics and a NaN test placeholder."""
    del teacher_probs, all_split_loader, test_nid
    model.eval()
    device = labels.device
    loss_list = []
    count_list = []

    for batch in val_loader:
        cpu_batch = batch.cpu()
        if args.dataset == "ogbn-mag":
            batch_feats = {
                relation: [tensor[cpu_batch].to(device) for tensor in feature]
                for relation, feature in feats.items()
            }
        else:
            batch_feats = (
                [tensor[cpu_batch].to(device) for tensor in feats]
                if isinstance(feats, list)
                else feats[cpu_batch].to(device)
            )
        batch_label_emb = (
            label_emb[cpu_batch].to(device) if label_emb is not None else None
        )
        if args.model in ["sagn", "plain_sagn"] and not args.avoid_features:
            output, _ = model(batch_feats, batch_label_emb)
        else:
            output = model(batch_feats, batch_label_emb)
        loss_list.append(loss_fcn(output, labels[cpu_batch]).cpu().item())
        count_list.append(len(batch))

    loss_values = np.asarray(loss_list)
    counts = np.asarray(count_list)
    validation_loss = float((loss_values * counts).sum() / counts.sum())

    # Inference for model selection is deliberately limited to train+valid.
    evaluation_nodes = upstream.torch.cat([train_nid, val_nid], dim=0)
    evaluation_loader = upstream.torch.utils.data.DataLoader(
        evaluation_nodes,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )
    predictions = []
    start = time.time()
    for batch in evaluation_loader:
        cpu_batch = batch.cpu()
        if args.dataset == "ogbn-mag":
            batch_feats = {
                relation: [tensor[cpu_batch].to(device) for tensor in feature]
                for relation, feature in feats.items()
            }
        else:
            batch_feats = (
                [tensor[cpu_batch].to(device) for tensor in feats]
                if isinstance(feats, list)
                else feats[cpu_batch].to(device)
            )
        batch_label_emb = (
            label_emb[cpu_batch].to(device) if label_emb is not None else None
        )
        if args.model in ["sagn", "plain_sagn"] and not args.avoid_features:
            output, _ = model(batch_feats, batch_label_emb)
        else:
            output = model(batch_feats, batch_label_emb)
        if args.dataset in ["ppi", "ppi_large", "yelp"]:
            predictions.append((output > 0).float())
        else:
            predictions.append(upstream.torch.argmax(output, dim=-1))

    predictions = upstream.torch.cat(predictions, dim=0)
    inference_time = time.time() - start
    train_count = len(train_nid)
    train_score = evaluator(predictions[:train_count], labels[train_nid])
    validation_score = evaluator(
        predictions[train_count:], labels[val_nid]
    )
    return (
        train_score,
        validation_score,
        float("nan"),
        validation_loss,
        inference_time,
    )


parser = upstream.define_parser()
args = parser.parse_args()
if args.dataset != "ogbn-products":
    raise SystemExit("This protocol wrapper is locked to ogbn-products")
print("VALIDATION_ONLY_PROTOCOL: test-label values are masked and never evaluated")
preprocessing.load_dataset = load_dataset_with_masked_test_labels
upstream.test = validation_only_test
try:
    upstream.main(args)
except ValueError as error:
    # Pinned upstream appends per-stage validation histories with different
    # lengths (300 vs. 100 epochs) and then asks NumPy to create a rectangular
    # array. Newer NumPy versions reject that ragged reporting structure. The
    # exception occurs only after the final probabilities have been saved.
    model_directory = (
        args.model
        if args.weight_style == "attention"
        else f"{args.model}_{args.weight_style}"
    )
    final_stage = len(args.epoch_setting) - 1
    final_checkpoint = (
        Path(args.probs_dir)
        / args.dataset
        / model_directory
        / (
            f"use_labels_{args.use_labels}_"
            f"use_feats_{not args.avoid_features}_K_{args.K}_"
            f"label_K_{args.label_K}_probs_seed_{args.seed}_"
            f"stage_{final_stage}.pt"
        )
    )
    known_reporting_error = "inhomogeneous shape" in str(error)
    checkpoint_complete = (
        final_checkpoint.is_file() and final_checkpoint.stat().st_size > 0
    )
    if not (known_reporting_error and checkpoint_complete):
        raise
    print(
        "KNOWN_UPSTREAM_SUMMARY_BUG: ignored ragged NumPy conversion after "
        f"verified final checkpoint {final_checkpoint} "
        f"({final_checkpoint.stat().st_size} bytes)"
    )
