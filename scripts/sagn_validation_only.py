#!/usr/bin/env python3
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
upstream.main(args)
