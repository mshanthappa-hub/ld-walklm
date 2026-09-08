#!/usr/bin/env python3
"""Compare reconstructed prefix embeddings with the archived tensor rows."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_walklm(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = value["walklm_x"]
    if not torch.is_tensor(value) or value.ndim != 2:
        raise TypeError(f"{path} does not contain a 2-D walklm_x tensor")
    return value.detach().cpu().float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstructed", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    reconstructed = load_walklm(args.reconstructed)
    archived = load_walklm(args.archived)
    if reconstructed.size(1) != archived.size(1):
        raise RuntimeError("Embedding dimensions do not match")
    if reconstructed.size(0) > archived.size(0):
        raise RuntimeError("Reconstruction has more rows than archived tensor")
    archived = archived[: reconstructed.size(0)]

    cosine = F.cosine_similarity(reconstructed, archived, dim=1)
    difference = reconstructed - archived
    payload = {
        "rows": reconstructed.size(0),
        "dimensions": reconstructed.size(1),
        "mean_cosine_similarity": float(cosine.mean()),
        "min_cosine_similarity": float(cosine.min()),
        "max_cosine_similarity": float(cosine.max()),
        "mean_l2_difference": float(difference.norm(dim=1).mean()),
        "max_absolute_difference": float(difference.abs().max()),
        "exactly_equal": bool(torch.equal(reconstructed, archived)),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
