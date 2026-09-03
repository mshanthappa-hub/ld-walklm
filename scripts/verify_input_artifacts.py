#!/usr/bin/env python3
"""Validate and hash the two external feature tensors used for training."""

import argparse
import gc
import hashlib
import os
from pathlib import Path

import torch


EXPECTED = {
    "ld": {"shape": (2449029, 768), "key": "hidden_state"},
    "walklm": {"shape": (2449029, 768), "key": "walklm_x"},
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tensor(path, preferred_key):
    try:
        value = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(str(path), map_location="cpu")

    if isinstance(value, dict):
        if preferred_key in value:
            value = value[preferred_key]
        else:
            tensors = [item for item in value.values() if torch.is_tensor(item)]
            if len(tensors) != 1:
                raise ValueError(
                    "Expected one tensor in {}; keys are {}".format(
                        path, sorted(value.keys())
                    )
                )
            value = tensors[0]
    if not torch.is_tensor(value):
        raise TypeError("{} did not contain a torch.Tensor".format(path))
    return value


def inspect(role, path):
    spec = EXPECTED[role]
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    checksum = sha256_file(path)
    tensor = load_tensor(path, spec["key"])
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype).removeprefix("torch.")
    if shape != spec["shape"]:
        raise ValueError(
            "{} shape is {}; expected {}".format(role, shape, spec["shape"])
        )
    if tensor.dtype != torch.float32:
        raise TypeError(
            "{} dtype is {}; expected torch.float32".format(role, tensor.dtype)
        )

    row = {
        "role": role,
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": checksum,
        "shape": "x".join(str(item) for item in shape),
        "dtype": dtype,
    }
    del tensor
    gc.collect()
    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Only inspect trusted .pt files; old PyTorch may deserialize via pickle.",
    )
    parser.add_argument("--ld", required=True, type=Path)
    parser.add_argument("--walklm", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [inspect("ld", args.ld), inspect("walklm", args.walklm)]
    header = "role\tfilename\tbytes\tsha256\tshape\tdtype"
    body = [
        "{role}\t{filename}\t{bytes}\t{sha256}\t{shape}\t{dtype}".format(**row)
        for row in rows
    ]
    report = "\n".join([header] + body) + "\n"
    print(report, end="")

    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(report, encoding="utf-8")
        os.replace(temporary, destination)
        print("Saved {}".format(destination))


if __name__ == "__main__":
    main()
