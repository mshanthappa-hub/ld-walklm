#!/usr/bin/env python3
"""Reconstruct the frozen EmbeddingGemma walk-text feature tensor.

This file was written after the 2026-08-21 generation. It reconstructs the
recorded semantic path but is not the original runtime program. It intentionally does not train or evaluate a classifier and
never reads OGB labels. Random walks are sampled from the directed edge list
with seed 42, textualized with the released product text, encoded with the
last hidden state of google/embeddinggemma-300m, mean pooled, and L2 normalized.

The archived tensor also contains raw_x because the original WalkLM serializer
stored both arrays in one torch checkpoint.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
import transformers


EXPECTED_NODES = 2_449_029
EXPECTED_EDGES = 123_718_280
EXPECTED_WALKLM_DIM = 768
EXPECTED_RAW_DIM = 100
TEXT_FILE_ID = "1I-S176-W4Bm1iPDjQv3hYwQBtxE0v8mt"
MODEL_REVISION = "57c266a7"
ARCHIVED_SHA256 = (
    "fef45b647df5155bf17505508cb08ee5a164ad9b33b69910ad1b437109721c90"
)
DEFAULT_OUTPUT = (
    "walklm_products_2449029_embeddinggemma300m_"
    "walks4_len4_mean_norm_embeddings.pt"
)


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_adj_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    row, col = edge_index.cpu()
    for source, destination in zip(row.tolist(), col.tolist()):
        adjacency[source].append(destination)
    return adjacency


def sample_walk(
    start: int,
    adjacency: List[List[int]],
    walk_length: int,
    generator: torch.Generator,
) -> List[int]:
    walk = [start]
    current = start
    for _ in range(walk_length - 1):
        neighbors = adjacency[current]
        if not neighbors:
            break
        offset = int(torch.randint(len(neighbors), (1,), generator=generator))
        current = neighbors[offset]
        walk.append(current)
    return walk


def build_text_walk_corpus(
    text: List[str],
    edge_index: torch.Tensor,
    num_walks: int,
    walk_length: int,
    seed: int,
    max_text_chars: int,
    max_nodes: Optional[int] = None,
) -> List[str]:
    generator = torch.Generator().manual_seed(seed)
    adjacency = build_adj_list(edge_index, len(text))
    corpus: List[str] = []
    output_nodes = len(text) if max_nodes is None else min(max_nodes, len(text))
    for node_index in range(output_nodes):
        walks = [
            sample_walk(node_index, adjacency, walk_length, generator)
            for _ in range(num_walks)
        ]
        walk_texts = []
        for walk in walks:
            walk_texts.append(
                " cites ".join(
                    text[node_id][:max_text_chars].replace("\n", " ")
                    for node_id in walk
                )
            )
        corpus.append(" ".join(walk_texts))
    return corpus


@torch.inference_mode()
def encode_texts(
    model,
    tokenizer,
    corpus: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    chunks: List[torch.Tensor] = []
    for start in range(0, len(corpus), batch_size):
        end = min(start + batch_size, len(corpus))
        batch = tokenizer(
            corpus[start:end],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch, output_hidden_states=True)
        hidden = output.hidden_states[-1]
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embedding = F.normalize(embedding, p=2, dim=-1)
        chunks.append(embedding.float().cpu())
        if start % (batch_size * 1000) == 0:
            print(f"Encoded {end}/{len(corpus)} nodes", flush=True)
    return torch.cat(chunks, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Parent OGB data directory")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--model", default="google/embeddinggemma-300m")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-walks", type=int, default=4)
    parser.add_argument("--walk-length", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-text-chars", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Reconstruct only the deterministic leading node prefix",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {output}; pass --force if intended")
    if args.max_nodes is not None and args.max_nodes <= 0:
        raise SystemExit("--max-nodes must be positive")

    from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
    from torch_geometric.data import download_google_url
    from torch_geometric import seed_everything
    from transformers import AutoModel, AutoTokenizer

    seed_everything(args.seed)
    root = Path(args.data_dir).expanduser().resolve()
    dataset = PygNodePropPredDataset(name="ogbn-products", root=str(root))
    data = dataset[0]
    if data.num_nodes != EXPECTED_NODES or data.edge_index.size(1) != EXPECTED_EDGES:
        raise RuntimeError(
            "Unexpected graph size: "
            f"nodes={data.num_nodes}, edges={data.edge_index.size(1)}"
        )
    if tuple(data.x.shape) != (EXPECTED_NODES, EXPECTED_RAW_DIM):
        raise RuntimeError(f"Unexpected raw feature shape: {tuple(data.x.shape)}")

    text_path = download_google_url(
        id=TEXT_FILE_ID,
        folder=str(root / "ogbn_products" / "raw"),
        filename="node-text.csv.gz",
        log=True,
    )
    text = list(pd.read_csv(text_path)["text"])
    if len(text) != EXPECTED_NODES:
        raise RuntimeError(f"Unexpected text row count: {len(text)}")

    print("Building attributed random-walk corpus...", flush=True)
    corpus = build_text_walk_corpus(
        text,
        data.edge_index,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        seed=args.seed,
        max_text_chars=args.max_text_chars,
        max_nodes=args.max_nodes,
    )
    print(f"Example walk text: {corpus[0][:500]}...", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading language model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision
    )
    model = AutoModel.from_pretrained(
        args.model, revision=args.model_revision, use_safetensors=True
    ).to(device)
    walklm_x = encode_texts(
        model,
        tokenizer,
        corpus,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    output_nodes = EXPECTED_NODES if args.max_nodes is None else min(
        args.max_nodes, EXPECTED_NODES
    )
    if tuple(walklm_x.shape) != (output_nodes, EXPECTED_WALKLM_DIM):
        raise RuntimeError(f"Unexpected embedding shape: {tuple(walklm_x.shape)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"walklm_x": walklm_x, "raw_x": data.x[:output_nodes].cpu()}, output
    )
    digest = sha256_file(output)
    metadata = {
        "output": str(output),
        "sha256": digest,
        "archived_sha256": ARCHIVED_SHA256,
        "matches_archived_container": digest == ARCHIVED_SHA256,
        "walklm_shape": list(walklm_x.shape),
        "walklm_dtype": str(walklm_x.dtype),
        "raw_shape": list(data.x[:output_nodes].shape),
        "model": args.model,
        "model_revision": args.model_revision,
        "script_status": "post_hoc_reconstruction",
        "max_nodes": args.max_nodes,
        "seed": args.seed,
        "num_walks": args.num_walks,
        "walk_length": args.walk_length,
        "pooling": "last_hidden_state_attention_masked_mean",
        "normalize_embeddings": True,
        "max_length": args.max_length,
        "max_text_chars": args.max_text_chars,
        "batch_size": args.batch_size,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "encoder_fine_tuned": False,
        "token_limit_effect": "walks_2_through_4_truncated_before_encoder",
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
