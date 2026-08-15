#!/usr/bin/env python3
# ruff: noqa: E402
"""Embed a prepared corpus and write the float32 memmap used by the six-domain pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from workflow_common import configure_logging, load_sentence_transformer


WORKFLOW_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = WORKFLOW_ROOT / "data" / "external" / "filtered_corpus.parquet"
DEFAULT_EMBEDDINGS = WORKFLOW_ROOT / "data" / "external" / "filtered_embeddings.f32"
DEFAULT_META = WORKFLOW_ROOT / "data" / "external" / "filtered_embeddings.meta.json"
DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--embedding-meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-chunk-size", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def read_corpus(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.embedding_file.exists() and not args.overwrite:
        raise SystemExit(
            f"Embedding file already exists at {args.embedding_file}. Pass --overwrite to replace it."
        )
    if args.embedding_meta.exists() and not args.overwrite:
        raise SystemExit(
            f"Embedding metadata already exist at {args.embedding_meta}. Pass --overwrite to replace them."
        )

    corpus = read_corpus(args.input).reset_index(drop=True)
    if args.text_column not in corpus.columns:
        raise SystemExit(f"Corpus is missing text column: {args.text_column}")
    if args.max_rows is not None:
        corpus = corpus.head(args.max_rows).copy()
    if corpus.empty:
        raise SystemExit("Corpus contains no rows to embed.")

    model = load_sentence_transformer(args.model_name)
    first_end = min(args.encode_chunk_size, len(corpus))
    first = model.encode(
        corpus[args.text_column].iloc[:first_end].fillna("").astype(str).tolist(),
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    if first.ndim != 2:
        raise SystemExit(f"Expected a two-dimensional embedding matrix, found shape {first.shape}")

    args.embedding_file.parent.mkdir(parents=True, exist_ok=True)
    args.embedding_meta.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.memmap(
        args.embedding_file,
        dtype="float32",
        mode="w+",
        shape=(len(corpus), first.shape[1]),
    )
    embeddings[:first_end] = first
    for start in range(first_end, len(corpus), args.encode_chunk_size):
        end = min(start + args.encode_chunk_size, len(corpus))
        part = model.encode(
            corpus[args.text_column].iloc[start:end].fillna("").astype(str).tolist(),
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")
        if part.shape[1] != embeddings.shape[1]:
            raise SystemExit(
                f"Embedding dimension changed from {embeddings.shape[1]} to {part.shape[1]}."
            )
        embeddings[start:end] = part
    embeddings.flush()

    metadata = {
        "input": str(args.input),
        "rows": int(len(corpus)),
        "dims": int(embeddings.shape[1]),
        "dtype": "float32",
        "model_name": args.model_name,
        "text_column": args.text_column,
        "normalize_embeddings": True,
        "embedding_file": str(args.embedding_file),
    }
    args.embedding_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(args.embedding_file)
    print(args.embedding_meta)


if __name__ == "__main__":
    main()
