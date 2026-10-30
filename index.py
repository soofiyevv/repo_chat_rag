"""
Step 2a. Build the vector index.

Reads chunks.jsonl, turns every chunk into an embedding with a local
sentence-transformers model and stores it in a persistent Chroma collection.
Test files are marked with is_test so search can skip them.
Each chunk is embedded together with its file path, name, signature and
docstring, so search can match on both meaning and names.
Every model has its own collection, rebuilt from scratch on every run.

Usage:
  python index.py
  python index.py --model bge-m3
  python index.py --chunks other_chunks.jsonl
"""
from __future__ import annotations

import argparse
import time

from retriever import is_test, load_chunks
from config import CHUNKS_FILE, DB_PATH, MODELS, load_model, model_config

import chromadb


BATCH = 256


def chunk_to_text(c: dict, prefix: str = "") -> str:
    parts = [
        f"File: {c['path']}",
        f"Name: {c['name']} ({c['kind']})",
    ]
    if c["signature"]:
        parts.append(f"Signature: {c['signature']}")
    if c["docstring"]:
        parts.append(f"Docstring: {c['docstring']}")
    parts.append("")
    parts.append(c["code"])
    return prefix + "\n".join(parts)


def chunk_metadata(c: dict) -> dict:
    return {
        "is_test": is_test(c["path"]),
        "repo": c["repo"],
        "path": c["path"],
        "name": c["name"],
        "kind": c["kind"],
        "start_line": c["start_line"],
        "end_line": c["end_line"],
        "signature": c["signature"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed chunks and store them in Chroma")
    ap.add_argument("--chunks", default=CHUNKS_FILE)
    ap.add_argument("--model", choices=list(MODELS), default=None)
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks from {args.chunks}")

    cfg = model_config(args.model)
    print(f"Loading model {cfg['name']} (first run downloads it)...")
    model = load_model(cfg)

    t0 = time.time()
    embeddings = model.encode(
        [chunk_to_text(c, cfg["passage"]) for c in chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"Embedded in {time.time() - t0:.1f}s, dimension {embeddings.shape[1]}")

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(cfg["collection"])
    except Exception:
        pass
    collection = client.create_collection(cfg["collection"], metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings[i:i + BATCH].tolist(),
            documents=[c["code"] for c in batch],
            metadatas=[chunk_metadata(c) for c in batch],
        )

    print(f"Stored {collection.count()} chunks in {DB_PATH}/ (collection '{cfg['collection']}')")


if __name__ == "__main__":
    main()
