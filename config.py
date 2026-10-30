"""
Shared settings for indexing, search and evaluation.

MODELS          embedding models that can be compared; each one gets its own
                Chroma collection, so switching models never mixes vectors
DEFAULT_MODEL   used when --model is not given (or set EMBED_MODEL env var)
DB_PATH         folder where Chroma stores the vector index
CHUNKS_FILE     output of chunker.py

e5 models expect "query: " / "passage: " prefixes; bge-m3 uses none.
"""
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

MODELS = {
    "e5-small": {"name": "intfloat/multilingual-e5-small", "query": "query: ", "passage": "passage: "},
    "e5-base": {"name": "intfloat/multilingual-e5-base", "query": "query: ", "passage": "passage: "},
    "bge-m3": {"name": "BAAI/bge-m3", "query": "", "passage": ""},
}
DEFAULT_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
MAX_SEQ_LENGTH = 512

DB_PATH = "chroma_db"
CHUNKS_FILE = "chunks.jsonl"


def model_config(key: str | None = None) -> dict:
    key = key or DEFAULT_MODEL
    if key not in MODELS:
        raise SystemExit(f"Unknown model '{key}'. Choose from: {', '.join(MODELS)}")
    return {**MODELS[key], "key": key, "collection": f"code_chunks__{key}"}


def load_model(cfg: dict):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg["name"])
    model.max_seq_length = min(model.max_seq_length or MAX_SEQ_LENGTH, MAX_SEQ_LENGTH)
    return model
