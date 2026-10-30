"""
Step 3. Hybrid retriever.

Combines two ways of finding code chunks:
  vector  semantic search with embeddings in Chroma (understands meaning,
          works across Russian and English)
  bm25    keyword search over identifiers, paths, docstrings and code
          (exact names like send_email or login.py)

Code identifiers are split into words (get_current_user -> get, current, user;
createAccessToken -> create, access, token) before keyword matching.

The two ranked lists are merged with Reciprocal Rank Fusion (RRF):
  score(chunk) = sum over lists of 1 / (RRF_K + rank)
so a chunk ranked high by either method rises to the top.
In vector mode the score is cosine similarity (0..1); in hybrid and bm25
modes it is the RRF score.
Test files are excluded unless include_tests=True.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter

from config import CHUNKS_FILE, DB_PATH, load_model, model_config

RRF_K = 60
CANDIDATES = 20
MODES = ("hybrid", "vector", "bm25")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "be", "to", "of", "in", "on", "for", "and",
    "or", "it", "its", "by", "with", "from", "at", "as", "this", "that", "do", "does",
    "how", "where", "what", "which", "who", "when", "why", "can", "i", "we", "you",
    "self", "none", "true", "false", "return", "def", "class", "import",
}


def load_chunks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def is_test(path: str) -> bool:
    parts = path.lower().split("/")
    name = parts[-1]
    return (
        any(p in ("tests", "test") for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", text):
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[А-Яа-яЁё]+|\d+", raw.replace("_", " "))
        words = parts if len(parts) == 1 else [raw] + parts
        for w in words:
            w = _stem(w.lower().strip("_"))
            if len(w) > 1 and w not in STOPWORDS:
                tokens.append(w)
    return tokens


def chunk_search_text(c: dict) -> str:
    return " ".join([c["path"], c["name"], c["signature"], c["docstring"], c["code"]])


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t in query:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out


class Retriever:
    def __init__(self, chunks_file: str = CHUNKS_FILE, include_tests: bool = False, model: str | None = None):
        self.cfg = model_config(model)
        self.include_tests = include_tests
        self.chunks = [c for c in load_chunks(chunks_file) if include_tests or not is_test(c["path"])]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25([tokenize(chunk_search_text(c)) for c in self.chunks])
        self._model = None
        self._collection = None

    def _load_vector(self) -> None:
        if self._model is None:
            import chromadb

            self._model = load_model(self.cfg)
            try:
                self._collection = chromadb.PersistentClient(path=DB_PATH).get_collection(self.cfg["collection"])
            except Exception:
                raise SystemExit(f"No index for model '{self.cfg['key']}'. Run: python index.py --model {self.cfg['key']}")

    def vector_hits(self, question: str, n: int) -> list[tuple[str, float]]:
        self._load_vector()
        emb = self._model.encode([self.cfg["query"] + question], normalize_embeddings=True)
        res = self._collection.query(
            query_embeddings=emb.tolist(),
            n_results=n,
            where=None if self.include_tests else {"is_test": False},
            include=["distances"],
        )
        return [(i, 1 - d) for i, d in zip(res["ids"][0], res["distances"][0]) if i in self.by_id]

    def vector_ids(self, question: str, n: int) -> list[str]:
        return [i for i, _ in self.vector_hits(question, n)]

    def bm25_ids(self, question: str, n: int) -> list[str]:
        scores = self.bm25.scores(tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i]["id"] for i in ranked[:n] if scores[i] > 0]

    def search(self, question: str, k: int = 5, mode: str = "hybrid") -> list[dict]:
        if mode == "vector":
            return [{**self.by_id[cid], "score": sim} for cid, sim in self.vector_hits(question, k)]
        if mode == "bm25":
            lists = [self.bm25_ids(question, k)]
        else:
            lists = [self.vector_ids(question, CANDIDATES), self.bm25_ids(question, CANDIDATES)]

        fused: dict[str, float] = {}
        for ids in lists:
            for rank, cid in enumerate(ids, 1):
                fused[cid] = fused.get(cid, 0.0) + 1 / (RRF_K + rank)

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{**self.by_id[cid], "score": fused[cid]} for cid in top]
