"""
Step 5. HTTP API with FastAPI.

Loads the embedding model, the Chroma index and the Groq client once at
startup and serves:

  GET  /health   liveness check and loaded settings
  POST /search   retrieval only: top-k chunks for a question
  POST /ask      full RAG: answer from the LLM plus the source chunks

If API_TOKEN is set in .env, every request except /health must send the
header  X-API-Key: <API_TOKEN>.  Interactive docs are at /docs.

Usage:
  uvicorn api:app --reload
  curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d "{\\"question\\": \\"how are emails sent?\\"}"
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from config import model_config

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from rag import RepoChat
from retriever import MODES, Retriever

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever = Retriever()
    retriever.search("warm up", 1, "vector")
    state["chat"] = RepoChat(retriever=retriever)
    yield
    state.clear()


app = FastAPI(title="Repo Chat", version="0.1.0", lifespan=lifespan)


def check_token(x_api_key: str | None = Header(default=None)) -> None:
    token = os.getenv("API_TOKEN")
    if token and x_api_key != token:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)


class SearchRequest(Question):
    mode: str = Field(default="vector", pattern="^(" + "|".join(MODES) + ")$")


class Source(BaseModel):
    path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    score: float
    code: str


class SearchResponse(BaseModel):
    sources: list[Source]
    took_ms: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    llm_model: str
    took_ms: int


def to_sources(chunks: list[dict]) -> list[Source]:
    return [Source(**{f: c[f] for f in Source.model_fields}) for c in chunks]


@app.get("/health")
def health() -> dict:
    chat: RepoChat | None = state.get("chat")
    return {
        "status": "ok" if chat else "starting",
        "embedding_model": model_config()["name"],
        "llm_model": chat.llm_model if chat else None,
        "chunks": len(chat.retriever.chunks) if chat else 0,
    }


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(check_token)])
def search(req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()
    chunks = state["chat"].retriever.search(req.question, req.k, req.mode)
    return SearchResponse(sources=to_sources(chunks), took_ms=int((time.perf_counter() - t0) * 1000))


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(check_token)])
def ask(req: Question) -> AskResponse:
    t0 = time.perf_counter()
    try:
        result = state["chat"].ask(req.question, req.k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}") from e
    return AskResponse(
        answer=result["answer"],
        sources=to_sources(result["sources"]),
        llm_model=result["model"],
        took_ms=int((time.perf_counter() - t0) * 1000),
    )
