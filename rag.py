"""
Step 4. Answer questions about the repository with an LLM (RAG).

1. Retrieve the top-k code chunks for the question (retriever.py).
2. Put them into a prompt as numbered sources with file path and lines.
3. Ask the Groq LLM to answer only from these sources and cite them as
   [n] path:start-end. The answer is written in the language of the question.

Settings come from .env (see .env.example):
  GROQ_API_KEY   required, free key from console.groq.com
  GROQ_MODEL     optional, default openai/gpt-oss-120b

Usage:
  python rag.py "где проверяется пароль пользователя?"
  python rag.py "how are emails sent?" --show-context
  python rag.py
"""
from __future__ import annotations

import argparse
import os

from config import MODELS

from dotenv import load_dotenv
from groq import Groq

from retriever import MODES, Retriever

load_dotenv()

DEFAULT_LLM = "openai/gpt-oss-120b"
TOP_K = 5

SYSTEM_PROMPT = """You are an assistant that answers questions about a software repository.
You get numbered source chunks of code. Rules:
- Answer only from the sources. If they do not contain the answer, say so plainly.
- Point to the exact place: file path, line range and function or class name.
- Cite sources inline as [n] path:start-end, for example [2] backend/app/crud.py:45-60.
- Explain briefly what the code does and how the pieces connect.
- Answer in the same language as the question.
- Keep it short: a few sentences, a short list if there are several places."""


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['path']}:{c['start_line']}-{c['end_line']}  {c['name']} ({c['kind']})"
        blocks.append(f"{header}\n```python\n{c['code']}\n```")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    user = f"Sources:\n\n{build_context(chunks)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class RepoChat:
    def __init__(self, retriever: Retriever | None = None, llm_model: str | None = None, k: int = TOP_K, mode: str = "vector"):
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise SystemExit("GROQ_API_KEY is not set. Copy .env.example to .env and put your key there.")
        self.client = Groq(api_key=key)
        self.llm_model = llm_model or os.getenv("GROQ_MODEL") or DEFAULT_LLM
        self.retriever = retriever or Retriever()
        self.k = k
        self.mode = mode

    def ask(self, question: str) -> dict:
        chunks = self.retriever.search(question, self.k, self.mode)
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=build_messages(question, chunks),
            temperature=0.2,
        )
        return {
            "question": question,
            "answer": response.choices[0].message.content.strip(),
            "sources": chunks,
            "model": self.llm_model,
        }


def show(result: dict, show_context: bool = False) -> None:
    print(f"\n{result['answer']}\n")
    print("Sources:")
    for i, c in enumerate(result["sources"], 1):
        print(f"  [{i}] {c['path']}:{c['start_line']}-{c['end_line']}  {c['name']}")
    if show_context:
        print("\n--- context sent to the LLM ---\n")
        print(build_context(result["sources"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask questions about the indexed repository")
    ap.add_argument("question", nargs="?")
    ap.add_argument("-k", type=int, default=TOP_K)
    ap.add_argument("--mode", choices=MODES, default="vector")
    ap.add_argument("--model", choices=list(MODELS), default=None, help="embedding model")
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()

    chat = RepoChat(retriever=Retriever(model=args.model), k=args.k, mode=args.mode)

    if args.question:
        show(chat.ask(args.question), args.show_context)
        return

    while True:
        q = input("\nQuestion (empty to exit): ").strip()
        if not q:
            break
        show(chat.ask(q), args.show_context)


if __name__ == "__main__":
    main()
