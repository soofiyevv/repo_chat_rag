"""
Search code chunks from the command line.

Uses vector search by default (best on the eval so far); --mode switches to
hybrid or bm25-only to compare. Test files are skipped unless --tests.
Without a question argument it starts an interactive loop (empty line exits).

Usage:
  python search.py "where is the password checked?"
  python search.py "где обрабатывается авторизация?" -k 3
  python search.py "send_email" --mode bm25
  python search.py "где логин?" --model bge-m3
  python search.py
"""
from __future__ import annotations

import argparse

from config import MODELS
from retriever import MODES, Retriever

PREVIEW_LINES = 6


def show(results: list[dict]) -> None:
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r['score']:.4f}] {r['path']}:{r['start_line']}-{r['end_line']}  {r['name']} ({r['kind']})")
        lines = r["code"].splitlines()
        for line in lines[:PREVIEW_LINES]:
            print(f"     {line}")
        if len(lines) > PREVIEW_LINES:
            print(f"     ... (+{len(lines) - PREVIEW_LINES} lines)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Search code chunks")
    ap.add_argument("question", nargs="?")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=MODES, default="vector")
    ap.add_argument("--model", choices=list(MODELS), default=None)
    ap.add_argument("--tests", action="store_true", help="include test files in results")
    args = ap.parse_args()

    retriever = Retriever(include_tests=args.tests, model=args.model)

    if args.question:
        show(retriever.search(args.question, args.k, args.mode))
        return

    while True:
        q = input("\nQuestion (empty to exit): ").strip()
        if not q:
            break
        show(retriever.search(q, args.k, args.mode))


if __name__ == "__main__":
    main()
