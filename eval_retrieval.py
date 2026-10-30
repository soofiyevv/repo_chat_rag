"""
Step 2c. Retrieval evaluation.

Runs every question from eval/questions.jsonl through search and checks
(negative questions without expected chunks are skipped here)
whether one of the expected chunks (path::name) appears in the top-k results.

Metrics:
  Hit@1   share of questions where a correct chunk is ranked first
  Hit@k   share of questions where a correct chunk is anywhere in the top-k
  MRR     mean of 1/rank of the first correct chunk (0 if not found)

The full report is written to eval/last_report.txt (UTF-8) and
each run is appended to eval/history.csv, so changes (model, chunking,
filters) can be compared over time.

Usage:
  python eval_retrieval.py
  python eval_retrieval.py --mode vector --note "e5-small"
  python eval_retrieval.py --mode bm25
  python eval_retrieval.py --model bge-m3
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from config import MODELS
from retriever import MODES, Retriever

QUESTIONS = Path("eval/questions.jsonl")
HISTORY = Path("eval/history.csv")
REPORT = Path("eval/last_report.txt")


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def first_hit_rank(results: list[dict], expected: list[str]) -> int | None:
    wanted = set(expected)
    for rank, r in enumerate(results, 1):
        if f"{r['path']}::{r['name'].split('#')[0]}" in wanted:
            return rank
    return None


def summarize(rows: list[dict], k: int) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "hit@1": sum(r["rank"] == 1 for r in rows) / n,
        f"hit@{k}": sum(r["rank"] is not None for r in rows) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
    }


def fmt(s: dict) -> str:
    return "  ".join(f"{key}={val:.2f}" if isinstance(val, float) else f"{key}={val}" for key, val in s.items())


def run(search_fn, questions: list[dict], k: int) -> list[dict]:
    rows = []
    for q in questions:
        if not q["expected"]:
            continue
        results = search_fn(q["question"], k)
        rank = first_hit_rank(results, q["expected"])
        top = f"{results[0]['path'].split('/')[-1]}::{results[0]['name']}" if results else "-"
        rows.append({**q, "rank": rank, "top": top})
    return rows


def report(rows: list[dict], k: int) -> tuple[dict, str]:
    out = [f"{'id':4} {'lang':4} {'rank':>4}  {'top result':40} question"]
    for r in rows:
        rank = str(r["rank"]) if r["rank"] else "-"
        out.append(f"{r['id']:4} {r['lang']:4} {rank:>4}  {r['top'][:40]:40} {r['question']}")

    total = summarize(rows, k)
    out.append("")
    out.append(f"ALL  {fmt(total)}")
    for lang in sorted({r["lang"] for r in rows}):
        out.append(f"{lang.upper():4} {fmt(summarize([r for r in rows if r['lang'] == lang], k))}")
    return total, "\n".join(out)


def save_history(total: dict, k: int, note: str, model_name: str) -> None:
    new = not HISTORY.exists()
    with open(HISTORY, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "model", "k", "n", "hit@1", "hit@k", "mrr", "note"])
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"), model_name, k, total["n"],
            f"{total['hit@1']:.3f}", f"{total[f'hit@{k}']:.3f}", f"{total['mrr']:.3f}", note,
        ])


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval quality")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=MODES, default="vector")
    ap.add_argument("--model", choices=list(MODELS), default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    retriever = Retriever(model=args.model)
    model_name = retriever.cfg["name"]
    rows = run(lambda q, k: retriever.search(q, k, args.mode), load_questions(QUESTIONS), args.k)
    note = f"[{args.mode}] {args.note}".strip()
    total, text = report(rows, args.k)
    print("\n" + text)
    REPORT.write_text(f"model: {model_name}  k={args.k}  note: {note}\n\n{text}\n", encoding="utf-8")
    save_history(total, args.k, note, model_name)
    print(f"\nSaved to {REPORT} and {HISTORY}")


if __name__ == "__main__":
    main()
