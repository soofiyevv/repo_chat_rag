"""
Step 7. Answer evaluation with an LLM judge.

For every question in eval/questions.jsonl the full RAG pipeline answers it
(retrieval + Groq), then a judge LLM grades the answer against the
reference facts and the retrieved code. The judge provider is set by
JUDGE_PROVIDER (groq or gemini, see judges.py).

--rejudge grades the answers already saved for --tag again with the current
judge, without generating new answers (no Groq tokens are used). Results go
to eval/runs/<tag>__<judge>.jsonl.

Judge scores (1-5):
  correctness   does the answer contain the reference facts, without errors
  groundedness  is every claim supported by the retrieved code (no inventions)
For negative questions (feature not in the repo) the answer must say so;
inventing an implementation is scored 1.

Computed without the judge:
  retrieval_hit   an expected chunk was among the retrieved sources
  cites_expected  the answer names an expected file or function

Summary metrics:
  pass rate      share of answers with correctness >= 4 and groundedness >= 4
  refusal rate   share of negative questions answered with "not found"

Results are cached per run tag in eval/runs/<tag>.jsonl, so an interrupted
run (rate limits) continues where it stopped. The summary is appended to
eval/answers_history.csv and written to eval/answers_report.txt.

Settings: GROQ_API_KEY, GROQ_MODEL (answers), JUDGE_PROVIDER, JUDGE_MODEL / GEMINI_MODEL.

Usage:
  python eval_answers.py --tag baseline
  python eval_answers.py --tag baseline --limit 5
  python eval_answers.py --tag k8 -k 8
  python eval_answers.py --tag baseline --rejudge
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

from config import MODELS

from groq import Groq

from judges import Judge
from rag import RepoChat, build_context
from retriever import Retriever, load_chunks

QUESTIONS = Path("eval/questions.jsonl")
RUNS = Path("eval/runs")
HISTORY = Path("eval/answers_history.csv")
REPORT = Path("eval/answers_report.txt")
ERROR_LOG = Path("eval/error_log.txt")

JUDGE_PROMPT = """You are a strict evaluator of answers about a code repository.

You get a question, reference facts written by a human who read the code,
the code chunks that were given to the assistant, and the assistant's answer.

Score two things from 1 to 5:

correctness:
  5 = covers all reference facts, no errors
  4 = covers the main facts, minor omissions, no errors
  3 = partially correct or missing important facts
  2 = mostly wrong or vague
  1 = wrong, or invents an implementation that does not exist

groundedness:
  5 = every claim is supported by the provided code chunks
  3 = some claims are not supported by the chunks but are plausible
  1 = contains invented functions, files, endpoints or behaviour

If the reference says the feature does not exist in the repository, the
correct answer says it could not find it. Such an answer gets correctness 5
and groundedness 5; describing an implementation gets 1 and 1.
Answers may be in Russian or English; judge the content, not the language.

Reply with JSON only:
{"correctness": <1-5>, "groundedness": <1-5>, "comment": "<one short sentence>"}"""


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def judge(judge_fn: Judge, q: dict, answer: str, sources: list[dict]) -> dict:
    facts = "\n".join(f"- {f}" for f in q["facts"])
    user = (
        f"Question: {q['question']}\n\nReference facts:\n{facts}\n\n"
        f"Code chunks given to the assistant:\n\n{build_context(sources)}\n\n"
        f"Assistant answer:\n{answer}"
    )
    text = judge_fn(JUDGE_PROMPT, user)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else {}
    return {
        "correctness": int(data.get("correctness", 0)),
        "groundedness": int(data.get("groundedness", 0)),
        "comment": str(data.get("comment", ""))[:300],
    }


def deterministic(q: dict, answer: str, sources: list[dict]) -> dict:
    if not q["expected"]:
        return {"retrieval_hit": None, "cites_expected": None}
    expected = set(q["expected"])
    hit = any(f"{s['path']}::{s['name'].split('#')[0]}" in expected for s in sources)
    low = answer.lower()
    keys = set()
    for e in q["expected"]:
        path, name = e.split("::")
        keys.add(path.split("/")[-1].lower())
        if not name.startswith("<"):
            keys.add(name.split(".")[-1].lower())
    return {"retrieval_hit": hit, "cites_expected": any(k in low for k in keys)}


def summarize(rows: list[dict]) -> dict:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    pos = [r for r in rows if r["type"] != "negative"]
    neg = [r for r in rows if r["type"] == "negative"]
    s = {
        "n": len(rows),
        "correctness": mean([r["correctness"] for r in rows]),
        "groundedness": mean([r["groundedness"] for r in rows]),
        "pass_rate": mean([1.0 if r["correctness"] >= 4 and r["groundedness"] >= 4 else 0.0 for r in rows]),
        "retrieval_hit": mean([1.0 if r["retrieval_hit"] else 0.0 for r in pos]) if pos else None,
        "cites_expected": mean([1.0 if r["cites_expected"] else 0.0 for r in pos]) if pos else None,
        "refusal_rate": mean([1.0 if r["correctness"] >= 4 else 0.0 for r in neg]) if neg else None,
        "latency_s": mean([r["latency_s"] for r in rows]),
    }
    return s


def fmt(s: dict) -> str:
    parts = []
    for k, v in s.items():
        if v is None:
            continue
        parts.append(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}")
    return "  ".join(parts)


def report(rows: list[dict], header: str) -> str:
    out = [header, "", f"{'id':4} {'lang':4} {'type':8} {'C':>2} {'G':>2} {'hit':>4} {'cite':>4}  comment"]
    for r in rows:
        hit = "-" if r["retrieval_hit"] is None else ("yes" if r["retrieval_hit"] else "NO")
        cite = "-" if r["cites_expected"] is None else ("yes" if r["cites_expected"] else "NO")
        out.append(f"{r['id']:4} {r['lang']:4} {r['type']:8} {r['correctness']:>2} {r['groundedness']:>2} {hit:>4} {cite:>4}  {r['comment']}")
    out += ["", f"ALL       {fmt(summarize(rows))}"]
    for t in ("locate", "explain", "negative"):
        sub = [r for r in rows if r["type"] == t]
        if sub:
            out.append(f"{t.upper():9} {fmt(summarize(sub))}")
    for lang in ("ru", "en"):
        sub = [r for r in rows if r["lang"] == lang]
        if sub:
            out.append(f"{lang.upper():9} {fmt(summarize(sub))}")

    worst = sorted((r for r in rows if r["correctness"] < 4 or r["groundedness"] < 4), key=lambda r: r["correctness"] + r["groundedness"])
    if worst:
        out += ["", "FAILED ANSWERS", ""]
        for r in worst:
            out += [f"--- {r['id']} C={r['correctness']} G={r['groundedness']}  {r['question']}",
                    f"sources: {', '.join(s.split('/')[-1] for s in r['source_ids'])}",
                    r["answer"].strip(), ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RAG answers with an LLM judge")
    ap.add_argument("--tag", required=True, help="run name, results are cached under eval/runs/<tag>.jsonl")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--model", choices=list(MODELS), default=None, help="embedding model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=2.0, help="pause between questions to respect rate limits")
    ap.add_argument("--fresh", action="store_true", help="ignore cached results for this tag")
    ap.add_argument("--rejudge", action="store_true", help="re-grade saved answers of --tag with the current judge")
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    judge_fn = Judge()
    judge_model = judge_fn.name
    tag = args.tag
    source_answers = {}
    if args.rejudge:
        src_file = RUNS / f"{args.tag}.jsonl"
        if not src_file.exists():
            raise SystemExit(f"No saved answers for tag '{args.tag}' ({src_file})")
        source_answers = {r["id"]: r for r in load_jsonl(src_file)}
        tag = f"{args.tag}__{judge_fn.short}"
    run_file = RUNS / f"{tag}.jsonl"
    if args.fresh and run_file.exists():
        run_file.unlink()

    questions = load_jsonl(QUESTIONS)
    if args.limit:
        questions = questions[: args.limit]
    if args.rejudge:
        questions = [q for q in questions if q["id"] in source_answers]
    done = {r["id"]: r for r in load_jsonl(run_file)}

    if args.rejudge:
        by_key = {f"{c['path']}::{c['name']}": c for c in load_chunks("chunks.jsonl")}
        chat = None
        llm_model = "(saved answers)"
        emb_key = emb_name = "(saved)"
    else:
        chat = RepoChat(retriever=Retriever(model=args.model), k=args.k)
        chat.client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=8)
        llm_model = chat.llm_model
        emb_key, emb_name = chat.retriever.cfg["key"], chat.retriever.cfg["name"]

    todo = [q for q in questions if q["id"] not in done]
    print(f"{len(questions)} questions, {len(done)} cached, {len(todo)} to run  |  answers: {llm_model}  judge: {judge_model}")

    for i, q in enumerate(todo, 1):
        t0 = time.perf_counter()
        print(f"[{i}/{len(todo)}] {q['id']} ...", end="\r", flush=True)
        try:
            if args.rejudge:
                saved = source_answers[q["id"]]
                result = {"answer": saved["answer"], "sources": [by_key[s] for s in saved["source_ids"] if s in by_key]}
                latency = saved["latency_s"]
            else:
                result = chat.ask(q["question"])
                latency = time.perf_counter() - t0
            scores = judge(judge_fn, q, result["answer"], result["sources"])
        except Exception as exc:
            with open(ERROR_LOG, "a", encoding="utf-8") as fh:
                fh.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S}  tag={tag}  {q['id']}\n")
                fh.write(traceback.format_exc())
            print(f"\nStopped at {q['id']}: {type(exc).__name__}: {str(exc)[:400]}")
            print(f"Progress is saved ({len(done)} done). Details: {ERROR_LOG}. Run the same command again to continue.")
            raise SystemExit(1)
        row = {
            **{k: q[k] for k in ("id", "lang", "type", "question")},
            "answer": result["answer"],
            "source_ids": [f"{s['path']}::{s['name']}" for s in result["sources"]],
            "latency_s": round(latency, 2),
            **deterministic(q, result["answer"], result["sources"]),
            **scores,
        }
        with open(run_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[q["id"]] = row
        print(f"[{i}/{len(todo)}] {q['id']} C={row['correctness']} G={row['groundedness']}  {q['question'][:60]}")
        if i < len(todo):
            time.sleep(args.sleep)

    rows = [done[q["id"]] for q in questions if q["id"] in done]
    header = (f"tag: {tag}  embedding: {emb_name}  k={args.k}  "
              f"llm: {llm_model}  judge: {judge_model}")
    text = report(rows, header)
    REPORT.write_text(text + "\n", encoding="utf-8")
    print("\n" + text.split("\nFAILED ANSWERS")[0])

    s = summarize(rows)
    new = not HISTORY.exists()
    with open(HISTORY, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "tag", "embedding", "k", "llm", "judge", *s.keys()])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), tag, emb_key, args.k,
                    llm_model, judge_model, *[f"{v:.3f}" if isinstance(v, float) else ("" if v is None else v) for v in s.values()]])
    print(f"\nFull report with failed answers: {REPORT}\nHistory: {HISTORY}")


if __name__ == "__main__":
    main()
