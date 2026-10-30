"""
Citation accuracy check (no LLM needed).

Parses every citation of the form  [n] path:start-end  in the cached answers
from eval/runs/<tag>.jsonl and checks it against the chunks that were
actually given to the model:
  valid       source [n] exists, the path matches it and the lines lie
              inside that chunk's line range
  wrong_path  [n] exists but points to a different file
  wrong_lines path is right but the lines are outside the chunk
  bad_index   [n] does not exist

Reports per run: answers with citations, citation count, accuracy, and
the list of invalid citations.

Usage:
  python eval_citations.py
  python eval_citations.py baseline v2-lines
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from retriever import load_chunks

RUNS = Path("eval/runs")
CITE = re.compile(r"\[(\d+)\][\s*`【】]*([\w./-]+\.py)\s*:\s*(\d+)(?:\s*[-‐‑–—]\s*(\d+))?")


def chunk_ranges() -> dict[str, tuple[int, int]]:
    return {f"{c['path']}::{c['name']}": (c["start_line"], c["end_line"]) for c in load_chunks("chunks.jsonl")}


def check_answer(answer: str, source_ids: list[str], ranges: dict) -> list[dict]:
    out = []
    for m in CITE.finditer(answer.replace("‑", "-")):
        n, path, a = int(m.group(1)), m.group(2), int(m.group(3))
        b = int(m.group(4) or a)
        if not 1 <= n <= len(source_ids):
            status = "bad_index"
        else:
            src_path = source_ids[n - 1].split("::")[0]
            lo, hi = ranges.get(source_ids[n - 1], (0, 0))
            if src_path.split("/")[-1] != path.split("/")[-1]:
                status = "wrong_path"
            elif lo <= a <= b <= hi:
                status = "valid"
            else:
                status = "wrong_lines"
        out.append({"cite": m.group(0), "n": n, "status": status})
    return out


def evaluate(tag: str, ranges: dict) -> None:
    rows = [json.loads(l) for l in open(RUNS / f"{tag}.jsonl", encoding="utf-8") if l.strip()]
    cites, bad = [], []
    with_cites = 0
    for r in rows:
        res = check_answer(r["answer"], r["source_ids"], ranges)
        with_cites += bool(res)
        cites += res
        bad += [(r["id"], c, r["source_ids"][c["n"] - 1] if 1 <= c["n"] <= len(r["source_ids"]) else "-") for c in res if c["status"] != "valid"]
    total = len(cites) or 1
    counts = {s: sum(c["status"] == s for c in cites) for s in ("valid", "wrong_path", "wrong_lines", "bad_index")}
    print(f"\n{tag}: {len(rows)} answers, {with_cites} with citations, {len(cites)} citations, "
          f"accuracy={counts['valid'] / total:.2f}  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    for qid, c, src in bad:
        print(f"  {qid} {c['status']:11} {c['cite']!r:55} source was {src}")


def main() -> None:
    tags = sys.argv[1:] or sorted(p.stem for p in RUNS.glob("*.jsonl"))
    ranges = chunk_ranges()
    for tag in tags:
        evaluate(tag, ranges)


if __name__ == "__main__":
    main()
