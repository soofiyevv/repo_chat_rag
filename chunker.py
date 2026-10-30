"""
Step 1. Code chunker.

Walks a repository, parses every .py file with ast and splits it into chunks:
  - function : a top-level function
  - method   : a class method (named Class.method)
  - class    : a class summary (header, docstring, attributes, method signatures)
  - module   : module-level code outside functions and classes (imports are skipped)
  - file     : fallback for files that cannot be parsed (split by line windows)

Chunks longer than MAX_LINES are split into overlapping parts.
Output: chunks.jsonl, one JSON object per line.

Usage:
  python chunker.py https://github.com/user/repo
  python chunker.py ./local/repo -o chunks.jsonl
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    "build", "dist", ".tox", ".mypy_cache", ".pytest_cache", "site-packages",
}
MAX_LINES = 120
OVERLAP = 15


@dataclass
class Chunk:
    id: str
    repo: str
    path: str
    kind: str
    name: str
    start_line: int
    end_line: int
    signature: str
    docstring: str
    code: str


def make_id(path: str, name: str, start: int) -> str:
    return hashlib.sha1(f"{path}:{name}:{start}".encode()).hexdigest()[:16]


def segment(lines: list[str], node: ast.AST) -> tuple[int, int, str]:
    start = node.lineno
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        start = min(d.lineno for d in decorators)
    end = node.end_lineno
    return start, end, "\n".join(lines[start - 1:end])


def func_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){ret}"


def class_signature(node: ast.ClassDef) -> str:
    bases = [ast.unparse(b) for b in node.bases] + [ast.unparse(k) for k in node.keywords]
    return f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"


def is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def split_long(chunk: Chunk) -> list[Chunk]:
    lines = chunk.code.splitlines()
    if len(lines) <= MAX_LINES:
        return [chunk]
    parts, step = [], MAX_LINES - OVERLAP
    for i, s in enumerate(range(0, len(lines), step), 1):
        piece = lines[s:s + MAX_LINES]
        start = chunk.start_line + s
        name = f"{chunk.name}#part{i}"
        parts.append(replace(
            chunk,
            id=make_id(chunk.path, name, start),
            name=name,
            start_line=start,
            end_line=start + len(piece) - 1,
            code="\n".join(piece),
        ))
        if s + MAX_LINES >= len(lines):
            break
    return parts


def chunk_file(path: Path, repo_root: Path, repo_name: str) -> list[Chunk]:
    source = path.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()
    rel = path.relative_to(repo_root).as_posix()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"[fallback] {rel}: {e.msg} (line {e.lineno})", file=sys.stderr)
        if not lines:
            return []
        whole = Chunk(id=make_id(rel, "<file>", 1), repo=repo_name, path=rel,
                      kind="file", name="<file>", start_line=1, end_line=len(lines),
                      signature="", docstring="", code="\n".join(lines))
        return split_long(whole)

    chunks: list[Chunk] = []

    def add(kind, name, start, end, signature, doc, code):
        chunks.append(Chunk(
            id=make_id(rel, name, start), repo=repo_name, path=rel, kind=kind,
            name=name, start_line=start, end_line=end, signature=signature,
            docstring=doc or "", code=code,
        ))

    def visit_class(node: ast.ClassDef, prefix: str) -> None:
        qual = f"{prefix}{node.name}"
        start, end, _ = segment(lines, node)
        summary = [class_signature(node) + ":"]
        doc = ast.get_docstring(node)
        if doc:
            summary.append(f'    """{doc}"""')

        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                summary.append(f"    {func_signature(stmt)}: ...")
                s, e, code = segment(lines, stmt)
                add("method", f"{qual}.{stmt.name}", s, e,
                    func_signature(stmt), ast.get_docstring(stmt), code)
            elif isinstance(stmt, ast.ClassDef):
                summary.append(f"    {class_signature(stmt)}: ...")
                visit_class(stmt, f"{qual}.")
            elif is_docstring(stmt):
                continue
            else:
                summary.append("\n".join(lines[stmt.lineno - 1:stmt.end_lineno]))

        add("class", qual, start, end, class_signature(node), doc, "\n".join(summary))

    module_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s, e, code = segment(lines, node)
            add("function", node.name, s, e,
                func_signature(node), ast.get_docstring(node), code)
        elif isinstance(node, ast.ClassDef):
            visit_class(node, "")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        elif is_docstring(node) and node is tree.body[0]:
            continue
        else:
            module_nodes.append(node)

    if module_nodes:
        code = "\n".join("\n".join(lines[n.lineno - 1:n.end_lineno]) for n in module_nodes)
        add("module", "<module>", module_nodes[0].lineno, module_nodes[-1].end_lineno,
            "", ast.get_docstring(tree), code)

    return [part for c in chunks for part in split_long(c)]


def resolve_repo(source: str) -> tuple[Path, str]:
    if source.startswith(("http://", "https://", "git@")):
        name = source.rstrip("/").split("/")[-1].removesuffix(".git")
        target = Path(tempfile.mkdtemp(prefix="repo_")) / name
        subprocess.run(["git", "clone", "--depth", "1", source, str(target)], check=True)
        return target, name
    path = Path(source).resolve()
    return path, path.name


def iter_py_files(root: Path):
    for p in sorted(root.rglob("*.py")):
        if not any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            yield p


def main() -> None:
    ap = argparse.ArgumentParser(description="Split a Python repository into chunks")
    ap.add_argument("repo", help="local path or GitHub URL")
    ap.add_argument("-o", "--out", default="chunks.jsonl")
    args = ap.parse_args()

    root, repo_name = resolve_repo(args.repo)
    files = list(iter_py_files(root))
    all_chunks = [c for f in files for c in chunk_file(f, root, repo_name)]

    with open(args.out, "w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    kinds = Counter(c.kind for c in all_chunks)
    sizes = sorted(all_chunks, key=lambda c: len(c.code.splitlines()), reverse=True)
    print(f"Repository: {repo_name}  |  files: {len(files)}  |  chunks: {len(all_chunks)}")
    print("By kind:", dict(kinds))
    print("Longest chunks:")
    for c in sizes[:5]:
        print(f"  {len(c.code.splitlines()):4d} lines  {c.path}::{c.name}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
