import json
import textwrap
from pathlib import Path

import chunker


def write(tmp: Path, rel: str, code: str) -> None:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(code))


def test_functions_methods_and_classes(tmp_path):
    write(tmp_path, "pkg/auth.py", '''
        import os
        LIMIT = 5

        def login(user, password):
            """Check the password."""
            return password == "x"

        class Service:
            """Auth service."""
            retries = 3

            def check(self, token):
                return bool(token)
        ''')
    chunks = chunker.chunk_file(tmp_path / "pkg/auth.py", tmp_path, "demo")
    by_name = {c.name: c for c in chunks}

    assert by_name["login"].kind == "function"
    assert by_name["login"].docstring == "Check the password."
    assert by_name["Service.check"].kind == "method"
    assert by_name["Service"].kind == "class"
    assert "def check(self, token): ..." in by_name["Service"].code
    assert by_name["<module>"].code.strip() == "LIMIT = 5"
    assert all("import os" not in c.code for c in chunks)
    assert by_name["login"].start_line == 5


def test_long_chunks_are_split(tmp_path):
    body = "\n".join(f"    x{i} = {i}" for i in range(300))
    write(tmp_path, "big.py", "def big():\n" + body + "\n")
    parts = chunker.chunk_file(tmp_path / "big.py", tmp_path, "demo")
    assert len(parts) > 1
    assert all(len(p.code.splitlines()) <= chunker.MAX_LINES for p in parts)
    assert parts[1].start_line == parts[0].start_line + chunker.MAX_LINES - chunker.OVERLAP


def test_unparseable_file_falls_back_to_lines(tmp_path):
    write(tmp_path, "broken.py", "def f(:\n    pass\n")
    chunks = chunker.chunk_file(tmp_path / "broken.py", tmp_path, "demo")
    assert len(chunks) == 1 and chunks[0].kind == "file"


def test_chunks_are_json_serializable(tmp_path):
    write(tmp_path, "a.py", "def f():\n    return 1\n")
    from dataclasses import asdict
    for c in chunker.chunk_file(tmp_path / "a.py", tmp_path, "demo"):
        json.dumps(asdict(c))
