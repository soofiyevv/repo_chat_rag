from tg_format import format_sources, md_to_tg_html, split_message


def test_markdown_to_telegram_html():
    out = md_to_tg_html("**Bold** and `code` <tag> & more\n* item\n### Title")
    assert "<b>Bold</b>" in out
    assert "<code>code</code>" in out
    assert "&lt;tag&gt; &amp; more" in out
    assert "• item" in out
    assert "<b>Title</b>" in out


def test_code_blocks_are_escaped():
    out = md_to_tg_html("```python\nif a < b:\n    pass\n```")
    assert out == "<pre>if a &lt; b:\n    pass</pre>"


def test_sources_are_escaped():
    out = format_sources([{"path": "a.py", "start_line": 1, "end_line": 2, "name": "f<x>"}])
    assert "<code>a.py:1-2</code> f&lt;x&gt;" in out


def test_split_message_respects_limit():
    parts = split_message("a" * 3000 + "\n\n" + "b" * 3000, limit=4000)
    assert len(parts) == 2 and all(len(p) <= 4000 for p in parts)
