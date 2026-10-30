"""
Formatting helpers for Telegram.

The LLM answers in Markdown. Telegram supports only a small HTML subset,
so the answer is escaped and converted:
  `code` -> <code>, **bold** -> <b>, ### heading -> <b>, * item -> • item,
  ```blocks``` -> <pre>.
Long messages are split into parts under Telegram's 4096 character limit.
"""
from __future__ import annotations

import html
import re

TG_LIMIT = 4000


def md_to_tg_html(text: str) -> str:
    blocks = []

    def keep_block(m: re.Match) -> str:
        blocks.append(f"<pre>{html.escape(m.group(2).strip())}</pre>")
        return f"\x00{len(blocks) - 1}\x00"

    text = re.sub(r"```(\w*)\n?(.*?)```", keep_block, text, flags=re.S)
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?m)^#{1,6}\s*(.+)$", r"<b>\1</b>", text)
    text = re.sub(r"(?m)^(\s*)[*-]\s+", r"\1• ", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: blocks[int(m.group(1))], text)
    return text.strip()


def format_sources(sources: list[dict]) -> str:
    lines = ["<b>Sources</b>"]
    for i, s in enumerate(sources, 1):
        loc = f"{s['path']}:{s['start_line']}-{s['end_line']}"
        lines.append(f"[{i}] <code>{html.escape(loc)}</code> {html.escape(s['name'])}")
    return "\n".join(lines)


def split_message(text: str, limit: int = TG_LIMIT) -> list[str]:
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts
