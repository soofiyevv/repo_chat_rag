"""
Step 6. Telegram bot (aiogram 3).

A thin client for the FastAPI server: every text message is sent to
POST /ask, and the answer is returned with its source files and lines.

Commands:
  /start   welcome picture with a short intro
  /help    how the bot works

Settings from .env:
  TELEGRAM_BOT_TOKEN   required, from @BotFather
  API_URL              FastAPI address, default http://127.0.0.1:8000
  API_TOKEN            optional, sent as X-API-Key if the API requires it

Usage (the API must be running first):
  uvicorn api:app
  python bot.py
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, FSInputFile, Message
from dotenv import load_dotenv

from tg_format import format_sources, md_to_tg_html, split_message

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN") or ""
WELCOME_IMAGE = Path(__file__).parent / "assets" / "welcome.png"
TIMEOUT = aiohttp.ClientTimeout(total=90)

WELCOME = """👋 <b>Hi! I'm Repo Chat</b> — I answer questions about a codebase.

Currently indexed: <code>{repo}</code>

Just ask in plain language (English or Russian), for example:
• Where is the user password checked?
• How does password recovery work?
• Как создаётся JWT-токен?

I'll find the relevant code and reply with file paths and line numbers.

/help — how it works"""

HELP = """<b>How it works</b>

1. Your question is turned into an embedding (bge-m3).
2. The most similar functions and classes are found in a Chroma index of the repository.
3. An LLM (Groq) reads only those code pieces and answers, citing files and lines.

If the code does not contain the answer, I'll say so instead of guessing.

Tips: ask about one thing at a time and use names from the code if you know them."""

dp = Dispatcher()
http: aiohttp.ClientSession | None = None


def headers() -> dict:
    return {"X-API-Key": API_TOKEN} if API_TOKEN else {}


async def api_get(path: str) -> dict:
    async with http.get(f"{API_URL}{path}", headers=headers()) as r:
        r.raise_for_status()
        return await r.json()


async def api_post(path: str, payload: dict) -> dict:
    async with http.post(f"{API_URL}{path}", json=payload, headers=headers()) as r:
        if r.status >= 400:
            raise RuntimeError(f"API {r.status}: {await r.text()}")
        return await r.json()


async def keep_typing(message: Message, stop: asyncio.Event) -> None:
    while not stop.is_set():
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        try:
            await asyncio.wait_for(stop.wait(), timeout=4)
        except asyncio.TimeoutError:
            pass


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    try:
        repo = (await api_get("/health")).get("repo") or "unknown"
    except Exception:
        repo = "unavailable (API is offline)"
    text = WELCOME.format(repo=repo)
    if WELCOME_IMAGE.exists():
        await message.answer_photo(FSInputFile(WELCOME_IMAGE), caption=text)
    else:
        await message.answer(text)


@dp.message(Command("help"))
async def on_help(message: Message) -> None:
    await message.answer(HELP)


@dp.message(F.text & ~F.text.startswith("/"))
async def on_question(message: Message) -> None:
    question = message.text.strip()
    if len(question) < 2:
        await message.answer("Please ask a longer question.")
        return

    stop = asyncio.Event()
    typing = asyncio.create_task(keep_typing(message, stop))
    try:
        data = await api_post("/ask", {"question": question[:1000]})
    except aiohttp.ClientConnectorError:
        await message.answer("⚠️ The API server is not running. Start it with <code>uvicorn api:app</code>.")
        return
    except Exception as e:
        logging.exception("ask failed")
        await message.answer(f"⚠️ Something went wrong: <code>{str(e)[:300]}</code>")
        return
    finally:
        stop.set()
        await typing

    text = md_to_tg_html(data["answer"]) + "\n\n" + format_sources(data["sources"])
    text += f"\n\n<i>{data['took_ms'] / 1000:.1f}s · {data['llm_model']}</i>"
    for part in split_message(text):
        try:
            await message.answer(part, disable_web_page_preview=True)
        except TelegramBadRequest:
            await message.answer(part, parse_mode=None, disable_web_page_preview=True)


@dp.message()
async def on_other(message: Message) -> None:
    await message.answer("Send me a text question about the code. /help")


async def main() -> None:
    global http
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set in .env")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bot = Bot(token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await bot.set_my_commands([
        BotCommand(command="start", description="Welcome and examples"),
        BotCommand(command="help", description="How it works"),
    ])
    http = aiohttp.ClientSession(timeout=TIMEOUT)
    try:
        await dp.start_polling(bot)
    finally:
        await http.close()


if __name__ == "__main__":
    asyncio.run(main())
