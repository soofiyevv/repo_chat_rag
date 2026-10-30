"""
LLM judges for answer evaluation.

A judge takes a system prompt and a user message and returns raw text
(expected to be JSON). Two providers are supported:
  groq    GROQ_API_KEY,   model JUDGE_MODEL   (default openai/gpt-oss-120b)
  gemini  GEMINI_API_KEY, model GEMINI_MODEL  (default gemini-3.8-flash)

Using a judge from a different provider than the answering model avoids
self-preference bias and keeps the Groq daily token budget for answers.
Rate-limit errors (429) are retried with growing pauses.
"""
from __future__ import annotations

import os
import time

DEFAULTS = {"groq": "openai/gpt-oss-120b", "gemini": "gemini-3.8-flash"}
RETRY_WAITS = [10, 20, 40, 60, 60, 60]


class Judge:
    def __init__(self, provider: str | None = None):
        self.provider = (provider or os.getenv("JUDGE_PROVIDER") or "groq").lower()
        if self.provider == "groq":
            from groq import Groq

            self.model = os.getenv("JUDGE_MODEL") or DEFAULTS["groq"]
            self.client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=8)
        elif self.provider == "gemini":
            from google import genai

            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise SystemExit("GEMINI_API_KEY is not set in .env (get one at aistudio.google.com)")
            from google.genai import types

            self.model = os.getenv("GEMINI_MODEL") or DEFAULTS["gemini"]
            try:
                http = types.HttpOptions(timeout=90_000, retry_options=types.HttpRetryOptions(attempts=1))
            except (AttributeError, TypeError):
                http = types.HttpOptions(timeout=90_000)
            self.client = genai.Client(api_key=key, http_options=http)
        else:
            raise SystemExit(f"Unknown JUDGE_PROVIDER '{self.provider}', use groq or gemini")

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def short(self) -> str:
        return f"{self.provider}-{self.model.split('/')[-1]}"

    def __call__(self, system: str, user: str) -> str:
        for attempt, wait in enumerate([0] + RETRY_WAITS):
            if wait:
                print(f"    judge rate-limited, waiting {wait}s (attempt {attempt}/{len(RETRY_WAITS)})", flush=True)
                time.sleep(wait)
            try:
                return self._call(system, user)
            except Exception as e:
                if not _is_rate_limit(e) or attempt == len(RETRY_WAITS):
                    raise
                print(f"    {str(e)[:300]}", flush=True)
                if "per day" in str(e).lower() or "perday" in str(e).lower():
                    raise
        raise RuntimeError("unreachable")

    def _call(self, system: str, user: str) -> str:
        if self.provider == "groq":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content

        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        return resp.text or ""


def _is_rate_limit(e: Exception) -> bool:
    text = f"{type(e).__name__} {e}".lower()
    return any(s in text for s in ("429", "503", "resource_exhausted", "unavailable", "overloaded")) or (
        "rate" in text and "limit" in text
    )
