from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-5.6-terra"


def api_key() -> str:
    if value := os.environ.get("OPENROUTER_API_KEY"):
        return value
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENROUTER_API_KEY is required")


def request_json(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8000,
    reasoning_effort: str = "low",
    retries: int = 4,
) -> tuple[Any, dict]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": reasoning_effort, "exclude": True},
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openai/hillclimb-midtraining",
            "X-Title": "hillclimb natural multivalue benchmark",
        },
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                body = json.load(response)
            content = body["choices"][0]["message"]["content"]
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(content), body.get("usage", {})
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            KeyError,
            json.JSONDecodeError,
        ) as error:
            if attempt + 1 == retries:
                raise RuntimeError(f"OpenRouter request failed after {retries} attempts") from error
            time.sleep(2**attempt)
    raise AssertionError("unreachable")
