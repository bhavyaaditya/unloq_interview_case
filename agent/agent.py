"""Core agent logic. Single public entrypoint: `get_response(question)`.

Pure-LLM design (no hardcoded policy logic):
- The ENTIRE corpus.md is injected into the system prompt; the model alone
  decides what is answerable, what is private, and what to cite. There are no
  keyword gates, no token scorers, no citation regexes — nothing to update
  when policies change.
- `eval/run_eval.py` calls only `get_response(question) -> dict`.
- Gemini-only. Citations come from native structured output
  (`response_schema`), thinking from native `ThinkingConfig` thought parts.
- `root_agent` (module-level ADK Agent, zero tools) is what `adk web`
  discovers for interactive manual testing with the same prompt.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from functools import lru_cache
from typing import Dict, List

from . import config
from .prompts import build_system_prompt


@lru_cache(maxsize=1)
def _corpus_text() -> str:
    """Load corpus.md once per process (20 eval calls share it)."""
    return config.load_corpus()


@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    """Full system instruction with the live corpus injected."""
    return build_system_prompt(_corpus_text())


# `adk web` scans for a module-level `root_agent`. Same prompt as eval.
root_agent = None
try:
    from google.adk.agents import Agent as _AdkAgent  # type: ignore

    root_agent = _AdkAgent(
        name="meridian_patient_assistant",
        model=config.DEFAULT_GEMINI_MODEL,
        instruction=get_system_prompt(),
        tools=[],  # strictly retrieval: no booking/refund tools
    )
except Exception:
    root_agent = None


def get_backend_name() -> str:
    """Kept for eval logging compatibility. Only backend is Gemini."""
    return "gemini"


# Native structured-output schema: the model returns citations itself, so
# eval gets a real list without any regex post-processing in code.
_RESPONSE_SCHEMA: Dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "bot_response": {
            "type": "STRING",
            "description": (
                "Final user-facing answer. Must follow the system prompt rules: "
                "cite the document letter for policy answers, or reproduce the "
                "exact refusal sentence when refusing."
            ),
        },
        "policies_cited": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": (
                'Document letters cited in the answer, each strictly in the '
                'format "Doc X" (e.g. ["Doc C"], ["Doc A", "Doc G"]). '
                "Empty list when refusing."
            ),
        },
    },
    "required": ["bot_response", "policies_cited"],
    "propertyOrdering": ["bot_response", "policies_cited"],
}


def _retry_delay(exc: Exception, attempt: int) -> float | None:
    """Seconds to wait before retrying, or None if the error is not retriable.

    Only transport throttling (429 quota, 503 overload) is retried. Anything
    else (auth, bad request, bad model) fails fast. Prefers the server's own
    "retry in Ns" hint, else exponential backoff.
    """
    msg = str(exc)
    if "429" not in msg and "503" not in msg and "UNAVAILABLE" not in msg:
        return None
    m = re.search(r"retry in ([\d.]+)s", msg)
    if m:
        return float(m.group(1)) + 1.0
    return min(60.0, 5.0 * (2**attempt))


def _generate(question: str) -> Dict[str, object]:
    """One live Gemini call. Returns raw model fields (no policy logic)."""
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    client = genai.Client()  # reads GOOGLE_API_KEY / GEMINI_API_KEY from env
    gemini_config = types.GenerateContentConfig(
        system_instruction=get_system_prompt(),
        response_mime_type="application/json",
        response_schema=_RESPONSE_SCHEMA,
        # Native extended thinking (Constraint 4): thought parts are returned
        # alongside the answer and logged as bot_thinking below.
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=config.THINKING_BUDGET,
        ),
    )
    resp = None
    for attempt in range(6):
        try:
            resp = client.models.generate_content(
                model=config.DEFAULT_GEMINI_MODEL,
                contents=question,
                config=gemini_config,
            )
            break
        except Exception as exc:
            delay = _retry_delay(exc, attempt)
            if delay is None or attempt == 5:
                raise
            time.sleep(delay)

    thinking_chunks: List[str] = []
    answer_text = ""
    candidate = (resp.candidates or [None])[0]
    parts = getattr(getattr(candidate, "content", None), "parts", None) or []
    for part in parts:
        text = getattr(part, "text", "") or ""
        if getattr(part, "thought", False):
            if text:
                thinking_chunks.append(text)
        elif text and not answer_text:
            answer_text = text
    if not answer_text:
        answer_text = getattr(resp, "text", "") or ""

    payload: Dict[str, object] = {"bot_response": "", "policies_cited": []}
    try:
        parsed = json.loads(answer_text)
        if isinstance(parsed, dict):
            payload["bot_response"] = str(parsed.get("bot_response", ""))
            cited = parsed.get("policies_cited", [])
            payload["policies_cited"] = [str(c) for c in cited] if isinstance(cited, list) else []
    except (json.JSONDecodeError, AttributeError):
        # Schema enforcement normally prevents this; fall back to raw text
        # rather than crashing the eval loop.
        payload["bot_response"] = answer_text

    payload["bot_thinking"] = "\n".join(thinking_chunks)
    return payload


def get_response(question: str) -> Dict[str, object]:
    """Answer one patient question via live Gemini.

    Returns dict with bot_response (str), bot_thinking (str, native thought
    parts, "" if the model returned none), policies_cited (list[str] from the
    model's own structured output). No keyword/regex/scoring logic anywhere.
    """
    try:
        result = _generate(question)
        return {
            "bot_response": str(result.get("bot_response", "")),
            "bot_thinking": str(result.get("bot_thinking", "") or ""),
            "policies_cited": list(result.get("policies_cited", []) or []),
        }
    except Exception as exc:
        return {
            "bot_response": config.NOT_FOUND_RESPONSE,
            "bot_thinking": (
                f"Gemini call failed with {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc(limit=3)}"
            ),
            "policies_cited": [],
        }
