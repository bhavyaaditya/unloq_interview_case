"""Central configuration: paths, model names, env handling.

Only declarative settings live here. There is deliberately NO per-policy
logic (no keyword lists, no scoring knobs): the LLM decides what is
answerable, private, or citable purely from the prompt + live corpus.md.
"""

import os
from pathlib import Path

# --- Paths ---
# ROOT = repo root (parent of `agent/`). corpus.md lives there per the spec.
ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT_DIR / "corpus.md"

# --- Model selection (Gemini only) ---
DEFAULT_GEMINI_MODEL = os.getenv("MERIDIAN_MODEL", "gemini-3.6-flash")

# Thinking budget for native Gemini thinking traces (Constraint 4).
# Small on purpose: 20 eval questions stay fast; raise for harder reasoning.
THINKING_BUDGET = int(os.getenv("MERIDIAN_THINKING_BUDGET", "512"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Exact refusal strings required by the spec. Referenced by the system prompt
# so the model reproduces them verbatim; never matched in code.
NOT_FOUND_RESPONSE = "I don't have that information in our current policies."
STAFF_REFUSAL_RESPONSE = (
    "I cannot disclose information about our staff. Please contact the clinic directly."
)


def load_corpus() -> str:
    """Read the full corpus.md text (the model's only source of truth)."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(ROOT_DIR / ".env")
        global GOOGLE_API_KEY
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
    except ImportError:
        pass  # dotenv is optional; env vars still work without it.

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {CORPUS_PATH}. "
            "The agent requires corpus.md in the repo root."
        )
    return CORPUS_PATH.read_text(encoding="utf-8")
