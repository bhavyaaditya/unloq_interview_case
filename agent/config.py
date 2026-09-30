"""Central configuration: paths, model names, env handling.

Why a separate config module:
- Both `agent/` and the ADK UI need the same corpus path / model name.
- Keeping it in one place avoids hardcoding `corpus.md` paths in multiple files
  (a common source of "works on my machine" eval failures under time pressure).
"""

import os
from pathlib import Path

# --- Paths ---
# ROOT = repo root (parent of `agent/`). corpus.md lives there per the spec.
# Using Path resolution instead of relative "./corpus.md" so imports work
# regardless of cwd (e.g. `python eval/run_eval.py` vs `pytest` vs `adk web`).
ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT_DIR / "corpus.md"

# --- Model selection ---
# Priority: explicit env override > sensible defaults.
# Gemini 2.5 Flash is cheap/fast and supports thinking; Claude Sonnet supports
# extended thinking. Either satisfies Constraint 4. We default to Gemini because
# ADK is Gemini-native.
DEFAULT_GEMINI_MODEL = os.getenv("MERIDIAN_MODEL", "gemini-3.5-flash")
DEFAULT_ANTHROPIC_MODEL = os.getenv("MERIDIAN_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# API keys: read from environment (and .env if python-dotenv is installed).
# We deliberately do NOT crash when keys are missing — the agent falls back to
# a deterministic offline retrieval mode so `eval/run_eval.py` always produces
# a trace file, even in CI / interview rooms without credentials.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Exact refusal strings required by the spec. Centralised here so agent logic
# and tests can never drift out of sync with the evaluation rubric.
NOT_FOUND_RESPONSE = "I don't have that information in our current policies."
STAFF_REFUSAL_RESPONSE = (
    "I cannot disclose information about our staff. Please contact the clinic directly."
)

# Keywords that signal a staff-privacy question (Constraint 2).
# Checked BEFORE any retrieval/citation so we never leak Document D contents.
# Kept intentionally broad (recall > precision) because leaking is worse than
# over-refusing on this task.
STAFF_KEYWORDS = [
    "dr. morgan", "dr morgan", "alice morgan",
    "dr. carter", "dr carter", "ben carter",
    "fiona reid", "nurse reid", "fiona",
    "david okafor", "receptionist",
    "staff", "doctor", "nurse", "absence", "absent",
    "leave", "sick", "surgery recovery", "bereavement",
    "schedule", "shift", "on-call", "on call", "roster",
    "where is", "why is", "when will.*back",
]


def load_corpus() -> str:
    """Read the full corpus.md text.

    Why inject the ENTIRE file (vs RAG/chunking):
    - The corpus is tiny (single file, a few KB). Full injection is simpler,
      fully deterministic, and eliminates retriever tuning within a 3h timebox.
    - It also guarantees the citation constraint is testable: the model always
      sees every Document letter.
    """
    # Try to load .env silently if python-dotenv is available (local dev convenience).
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(ROOT_DIR / ".env")
        # Re-read after dotenv load (env may have been empty at import time).
        global GOOGLE_API_KEY, ANTHROPIC_API_KEY
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    except ImportError:
        pass  # dotenv is optional; env vars still work without it.

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {CORPUS_PATH}. "
            "The agent requires corpus.md in the repo root."
        )
    return CORPUS_PATH.read_text(encoding="utf-8")
