"""Evaluation runner: queries the agent on the client's 20 patient questions.

Modularity contract:
- This file knows NOTHING about how the agent is built. It imports exactly
  one symbol: `agent.get_response(question)`.
- The 20 questions + expected answers are the client's set, transcribed
  verbatim below. Nothing added, nothing reworded.
- Output is `eval/evaluation_traces.json`: a single JSON array, so it loads
  with plain `json.load()` and opens directly in pandas
  (`pd.read_json(path)`) or Excel — no fix-up code needed.
- `eval_status` defaults to "PENDING"; scoring happens separately. This
  runner never auto-grades.

Run:
    python eval/run_eval.py
    # or: python -m eval.run_eval  (from repo root)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# --- Import the agent as a black box ---
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent import get_backend_name, get_response  # noqa: E402

try:
    from .logger import save_traces  # noqa: E402  # `python -m eval.run_eval`
except ImportError:
    import importlib.util as _ilu  # noqa: E402  # `python eval/run_eval.py`

    _spec = _ilu.spec_from_file_location(
        "eval_logger", Path(__file__).resolve().parent / "logger.py"
    )
    assert _spec and _spec.loader
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)  # type: ignore
    save_traces = _mod.save_traces  # type: ignore

TRACE_PATH = Path(__file__).resolve().parent / "evaluation_traces.json"

# ---------------------------------------------------------------------------
# Client's evaluation set, transcribed verbatim (Q1–Q20).
# ---------------------------------------------------------------------------

EVAL_CASES: list[dict[str, str]] = [
    {
        "question": "How much notice do I have to give if I want to cancel?",
        "fiona_expected": "At least 24 hours.",
    },
    {
        "question": "Do you offer laser hair removal?",
        "fiona_expected": "Yes, at all three sites.",
    },
    {
        "question": "Can I have filler on the same day as my consultation?",
        "fiona_expected": "No, there is a 14 day cooling-off period first.",
    },
    {
        "question": "Are deposits refundable?",
        "fiona_expected": "Yes, within the 14 day cooling-off period.",
    },
    {
        "question": "How do I get a copy of my records?",
        "fiona_expected": "Request them at any time. Answered within 30 days, no charge.",
    },
    {
        "question": "Is there parking at the Birmingham clinic?",
        "fiona_expected": "Yes, there is free patient parking.",
    },
    {
        "question": "My clinician cancelled on me. Am I charged?",
        "fiona_expected": "No.",
    },
    {
        "question": "I paid a deposit for filler and changed my mind a week later. Do I get it back?",
        "fiona_expected": "No, deposits are non-refundable.",
    },
    {
        "question": "Can I pay cash for a GBP 700 treatment?",
        "fiona_expected": "No, cash above GBP 500 is not accepted.",
    },
    {
        "question": "How much is a filling?",
        "fiona_expected": "From GBP 140.",
    },
    {
        "question": "Is Dr Doyle unwell? My appointment with her was moved and nobody will tell me if she is coming back.",
        "fiona_expected": "Dr Doyle is on extended leave and we expect her back later in the year.",
    },
    {
        "question": "What happens if I just do not turn up?",
        "fiona_expected": "You are charged the full cost of the booked treatment.",
    },
    {
        "question": "How long until I get a response to a complaint?",
        "fiona_expected": "Acknowledged within 3 working days, full written response within 20 working days.",
    },
    {
        "question": "Can I get a sick note for work?",
        "fiona_expected": "Yes, for GBP 25.",
    },
    {
        "question": "How long does a refund take to reach me?",
        "fiona_expected": "Ten working days, to the original payment method.",
    },
    {
        "question": "Can my 17 year old have lip filler if I give consent?",
        "fiona_expected": "No, not to anyone under 18 in any circumstances.",
    },
    {
        "question": "I have not paid my bill for six weeks. What happens?",
        "fiona_expected": "After 30 days it is referred to collections and GBP 25 is added.",
    },
    {
        "question": "Is the Leeds clinic open on Friday evening?",
        "fiona_expected": "No, Leeds closes at 16:00 on Fridays.",
    },
    {
        "question": "Can I get a refund if I do not like how my treatment looks?",
        "fiona_expected": "No, that is handled under the complaints procedure.",
    },
    {
        "question": "How long do you keep my records after I stop coming?",
        "fiona_expected": "Eight years after your last appointment.",
    },
]


def run_eval(verbose: bool = True) -> Path:
    """Query the agent on all 20 questions, save one JSON array file."""
    backend = get_backend_name()
    if verbose:
        print(f"[eval] backend={backend} | questions={len(EVAL_CASES)} | out={TRACE_PATH}")

    records: list[dict] = []
    for i, case in enumerate(EVAL_CASES, start=1):
        t0 = time.time()
        result = get_response(case["question"])
        latency_ms = int((time.time() - t0) * 1000)
        records.append(
            {
                "question": case["question"],
                "fiona_expected": case["fiona_expected"],
                "bot_response": str(result.get("bot_response", "")),
                "bot_thinking": str(result.get("bot_thinking", "") or ""),
                "policies_cited": list(result.get("policies_cited", []) or []),
                "eval_status": "PENDING",
                "backend": backend,
                "latency_ms": latency_ms,
            }
        )
        if verbose:
            cited = ", ".join(records[-1]["policies_cited"]) or "—"
            print(f"[{i:02d}/20] cited=[{cited}] ({latency_ms}ms)")

    save_traces(TRACE_PATH, records)
    if verbose:
        print(f"[eval] done. Wrote {len(records)} traces to {TRACE_PATH}")
    return TRACE_PATH


if __name__ == "__main__":
    run_eval()
