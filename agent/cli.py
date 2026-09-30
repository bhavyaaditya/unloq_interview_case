"""Simple CLI for quick manual testing (no custom frontend per spec).

Why a CLI exists alongside ADK UI:
- `adk web` is the primary interactive path, but it needs google-adk + creds.
- `python -m agent` works with ZERO dependencies/keys (offline fallback), so
  reviewers can smoke-test retrieval behaviour in seconds.

Usage:
    python -m agent "What are your opening hours?"
    python -m agent   # interactive REPL
"""

from __future__ import annotations

import sys

from .agent import get_backend_name, get_response


def _print_result(question: str) -> None:
    result = get_response(question)
    print(f"\n[backend: {get_backend_name()}]")
    print(f"Q: {question}")
    print(f"A: {result['bot_response']}")
    print(f"Cited: {result['policies_cited']}")
    if result.get("bot_thinking"):
        print(f"Thinking:\n{result['bot_thinking']}")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv:  # single-shot: python -m agent "question..."
        _print_result(" ".join(argv))
        return 0
    # REPL mode
    print(f"Meridian Clinic assistant (backend={get_backend_name()}). Type 'quit' to exit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q.lower() in {"quit", "exit", "q"}:
            return 0
        if not q:
            continue
        _print_result(q)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
