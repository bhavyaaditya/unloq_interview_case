"""Robust JSONL logger for evaluation traces.

Why JSONL (not CSV / single JSON array):
- LLM outputs contain commas, quotes, and newlines. CSV parsing breaks on
  these unless every field is perfectly escaped; JSONL stores one valid JSON
  object per line, so a weird model output can never corrupt neighbouring rows.
- Append-per-question means a crash on Q17 still leaves Q1–Q16 on disk for
  partial analysis — a single JSON array would be lost entirely.
- Each line is independently parseable (`json.loads(line)`), which is ideal
  for strict, possibly automated, scoring harnesses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class EvalLogger:
    """Overwrite-clean file writer: one JSON object per line, UTF-8."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # Ensure parent dir exists so `python eval/run_eval.py` works from any cwd.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Open in WRITE mode once at construction -> clean overwrite each run.
        # (Spec: "append-only or overwritten cleanly on each run". We choose
        # overwrite so re-runs never duplicate rows and confuse scoring.)
        # The file handle stays open in append mode afterwards via `log()`.
        self.path.write_text("", encoding="utf-8")
        self._count = 0

    def log(self, record: Dict[str, Any]) -> None:
        """Append one trace record. Never raises (eval loop must survive)."""
        # Enforce the required schema with safe defaults so downstream scoring
        # never KeyErrors on a missing field.
        safe: Dict[str, Any] = {
            "question": str(record.get("question", "")),
            "fiona_expected": str(record.get("fiona_expected", "")),
            "bot_response": str(record.get("bot_response", "")),
            "bot_thinking": str(record.get("bot_thinking", "") or ""),
            "policies_cited": list(record.get("policies_cited", []) or []),
            "eval_status": str(record.get("eval_status", "PENDING")),
        }
        # Preserve any extra debug keys (e.g. backend, latency) without breaking schema.
        for k, v in record.items():
            if k not in safe:
                safe[k] = v
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(safe, ensure_ascii=False) + "\n")
            self._count += 1
        except Exception as exc:  # Last resort: surface to stderr, don't crash eval
            print(f"[EvalLogger] FAILED to write record: {exc}")

    @property
    def count(self) -> int:
        return self._count
