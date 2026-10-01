"""Trace writer for evaluation runs.

Output is a single JSON array (`evaluation_traces.json`), so the file loads
with a plain `json.load()` and opens directly via `pd.read_json()` or Excel
(Data > From File > JSON) — no notebook fix-up code needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _sanitize(record: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce the trace schema with safe defaults (no KeyError downstream)."""
    safe: Dict[str, Any] = {
        "question": str(record.get("question", "")),
        "fiona_expected": str(record.get("fiona_expected", "")),
        "bot_response": str(record.get("bot_response", "")),
        "bot_thinking": str(record.get("bot_thinking", "") or ""),
        "policies_cited": list(record.get("policies_cited", []) or []),
        "eval_status": str(record.get("eval_status", "PENDING")),
    }
    for k, v in record.items():
        if k not in safe:
            safe[k] = v
    return safe


def save_traces(path: str | Path, records: List[Dict[str, Any]]) -> Path:
    """Write all trace records as one JSON array, overwriting any prior run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_sanitize(r) for r in records]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
