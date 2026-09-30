"""Meridian Clinic patient assistant — package entrypoint.

Why this file exists:
- `eval/` must treat the agent as a black box and call only `get_response(question)`.
- This `__init__` re-exports the stable public interface so the eval runner can do
  `from agent import get_response` without knowing whether ADK, Gemini, Claude,
  or the offline fallback is active underneath (strict modularity).
"""

from .agent import get_response, get_backend_name, get_system_prompt  # noqa: F401

__all__ = ["get_response", "get_backend_name", "get_system_prompt"]
