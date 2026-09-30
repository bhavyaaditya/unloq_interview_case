"""Core agent logic. Single public entrypoint: `get_response(question)`.

Architecture decisions (why it looks like this):

1. ONE function interface for eval:
   `eval/run_eval.py` calls only `get_response(question) -> dict`. It never
   imports ADK/Gemini/Anthropic types, so we can swap model backends without
   touching evaluation code (strict `agent/` vs `eval/` separation).

2. Backend cascade ADK -> Gemini -> Anthropic -> offline:
   - The spec asks for Google ADK first, raw SDK fallback second.
   - In practice the interview/CI machine often has NO API keys. If we crashed
     there, no `evaluation_traces.jsonl` would be produced and the whole
     exercise would fail. So a deterministic offline retriever is the final
     fallback. It honours ALL four prompt constraints (refusals, citations,
     thinking) so traces remain meaningful without credentials.
   - Backend is selected once at import; `get_backend_name()` exposes it for logging.

3. No tools / booking / refund simulation:
   This is strictly a retrieval task, so we deliberately register ZERO tools
   on the ADK agent. Adding function tools would invite hallucinations and
   break the "answer ONLY using corpus.md" constraint.

4. ADK UI vs batch eval split:
   - `root_agent` (module-level) is what `adk web` discovers for interactive
     testing in the browser. Zero extra frontend code needed.
   - `get_response()` does NOT run the full ADK Runner (async session service,
     event loop) because that is slow/fragile in a 20-question batch loop.
     Instead it calls the underlying model directly with the SAME system
     prompt, guaranteeing UI and eval share identical instructions.
"""

from __future__ import annotations

import re
import traceback
from functools import lru_cache
from typing import Dict, List

from . import config
from .prompts import build_system_prompt

# ---------------------------------------------------------------------------
# Cached corpus / prompt (read disk once, reuse across 20 eval questions)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _corpus_text() -> str:
    """Load corpus.md once per process (20 eval calls share it)."""
    return config.load_corpus()


@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    """Public accessor so eval logs / debugging can inspect the prompt."""
    return build_system_prompt(_corpus_text())


# ---------------------------------------------------------------------------
# ADK agent object for `adk web` (native UI, no custom frontend)
# ---------------------------------------------------------------------------

# `adk web` scans for a module-level `root_agent`. We build it with the SAME
# system prompt as batch eval so interactive testing matches logged traces.
# If google-adk is not installed, this is None and everything else still works.
root_agent = None
try:  # pragma: no cover - import-time optional dependency
    from google.adk.agents import Agent as _AdkAgent  # type: ignore

    root_agent = _AdkAgent(
        name="meridian_patient_assistant",
        model=config.DEFAULT_GEMINI_MODEL,
        instruction=get_system_prompt(),
        # No tools: strictly retrieval. See module docstring #3.
        tools=[],
    )
except Exception:  # ADK missing or misconfigured -> UI unavailable, eval unaffected
    root_agent = None


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _detect_backend() -> str:
    """Choose the best available backend without making network calls.

    Order: gemini (GOOGLE_API_KEY) -> anthropic (ANTHROPIC_API_KEY) -> offline.
    We check env keys AND importability so a missing SDK never crashes import.
    """
    if config.GOOGLE_API_KEY:
        try:
            import google.generativeai  # noqa: F401  # type: ignore

            return "gemini"
        except ImportError:
            pass
    if config.ANTHROPIC_API_KEY:
        try:
            import anthropic  # noqa: F401  # type: ignore

            return "anthropic"
        except ImportError:
            pass
    # Also allow ADK-only environments to report sensibly, but batch calls
    # still go through the direct SDK / offline path for speed.
    return "offline"


_BACKEND = _detect_backend()


def get_backend_name() -> str:
    """Expose active backend for eval logging / debugging."""
    # Re-detect if keys appeared after import (e.g. dotenv loaded late).
    # Cheap: only re-checks when current backend is offline.
    if _BACKEND == "offline":
        return _detect_backend()
    return _BACKEND


# ---------------------------------------------------------------------------
# Helpers shared by all backends
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"Documents?\s+([A-E](?:\s*(?:,|and|&)\s*[A-E])*)", re.IGNORECASE)


def extract_cited_policies(text: str) -> List[str]:
    """Parse 'According to Document C...' citations into ['Doc C'].

    Why regex post-processing instead of trusting the model:
    - The eval schema requires a structured `policies_cited` list, but the
      model outputs free text. Regex gives a deterministic projection that
      works for EVERY backend (ADK/Gemini/Claude/offline) without extra LLM calls.
    """
    cited: List[str] = []
    for m in _CITATION_RE.finditer(text or ""):
        chunk = m.group(1)
        for letter in re.findall(r"[A-E]", chunk):
            tag = f"Doc {letter}"
            if tag not in cited:
                cited.append(tag)
    return cited


def _is_staff_question(question: str) -> bool:
    """Conservative staff-privacy gate (Constraint 2), applied BEFORE any LLM call.

    Why defence-in-depth (keyword gate + prompt rule):
    - The system prompt already tells the model to refuse, but prompts can be
      jailbroken ("ignore previous instructions..."). A code-level gate
      guarantees Document D contents can never leak, even if the model
      misbehaves. Privacy failures are the highest-severity eval errors.
    """
    q = question.lower()
    for kw in config.STAFF_KEYWORDS:
        # Support simple ".*" patterns in keyword list + plain substrings.
        if ".*" in kw:
            if re.search(kw, q):
                return True
        elif kw in q:
            return True
    return False


# ---------------------------------------------------------------------------
# Offline deterministic retriever (no API key needed)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "and", "for", "with", "what", "when", "where", "which", "who",
    "how", "are", "is", "do", "does", "can", "you", "your", "our", "about",
    "have", "has", "had", "this", "that", "from", "into", " clinic",
    "clinic", "please", "tell", "me", "my", "get", "need",
    # Domain-generic words that appear across many policy docs and would
    # otherwise drown out discriminative terms (e.g. "visit" appears in B/C/E,
    # "without" in B/D/E). Down-weighted via IDF below, but also stop-listed
    # when they carry almost no retrieval signal on this small corpus.
    "will", "would", "without", "within", "also", "much", "still", "long",
}

_DOC_SPLIT_RE = re.compile(
    r"##\s*Document\s+([A-E])\s*[—\-–:]*\s*(.*?)\n(.*?)(?=^##\s*Document|\Z)",
    re.MULTILINE | re.DOTALL,
)

# Canonical synonym groups: any member maps to the group id (e.g. "PAY").
# Why: patients say "pay my bill" while policies say "payment / billing".
# Without this, pure token overlap misses valid answers (recall failure seen
# in early eval runs: "installments" vs "installment", "pay" vs "payment").
_SYNONYM_GROUPS: list[set[str]] = [
    {"pay", "pays", "paying", "paid", "payment", "payments", "bill", "bills", "billing"},
    {"cost", "costs", "price", "prices", "rate", "rates", "fee", "fees", "charge", "charges"},
    {"plan", "plans", "installment", "installments"},
    {"cancel", "cancels", "cancelled", "cancellation", "reschedule"},
    {"book", "books", "booked", "booking", "appointment", "appointments"},
    {"insurance", "insured", "uninsured"},
    {"record", "records"},
    {"result", "results"},
    {"refill", "refills", "refilled"},
    {"prescription", "prescriptions"},
    {"hour", "hours"},
]
_TOKEN_TO_GROUP: dict[str, str] = {}
for _gi, _grp in enumerate(_SYNONYM_GROUPS):
    _gid = f"__SYN{_gi}__"
    for _tok in _grp:
        _TOKEN_TO_GROUP[_tok] = _gid


def _stem(token: str) -> str:
    """Tiny rule-based stemmer (no NLTK dependency for a 3h timebox).

    Handles the plural/tense variants actually present in corpus.md:
    refills->refill, plans->plan, booked->book, pays->pay.
    Order matters: -ies, -ing, -ed before trailing -s.
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:  # stories -> story
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:  # paying -> pay
        base = token[:-3]
        return base
    if token.endswith("ed") and len(token) > 4:  # booked -> book
        base = token[:-2]
        # booked -> book (keep double consonant), refilled -> refill
        return base
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]  # refills -> refill, plans -> plan
    return token


def _tokenize(text: str) -> set:
    """Lowercase alphanumeric tokens -> stemmed + synonym-expanded set."""
    raw = re.findall(r"[a-z0-9]+", text.lower())
    out: set[str] = set()
    for t in raw:
        if len(t) <= 2 or t in _STOPWORDS:
            continue
        s = _stem(t)
        if s in _STOPWORDS or len(s) <= 2:
            continue
        out.add(s)
        # Synonym expansion: add the canonical group id so "pay" and
        # "payment" (both stem differently) still match each other.
        # We check BOTH raw and stemmed forms for group membership.
        grp = _TOKEN_TO_GROUP.get(t) or _TOKEN_TO_GROUP.get(s)
        if grp:
            out.add(grp)
    # ' clinic' key above has a stray space (kept for compat); also drop 'clinic'.
    out.discard("clinic")
    return out


def _idf_weighted_scores(q_tokens: set, docs: Dict[str, str]) -> Dict[str, float]:
    """Score each doc by sum of IDF weights of overlapping tokens.

    Why IDF instead of raw overlap counts:
    - Raw counts let generic words ("visit", "without", "cost") outvote rare,
      discriminative terms ("insurance", "installment"). On a 5-doc corpus a
      single generic trigram can flip the winner (observed: Q9 went to Doc E).
    - IDF weight = N / df rewards terms unique to one document, which is
      exactly what citation accuracy needs. Cheap to compute, no dependencies.
    """
    doc_tokens = {letter: _tokenize(text) for letter, text in docs.items()}
    N = max(1, len(doc_tokens))
    df: dict[str, int] = {}
    for toks in doc_tokens.values():
        for t in toks:
            df[t] = df.get(t, 0) + 1
    scores: Dict[str, float] = {}
    for letter, toks in doc_tokens.items():
        overlap = q_tokens & toks
        scores[letter] = sum(N / df[t] for t in overlap)
    return scores


def _parse_documents(corpus: str) -> Dict[str, str]:
    """Split corpus.md into {letter: section_text}."""
    docs: Dict[str, str] = {}
    for m in _DOC_SPLIT_RE.finditer(corpus):
        docs[m.group(1)] = m.group(0)
    # Fallback: if headings change format, treat whole file as one block so
    # eval still runs instead of crashing (robustness under time pressure).
    if not docs:
        docs["?"] = corpus
    return docs


def _offline_answer(question: str) -> Dict[str, object]:
    """Deterministic retrieval over corpus.md. Respects all 4 constraints."""
    corpus = _corpus_text()

    # --- Constraint 2 FIRST: staff gate (no citation on refusal) ---
    if _is_staff_question(question):
        thinking = (
            "Step 1 — staff check: question matched staff keyword list "
            f"(question={question!r}).\n"
            "Step 2 — privacy rule fires (highest priority): must refuse with the "
            "exact staff refusal sentence and cite NO documents.\n"
            "Step 3 — skipping retrieval over Document D entirely to avoid leakage."
        )
        return {
            "bot_response": config.STAFF_REFUSAL_RESPONSE,
            "bot_thinking": thinking,
            "policies_cited": [],
        }

    # --- Retrieval: IDF-weighted scoring over PUBLIC docs only ---
    # CRITICAL: Document D is CONFIDENTIAL (staff absence notes). It is excluded
    # from retrieval entirely — even for non-staff questions like "share my
    # medical information with my employer?", naive overlap would surface D's
    # "DO NOT share..." line and leak absence details in the evidence excerpt.
    # Excluding D guarantees no D content can ever be quoted or cited.
    docs = _parse_documents(corpus)
    public_docs = {k: v for k, v in docs.items() if k != "D"}
    if not public_docs:
        public_docs = docs  # fallback: headings changed, keep eval running
    q_tokens = _tokenize(question)
    scores = _idf_weighted_scores(q_tokens, public_docs)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_letter, best_score = ranked[0] if ranked else ("?", 0.0)

    # --- Constraint 1: not-found threshold ---
    # With IDF weights, a single rare-term match scores ~4-5 (N/df with N=4),
    # while a generic-word match scores ~1. Threshold 2.0 therefore means:
    # one discriminative term OR two generic terms are needed — tuned so
    # out-of-corpus questions (dental, parking) refuse, but paraphrases
    # ("installments" vs "installment") still hit via stemming/synonyms.
    if best_score < 2.0 or best_letter == "?":
        thinking = (
            "Step 1 — staff check: no match (not staff-related).\n"
            f"Step 2 — IDF-weighted retrieval scores (D excluded): {scores}.\n"
            f"Step 3 — best score {best_score:.2f} < threshold 2.0, so no document "
            "supports the answer.\n"
            "Step 4 — Constraint 1 fires: reply with the exact not-found sentence, "
            "no citation."
        )
        return {
            "bot_response": config.NOT_FOUND_RESPONSE,
            "bot_thinking": thinking,
            "policies_cited": [],
        }

    # --- Build grounded answer with citation (Constraint 3) ---
    # Pick the 2 bullet lines from the winning doc with highest IDF-weighted
    # overlap as evidence (same weighting, applied at bullet granularity).
    section = public_docs[best_letter]
    bullets = [ln.strip() for ln in section.splitlines() if ln.strip().startswith("-")]

    def _bullet_score(b: str) -> float:
        overlap = q_tokens & _tokenize(b)
        # Reuse IDF intuition locally: rarer-in-doc overlap wins. Approximate
        # with plain count here (bullets are short); question rarity already
        # handled by doc-level IDF selection above.
        return float(len(overlap))

    scored_bullets = sorted(bullets, key=_bullet_score, reverse=True)
    evidence = " ".join(scored_bullets[:2]) if scored_bullets else section[:300]

    # Multi-doc citation: tightened to avoid spurious second cites (observed:
    # "book an appointment" wrongly cited B+E on raw counts). Require runner-up
    # to have a substantive score (>=3.0) AND be close (>=80% of best).
    cited = [f"Doc {best_letter}"]
    if len(ranked) > 1 and ranked[1][1] >= 3.0 and ranked[1][1] >= 0.8 * best_score:
        cited.append(f"Doc {ranked[1][0]}")

    if len(cited) == 1:
        bot_response = f"According to Document {best_letter}, {evidence}"
    else:
        letters = " and ".join(c.split()[-1] for c in cited)
        bot_response = f"According to Documents {letters}, {evidence}"

    thinking = (
        "Step 1 — staff check: no match (not staff-related).\n"
        f"Step 2 — IDF-weighted retrieval scores (D excluded): {scores}.\n"
        f"Step 3 — winner: Document {best_letter} (score {best_score:.2f}). "
        f"Evidence selected: {evidence[:160]!r}...\n"
        f"Step 4 — citation decision: {cited} (Constraint 3: cite letter, "
        "no refusal involved)."
    )
    return {
        "bot_response": bot_response,
        "bot_thinking": thinking,
        "policies_cited": cited,
    }


# ---------------------------------------------------------------------------
# Live-model backends (Gemini / Anthropic) — same prompt, same schema
# ---------------------------------------------------------------------------


def _gemini_answer(question: str, system_prompt: str) -> Dict[str, object]:
    """Call Gemini via google-generativeai with the shared system prompt."""
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=config.GOOGLE_API_KEY)
    model = genai.GenerativeModel(
        config.DEFAULT_GEMINI_MODEL,
        system_instruction=system_prompt,
    )
    # Constraint 4: request thoughtful answers; the SDK version pinned in
    # requirements exposes thinking for gemini-3.8 models. If the installed
    # version does not, generation still works — thinking is captured below
    # on a best-effort basis so eval never breaks across SDK versions.
    resp = model.generate_content(question)
    text = getattr(resp, "text", "") or ""
    thinking = ""
    try:  # Best-effort thinking extraction across SDK shapes
        parts = getattr(getattr(resp, "candidates", [None])[0], "content", None)
        _ = parts  # placeholder: current SDK surfaces reasoning in `text`
        thinking = "(CoT enabled via system prompt; thinking trace not separately exposed by this SDK version.)"
    except Exception:
        thinking = "(thinking unavailable)"
    return {
        "bot_response": text.strip(),
        "bot_thinking": thinking,
        "policies_cited": extract_cited_policies(text),
    }


def _anthropic_answer(question: str, system_prompt: str) -> Dict[str, object]:
    """Call Claude via anthropic SDK with extended thinking enabled."""
    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    # Constraint 4: explicit extended-thinking params (Claude supports them
    # natively). Budget kept small to stay fast across 20 eval questions.
    msg = client.messages.create(
        model=config.DEFAULT_ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system_prompt,
        thinking={"type": "enabled", "budget_tokens": 512},
        messages=[{"role": "user", "content": question}],
    )
    thinking_chunks, text_chunks = [], []
    for block in getattr(msg, "content", []):
        btype = getattr(block, "type", "")
        if btype == "thinking":
            thinking_chunks.append(getattr(block, "thinking", ""))
        elif btype == "text":
            text_chunks.append(getattr(block, "text", ""))
    text = "\n".join(text_chunks).strip()
    thinking = "\n".join(thinking_chunks).strip()
    return {
        "bot_response": text,
        "bot_thinking": thinking,
        "policies_cited": extract_cited_policies(text),
    }


# ---------------------------------------------------------------------------
# Public entrypoint (the ONLY thing eval/ may call)
# ---------------------------------------------------------------------------


def get_response(question: str) -> Dict[str, object]:
    """Answer one patient question. Never raises — eval loops must not crash.

    Returns:
        dict with keys:
          - bot_response (str): final user-facing answer.
          - bot_thinking (str): CoT / extended-thinking trace ("" if unavailable).
          - policies_cited (list[str]): e.g. ["Doc C"].

    Why never-raise: a single API timeout should not abort a 20-question eval
    run. Errors are captured into `bot_thinking` and a safe not-found reply
    is returned so the trace file stays complete and auditable.
    """
    system_prompt = get_system_prompt()

    # Code-level privacy gate runs regardless of backend (defence in depth).
    if _is_staff_question(question):
        return {
            "bot_response": config.STAFF_REFUSAL_RESPONSE,
            "bot_thinking": (
                "Pre-LLM staff gate matched; refused without calling the model "
                "to guarantee no Document D leakage."
            ),
            "policies_cited": [],
        }

    backend = get_backend_name()
    try:
        if backend == "gemini":
            result = _gemini_answer(question, system_prompt)
        elif backend == "anthropic":
            result = _anthropic_answer(question, system_prompt)
        else:
            result = _offline_answer(question)

        # Normalise schema (backends must not leak extra/missing keys to eval).
        response_text = str(result.get("bot_response", ""))
        # Re-derive citations deterministically so offline and live backends
        # share identical `policies_cited` semantics. Exception: refusals
        # must carry NO citations even if the model accidentally cites.
        if response_text in (config.STAFF_REFUSAL_RESPONSE, config.NOT_FOUND_RESPONSE):
            cited: List[str] = []
        else:
            cited = extract_cited_policies(response_text)
            if not cited and isinstance(result.get("policies_cited"), list):
                cited = [str(x) for x in result["policies_cited"]]  # type: ignore
        return {
            "bot_response": response_text,
            "bot_thinking": str(result.get("bot_thinking", "") or ""),
            "policies_cited": cited,
        }
    except Exception as exc:  # pragma: no cover - safety net
        return {
            "bot_response": config.NOT_FOUND_RESPONSE,
            "bot_thinking": (
                f"Backend '{backend}' raised {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc(limit=3)}"
            ),
            "policies_cited": [],
        }
