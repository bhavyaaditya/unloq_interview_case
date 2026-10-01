# Meridian Clinic — Patient Q&A Agent (unloq_interview_case)

A minimal Gemini + Google ADK retrieval agent that answers patient questions
**only** from `corpus.md` (the clinic policy knowledge base), with native
citations and thinking traces, plus a strict 20-question evaluation harness.

## Prerequisites

- Python 3.10+ (tested on 3.14)
- The `.env` file in the repo root containing the API key (already provided —
  just keep it next to `corpus.md`):
  ```
  GOOGLE_API_KEY=AQ.Ab8RN6KrZZM4FuygykE82URtFzwNWmYt-2OWr8OvmN6Xx5pGsg
  ```
### A VERY IMPORTANT NOTE
***The Google AI Studio API key provided above is an actual key on a billing enabled account (with spend limits enabled) setup for this project by Bhavya Aditya and will be deleted on or before October-7. Kindly use it only for evaluating this project and avoid misuse. You are encouraged to use your own Google AI Studio API key to test this project.***

***Only for the purpose of evaluation of this case study, the `.env` file will be shared along with the entire codebase. Not a safe practice but doing so for this once for the evaluator's convenience.***

## Setup

```bash
# from the repo root
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No other configuration is needed. `corpus.md` in the repo root is the live
knowledge base — the agent reads it on every run.

## Run the agent (manual testing / conversation)

**Option A — quick CLI (no browser):**

```bash
python -m agent "What are the opening hours for the Birmingham site?"
python -m agent   # interactive REPL, type quit to exit
```

**Option B — ADK web UI (recommended for conversation):**

```bash
adk web --port 8003        # any free port works, e.g. --port 8080
```

Then open the printed URL (e.g. `http://localhost:8003`), select
`meridian_patient_assistant`, and chat. The UI is ADK-native — no custom
frontend code in this repo.

## Run the evaluation (20 client questions)

```bash
python eval/run_eval.py
# or: python -m eval.run_eval
```

This queries the agent on the client's fixed Q1–Q20 set and writes a single
JSON array to `eval/evaluation_traces.json` (overwritten on each run).
Each entry has `question`, `fiona_expected`, `bot_response`, `bot_thinking`,
`policies_cited`, `eval_status` (`PENDING` until you score it), plus
`backend`/`latency_ms`.

Open the results directly — no conversion needed:

```python
import pandas as pd
df = pd.read_json("eval/evaluation_traces.json")
df[["question", "fiona_expected", "bot_response", "bot_thinking"]]
```

## Project layout

```
corpus.md                  # knowledge base (single source of truth)
requirements.txt           # google-adk, google-genai, python-dotenv
agent/                     # importable agent package (no eval logic here)
  config.py                # paths, model name, key loading
  prompts.py               # system prompt with the 4 policy constraints
  agent.py                 # get_response(question) via live Gemini (ADK Agent for `adk web`)
  cli.py                   # `python -m agent` REPL
eval/                      # harness only — calls get_response(), knows nothing else
  run_eval.py              # the 20 questions + runner
  logger.py                # JSON-array trace writer
  evaluation_traces.json   # generated output (not committed)
```

## Design notes

- No hardcoded policy logic anywhere: no keyword gates, no scorers, no
  regexes. Privacy refusals, not-found replies, and citations are all the
  model's own decisions from the system prompt, so policy edits only ever
  touch `corpus.md`.
- Citations use Gemini native structured output; `bot_thinking` uses Gemini
  native thought traces — nothing hand-rolled.
