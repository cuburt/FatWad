# Design

## What it is

FatWad is a manual wealth simulator with an LLM agent. The user enters every asset,
income stream, fixed bill, and weekly variable burn by hand; the app projects
forward, suggests a deployment plan, and exposes an agent that can answer
free-form forecasting questions with deterministic math + web-grounded prose.

## Layout

```
src/
  main.py                 FastAPI entry. Mounts route groups, runs init_db at boot.
  metrics.py              In-process counters; backs /metrics.
  safety.py               PII redaction (API keys, SSN, card-shaped digit runs).
  agents/
    state.py              AgentState TypedDict shared by both graphs.
    prompts.py            All system prompts in one file.
    graph.py              Wires up two LangGraph workflows (ask_app, agent_app).
    nodes.py              Node implementations (safety, snapshot, planner, tool, summarize).
  api/
    schemas.py            Pydantic request/response models.
    sessions_db.py        Chat session helpers (DB-backed).
    agent_routes.py       /ask and /agent.
    session_routes.py     /sessions, /sessions/{id}/messages.
    resource_routes.py    CRUD for assets, cashflow, scenarios, settings + /state, /report, /ai/*.
    metrics_routes.py     /metrics.
  db/
    models.py             SQLModel tables.
    session.py            Engine + init_db.
    seed.py               Seeds default user + settings row.
  reports/
    pdf.py                ReportLab Wealth Report.
  tools/
    forecast.py           Pure-math compute tools.
    advice.py             Allocation / rebalance / risk-profile constants.
    snapshot.py           Loads user state out of Postgres into the legacy dict shape.
    market.py             Web-grounded one-shots via litellm + OpenRouter :online.

frontend/app.py           Streamlit UI; talks to FastAPI over httpx. No DB access.
evals/run_evals.py        Deterministic eval harness for the math tools.
scripts/demo.sh           Five canned curls hitting each agent path.
```

## Two graphs, one state

`ask_app` (POST `/ask`) is `safety -> snapshot -> summarize`. No tools, no web.
Read-only Q&A about already-stored numbers.

`agent_app` (POST `/agent`) is `safety -> snapshot -> planner -> tool -> summarize`.
Planner picks one of `{forecast, scenario, advice, lookup, summarize}`. The
tool node dispatches:
- `forecast` / `scenario` → litellm tool-calling loop with the deterministic
  compute schema (compute_compound, compute_freedom_date, simulate_scenario,
  compute_required_monthly, compute_required_return).
- `advice` → pure Python (buy_list + rebalance_signals).
- `lookup` → market.prefill_assumptions (web-grounded JSON).

The summarizer streams the final answer using the SYNTHESIS-tier model with
the `:online` suffix so it can browse for citations.

## Why the split

Most providers don't combine tool calls and web grounding reliably in one
request. Doing the deterministic math first and the web-grounded prose second
keeps each step cheap and reliable.

## API key handling

OpenRouter API key is loaded from `.env` on the server. The frontend never
sees it and never sends it. `/ai/*` endpoints raise 503 if the key isn't set.
