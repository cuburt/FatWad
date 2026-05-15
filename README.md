# FatWad

Manual wealth simulator with an LLM agent. Brutalist UI. FastAPI + LangGraph + Postgres.

```
src/                     FastAPI backend, LangGraph agent, SQLModel DB
frontend/app.py          Streamlit, talks to backend over httpx
evals/                   Deterministic eval harness
scripts/demo.sh          One curl per agent path
docs/                    Design + prompts
```

## Quickstart

```bash
cp .env.example .env                           # set OPENROUTER_API_KEY (server-side)
docker compose up -d                           # postgres + api + frontend
open http://localhost:8501                     # UI
open http://localhost:8000/docs                # OpenAPI
./scripts/demo.sh                              # one curl per agent path
```

Or run pieces locally:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
docker compose up -d db                        # just postgres
uvicorn src.main:app --reload                  # backend on :8000
API_URL=http://localhost:8000 streamlit run frontend/app.py
```

## Two graphs, one state

- `POST /ask` — safety → snapshot → summarize. Read-only Q&A.
- `POST /agent` — safety → snapshot → planner → tool → summarize. Tools + web.

Planner picks one of `{forecast, scenario, advice, lookup, summarize}`. The
tool node dispatches to deterministic Python (advice), a litellm tool-call
loop (forecast/scenario), or a web-grounded JSON prefill (lookup). The
summarizer streams the final answer with `:online` so it can cite sources.

See [docs/design.md](docs/design.md) for the why.

## Privacy

The OpenRouter API key is loaded from `.env` on the server. The frontend
never sees it and never sends it. `/ai/*` endpoints return 503 if it's
unset. Postgres in `docker-compose.yml` binds to `127.0.0.1` only.
Ghost Mode blurs every dollar amount on screen.
