# Prompts

All system prompts live in [src/agents/prompts.py](../src/agents/prompts.py).

## Planner

Picks one of `forecast`, `scenario`, `advice`, `lookup`, `summarize`. Returns
exactly one word. Bad outputs fall back to `summarize` so we never call a tool
the user didn't intend.

## Summarizer

Brutalist style: terse, numbers-first, no fluff. Cites web sources inline as
`[1]`, `[2]` and lists URLs at the end. Conditioned on the user's wealth
snapshot + the tool results from this turn.

## Tool prompt

Drives the litellm function-call loop on the AGENTIC tier. The tool list is
deterministic — `compute_compound`, `compute_freedom_date`, `simulate_scenario`,
`compute_required_monthly`, `compute_required_return`. The model can stop
calling tools whenever it has enough data.
