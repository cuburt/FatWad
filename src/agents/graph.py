"""Wires up the two LangGraph workflows that share the same AgentState.

  ask_app   (POST /ask)    safety -> load_snapshot -> summarize     (read-only)
  agent_app (POST /agent)  safety -> load_snapshot -> planner -> tool -> summarize

LLM calls go through src.agents.llm, which walks a per-tier fallback chain
(OpenRouter -> Vercel AI Gateway) on rate limits or upstream failures.
"""

from langgraph.graph import END, StateGraph

from src.agents.state import AgentState

# Re-exports so existing call sites and tests keep importing tier names from
# `src.agents.graph`. Real implementation lives in `src.agents.llm`.
from src.agents.llm import AGENTIC, ROUTER, SYNTHESIS  # noqa: F401


def build_graphs() -> tuple:
    from src.agents.nodes import (
        planner_node,
        safety_node,
        snapshot_node,
        summarize_node,
        tool_node,
    )

    # /agent: full multi-tool flow
    agent_g = StateGraph(AgentState)
    agent_g.add_node("safety", safety_node)
    agent_g.add_node("load_snapshot", snapshot_node)
    agent_g.add_node("planner", planner_node)
    agent_g.add_node("tool", tool_node)
    agent_g.add_node("summarize", summarize_node)
    agent_g.set_entry_point("safety")
    agent_g.add_edge("safety", "load_snapshot")
    agent_g.add_edge("load_snapshot", "planner")
    agent_g.add_conditional_edges(
        "planner",
        lambda s: s.get("next_step") or "summarize",
        {"forecast": "tool", "scenario": "tool", "advice": "tool",
         "lookup": "tool", "asset_lookup": "tool", "mutate": "tool",
         "summarize": "summarize"},
    )
    agent_g.add_edge("tool", "summarize")
    agent_g.add_edge("summarize", END)
    agent_app = agent_g.compile()

    # /ask: read-only, no planner, no tools
    ask_g = StateGraph(AgentState)
    ask_g.add_node("safety", safety_node)
    ask_g.add_node("load_snapshot", snapshot_node)
    ask_g.add_node("summarize", summarize_node)
    ask_g.set_entry_point("safety")
    ask_g.add_edge("safety", "load_snapshot")
    ask_g.add_edge("load_snapshot", "summarize")
    ask_g.add_edge("summarize", END)
    ask_app = ask_g.compile()

    return agent_app, ask_app


agent_app, ask_app = build_graphs()
