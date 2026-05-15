"""Planner routing tests. Patches `chat()` so no network is required."""

from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src.agents.nodes import VALID_STEPS, planner_node


def _stub_response(content: str):
    """Mimic the litellm.completion response shape used by planner_node."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _state(text: str) -> dict:
    return {"messages": [HumanMessage(content=text)],
            "user_id": "user_1", "tool_results": [], "citations": [], "scratchpad": ""}


def test_planner_picks_forecast():
    with patch("src.agents.nodes._llm.chat", return_value=_stub_response("forecast")):
        out = planner_node(_state("In 10 years what will my NW be?"))
    assert out["next_step"] == "forecast"


def test_planner_picks_advice():
    with patch("src.agents.nodes._llm.chat", return_value=_stub_response("advice")):
        out = planner_node(_state("Where should I deploy this surplus?"))
    assert out["next_step"] == "advice"


def test_planner_falls_back_to_summarize_on_garbage():
    with patch("src.agents.nodes._llm.chat", return_value=_stub_response("???")):
        out = planner_node(_state("hi"))
    assert out["next_step"] == "summarize"


def test_planner_strips_quotes_and_punctuation():
    with patch("src.agents.nodes._llm.chat", return_value=_stub_response("'lookup'.")):
        out = planner_node(_state("what's a typical HYSA APY?"))
    assert out["next_step"] == "lookup"


def test_planner_falls_back_to_summarize_when_chain_exhausted():
    """If every model in the chain raises, planner should still return a
    valid next_step rather than crashing the graph."""
    with patch("src.agents.nodes._llm.chat", side_effect=RuntimeError("all models down")):
        out = planner_node(_state("anything"))
    assert out["next_step"] == "summarize"


def test_valid_steps_set_complete():
    assert VALID_STEPS == {"forecast", "scenario", "advice", "lookup",
                            "asset_lookup", "mutate", "summarize"}


def test_planner_picks_asset_lookup():
    with patch("src.agents.nodes._llm.chat", return_value=_stub_response("asset_lookup")):
        out = planner_node(_state("Name: BTC\nType: Crypto\nLook up value."))
    assert out["next_step"] == "asset_lookup"


def test_planner_picks_mutate():
    with patch("src.agents.nodes._llm.chat", return_value=_stub_response("mutate")):
        out = planner_node(_state("add 1.5 BTC at ₱4,500,000"))
    assert out["next_step"] == "mutate"
