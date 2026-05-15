"""Tests for the asset_lookup intent: query parsing + tool_node dispatch."""

from unittest.mock import patch

import pytest

from src.agents.nodes import _parse_asset_query, tool_node


def test_parses_structured_name_and_type():
    name, atype, pdate = _parse_asset_query("Look up value.\nName: 1.5 BTC\nType: Crypto")
    assert name == "1.5 BTC"
    assert atype == "Crypto"
    assert pdate is None


def test_parses_purchase_date():
    name, atype, pdate = _parse_asset_query(
        "Name: 3.154 shares of TSM\nType: Equity\nPurchaseDate: 2023-06-15"
    )
    assert name == "3.154 shares of TSM"
    assert atype == "Equity"
    assert pdate == "2023-06-15"


def test_parses_lowercase_and_extra_whitespace():
    name, atype, _ = _parse_asset_query("name:   100 VOO shares  \ntype:   Equity")
    assert name == "100 VOO shares"
    assert atype == "Equity"


def test_falls_back_to_keyword_classification_for_freeform():
    name, atype, _ = _parse_asset_query("what's 1.5 BTC worth in pesos?")
    assert "BTC" in name
    assert atype == "Crypto"


def test_keyword_classification_for_equity_keywords():
    _, atype, _ = _parse_asset_query("100 shares of VOO")
    assert atype == "Equity"


def test_keyword_classification_for_real_estate():
    _, atype, _ = _parse_asset_query("BGC condo unit")
    assert atype == "Real Estate"


def test_keyword_classification_defaults_to_cash():
    _, atype, _ = _parse_asset_query("some random thing")
    assert atype == "Cash"


def test_tool_node_dispatches_to_prefill_asset_value(monkeypatch):
    """Planner says asset_lookup -> tool_node should call prefill_asset_value
    with parsed args and surface the result through tool_results."""
    from langchain_core.messages import HumanMessage

    fake_result = {"current_value": 4_500_000, "assumed_unit": "per BTC",
                    "sources": ["https://example/btc"], "note": "spot"}

    with patch("src.agents.nodes.tools_market.prefill_asset_value",
                 return_value=fake_result) as mocked:
        out = tool_node({
            "next_step": "asset_lookup",
            "snapshot": {},
            "snapshot_text": "",
            "messages": [HumanMessage(content="Name: 1.5 BTC\nType: Crypto")],
        })

    mocked.assert_called_once_with("1.5 BTC", "Crypto", purchase_date=None)
    assert out["tool_results"][0]["tool"] == "prefill_asset_value"
    assert out["tool_results"][0]["result"] == fake_result
    assert out["tool_results"][0]["args"] == {"name": "1.5 BTC", "type": "Crypto",
                                                 "purchase_date": None}
    assert out["citations"] == ["https://example/btc"]


def test_tool_node_surfaces_errors_without_raising(monkeypatch):
    from langchain_core.messages import HumanMessage

    def _boom(*_a, **_kw):
        raise RuntimeError("upstream down")

    with patch("src.agents.nodes.tools_market.prefill_asset_value", side_effect=_boom):
        out = tool_node({
            "next_step": "asset_lookup", "snapshot": {}, "snapshot_text": "",
            "messages": [HumanMessage(content="Name: BTC\nType: Crypto")],
        })

    assert "error" in out["tool_results"][0]
    assert out["tool_results"][0]["tool"] == "prefill_asset_value"
