"""Tests for src.tools.market.prefill_asset_value — covers the regression
where the LLM looked up the right per-share price but botched quantity ×
price × FX. We now compute current_value in Python."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.tools import market


def _resp(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


def test_computes_php_from_usd_components():
    """3.154 shares of TSM @ USD 405.15 with FX 58.5 = PHP 74,759.46"""
    llm_json = (
        '{"quantity": 3.154, "unit_price_local": 405.15, '
        '"source_currency": "USD", "fx_rate_to_php": 58.5, '
        '"assumed_unit": "per share", "sources": ["https://example/tsm"], '
        '"note": "TSM NYSE ADR"}'
    )
    with patch.object(market, "litellm",
                        SimpleNamespace(completion=lambda **_: _resp(llm_json))):
        out = market.prefill_asset_value("3.154 shares of TSM", "Equity")

    assert out["current_value"] == pytest.approx(74_759.46, rel=1e-4)
    comp = out["computation"]
    assert comp["quantity"] == 3.154
    assert comp["unit_price_local"] == 405.15
    assert comp["fx_rate_to_php"] == 58.5
    assert "3.154 × 405.15" in comp["formula"]


def test_php_source_currency_no_fx_conversion():
    """Local PSE stock priced in PHP — fx should be 1, computed = qty × price."""
    llm_json = (
        '{"quantity": 100, "unit_price_local": 950.0, '
        '"source_currency": "PHP", "fx_rate_to_php": 1, '
        '"assumed_unit": "per share", "sources": [], "note": "SM PSE"}'
    )
    with patch.object(market, "litellm",
                        SimpleNamespace(completion=lambda **_: _resp(llm_json))):
        out = market.prefill_asset_value("100 shares of SM", "Equity")

    assert out["current_value"] == pytest.approx(95_000.0)


def test_missing_fields_default_safely():
    """If the model leaves out fx_rate, default 1 and don't crash."""
    llm_json = (
        '{"quantity": 1, "unit_price_local": 1000.0, '
        '"source_currency": "PHP", '
        '"assumed_unit": "per share", "sources": [], "note": "missing fx"}'
    )
    with patch.object(market, "litellm",
                        SimpleNamespace(completion=lambda **_: _resp(llm_json))):
        out = market.prefill_asset_value("X", "Equity")

    assert out["current_value"] == pytest.approx(1000.0)


def test_with_purchase_date_computes_both_current_and_basis():
    """Same TSM example, but with purchase date — should also compute cost basis."""
    llm_json = (
        '{"quantity": 3.154, "unit_price_local": 405.15, '
        '"source_currency": "USD", "fx_rate_to_php": 58.5, '
        '"unit_price_local_at_purchase": 100.0, "fx_rate_to_php_at_purchase": 54.0, '
        '"assumed_unit": "per share", "sources": ["https://example/tsm"], '
        '"note": "TSM NYSE ADR"}'
    )
    with patch.object(market, "litellm",
                        SimpleNamespace(completion=lambda **_: _resp(llm_json))):
        out = market.prefill_asset_value(
            "3.154 shares of TSM", "Equity", purchase_date="2023-06-15"
        )

    assert out["current_value"] == pytest.approx(74_759.46, rel=1e-4)
    # 3.154 × 100 × 54 = 17031.60
    assert out["purchase_price"] == pytest.approx(17_031.60, rel=1e-4)
    assert out["purchase_date"] == "2023-06-15"
    assert "computation_at_purchase" in out
    assert "2023-06-15" in out["computation_at_purchase"]["formula"]


def test_purchase_date_with_missing_historical_fx_falls_back_to_current_fx():
    """If the LLM omits fx_rate_to_php_at_purchase, we use today's FX rather
    than collapsing the cost basis to zero."""
    llm_json = (
        '{"quantity": 1, "unit_price_local": 1000, '
        '"source_currency": "USD", "fx_rate_to_php": 58.5, '
        '"unit_price_local_at_purchase": 500, '
        '"assumed_unit": "per share", "sources": [], "note": "no historical fx"}'
    )
    with patch.object(market, "litellm",
                        SimpleNamespace(completion=lambda **_: _resp(llm_json))):
        out = market.prefill_asset_value("X", "Equity", purchase_date="2023-01-01")

    # current = 1 × 1000 × 58.5 = 58500
    # purchase = 1 × 500 × 58.5 = 29250 (falls back to today's fx)
    assert out["current_value"] == pytest.approx(58_500.0)
    assert out["purchase_price"] == pytest.approx(29_250.0)


def test_garbage_values_dont_crash():
    """Non-numeric strings collapse to 0 / 1 defaults; current_value ends up 0."""
    llm_json = (
        '{"quantity": "lots", "unit_price_local": null, '
        '"source_currency": "USD", "fx_rate_to_php": "around 58", '
        '"assumed_unit": "per share", "sources": [], "note": "garbage"}'
    )
    with patch.object(market, "litellm",
                        SimpleNamespace(completion=lambda **_: _resp(llm_json))):
        out = market.prefill_asset_value("garbled", "Equity")

    assert out["current_value"] == 0.0
