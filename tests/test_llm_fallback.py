"""Fallback-chain tests for src.agents.llm.chat()."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.agents import llm


def _resp(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Each test gets a fresh cooldown clock and predictable env."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("VERCEL_API_KEY", "vrc-test")
    llm.reset_openrouter_cooldown()


def test_chain_uses_first_entry_when_it_succeeds():
    captured = {}

    def fake(messages, **kwargs):
        captured["model"] = kwargs.get("model")
        captured["api_base"] = kwargs.get("api_base")
        return _resp("ok")

    with patch.object(llm, "litellm", SimpleNamespace(completion=fake)):
        out = llm.chat([{"role": "user", "content": "hi"}], tier=llm.ROUTER)

    assert out.choices[0].message.content == "ok"
    assert captured["model"].startswith("openrouter/")
    assert captured["api_base"] == llm.OPENROUTER_BASE


def test_chain_falls_through_to_vercel_on_rate_limit():
    """Every openrouter/* entry 429s -> we land on vercel/ with the right
    api_base and the model prefix rewritten to openai/..."""
    calls: list[dict] = []

    class _RateLimit(Exception):
        pass
    _RateLimit.__name__ = "RateLimitError"

    def fake(messages, **kwargs):
        calls.append({"model": kwargs.get("model"), "api_base": kwargs.get("api_base")})
        if "openrouter/" in kwargs.get("model", ""):
            raise _RateLimit("429 too many requests")
        return _resp("from-vercel")

    with patch.object(llm, "litellm", SimpleNamespace(completion=fake)):
        out = llm.chat([{"role": "user", "content": "x"}], tier=llm.SYNTHESIS)

    assert out.choices[0].message.content == "from-vercel"
    assert calls[-1]["api_base"] == llm.VERCEL_BASE
    assert calls[-1]["model"].startswith("openai/")
    # First call was openrouter and triggered the cooldown.
    assert llm._openrouter_skip_until > 0


def test_cooldown_skips_remaining_openrouter_entries(monkeypatch):
    """Once cooldown is set, every openrouter/* entry should be skipped on the
    NEXT call without even hitting litellm."""
    # Force cooldown to be active.
    import time as _time
    monkeypatch.setattr(llm, "_openrouter_skip_until", _time.time() + 30)

    calls: list[str] = []

    def fake(messages, **kwargs):
        calls.append(kwargs.get("model", ""))
        return _resp("vercel-only")

    with patch.object(llm, "litellm", SimpleNamespace(completion=fake)):
        out = llm.chat([{"role": "user", "content": "x"}], tier=llm.AGENTIC)

    assert out.choices[0].message.content == "vercel-only"
    # No openrouter call should have been attempted.
    assert all(not m.startswith("openrouter/") for m in calls)
    assert any("openai/" in m for m in calls)  # vercel routes via openai/


def test_chain_raises_when_no_keys_configured(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("VERCEL_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="No usable model"):
        llm.chat([{"role": "user", "content": "x"}], tier=llm.ROUTER)


def test_chain_propagates_kwargs():
    """tools/tool_choice/temperature must reach litellm.completion."""
    captured = {}

    def fake(messages, **kwargs):
        captured.update(kwargs)
        return _resp("ok")

    with patch.object(llm, "litellm", SimpleNamespace(completion=fake)):
        llm.chat([{"role": "user", "content": "x"}], tier=llm.AGENTIC,
                  tools=[{"type": "function"}], tool_choice="auto",
                  temperature=0.42, response_format={"type": "json_object"})

    assert captured["tools"] == [{"type": "function"}]
    assert captured["tool_choice"] == "auto"
    assert captured["temperature"] == 0.42
    assert captured["response_format"] == {"type": "json_object"}


def test_unknown_tier_raises():
    with pytest.raises(ValueError, match="Unknown tier"):
        llm.chat([{"role": "user", "content": "x"}], tier="bogus")
