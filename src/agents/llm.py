"""LLM client with per-tier fallback chains.

Three tiers — ROUTER, AGENTIC, SYNTHESIS — each with an ordered chain of
models. The first entry per tier comes from .env, the rest are hardcoded
fallbacks. We walk the chain on rate-limit (429) or any other call failure.

When an `openrouter/*` entry hits 429 we mark a short cooldown and skip every
remaining `openrouter/*` entry until it expires — OpenRouter's free-tier cap
is account-wide, so retrying other free models burns latency for nothing.

Vercel AI Gateway sits at the bottom of every chain as the paid escape hatch.
It speaks OpenAI-compatible at https://ai-gateway.vercel.sh/v1, so we route
to it through litellm's `openai/...` provider with `api_base` swapped.
"""

import os
import time
from typing import Any

import litellm
import structlog

logger = structlog.get_logger(__name__)

os.environ.setdefault("LITELLM_LOG", "WARNING")

ROUTER = "router"
AGENTIC = "agentic"
SYNTHESIS = "synthesis"

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
VERCEL_BASE = "https://ai-gateway.vercel.sh/v1"

# Primary model per tier from .env; fallbacks hardcoded so a missing env var
# can't accidentally drop a tier to nothing.
MODEL_CHAINS: dict[str, list[str]] = {
    ROUTER: [
        os.getenv("ROUTER_MODEL", "openrouter/openai/gpt-4o-mini"),
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "vercel/openai/gpt-4o-mini",
    ],
    AGENTIC: [
        os.getenv("AGENTIC_MODEL", "openrouter/openai/gpt-4o-mini"),
        "openrouter/anthropic/claude-3.5-sonnet",
        "vercel/openai/gpt-4o-mini",
    ],
    SYNTHESIS: [
        os.getenv("SYNTHESIS_MODEL", "openrouter/openai/gpt-4o-mini"),
        "openrouter/anthropic/claude-3.5-sonnet",
        "vercel/openai/gpt-4o-mini",
    ],
}

# OpenRouter's free-tier 20 RPM cap is account-wide, so when we see a 429 from
# one model the rest are about to 429 too. Mark a cooldown and skip every
# openrouter/* entry until it expires.
_OPENROUTER_COOLDOWN_S = 60.0
_openrouter_skip_until: float = 0.0


def _is_openrouter(model: str) -> bool:
    return model.startswith("openrouter/")


def _is_vercel(model: str) -> bool:
    return model.startswith("vercel/")


def _is_rate_limit(exc: Exception) -> bool:
    name = type(exc).__name__
    if "RateLimit" in name or "TooManyRequests" in name:
        return True
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


def _provider_kwargs(model: str) -> dict[str, Any]:
    """Returns the litellm.completion kwargs for one chain entry, or {} if its
    API key isn't configured (caller should skip)."""
    if _is_vercel(model):
        key = os.environ.get("VERCEL_API_KEY", "").strip()
        if not key:
            return {}
        # Vercel AI Gateway speaks OpenAI-compat; strip the "vercel/" prefix
        # and route through litellm's openai/... provider.
        return {
            "model": "openai/" + model[len("vercel/"):],
            "api_base": VERCEL_BASE,
            "api_key": key,
        }
    if _is_openrouter(model):
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            return {}
        return {"model": model, "api_base": OPENROUTER_BASE, "api_key": key}
    # Native litellm provider (anthropic/, gemini/, openai/, ...).
    # litellm picks up its own provider env keys.
    return {"model": model}


def chat(messages: list[dict[str, Any]], tier: str, **call_kwargs: Any) -> Any:
    """Walk the chain for `tier` and return the first successful litellm
    completion. Raises the last exception if every entry fails or is unconfigured.

    Pass through any litellm.completion kwargs (tools, tool_choice, temperature,
    response_format, stream, etc.).
    """
    global _openrouter_skip_until
    chain = MODEL_CHAINS.get(tier)
    if not chain:
        raise ValueError(f"Unknown tier: {tier}")

    last_exc: Exception | None = None
    last_skip_reason: str | None = None
    for entry in chain:
        if _is_openrouter(entry) and time.time() < _openrouter_skip_until:
            last_skip_reason = "openrouter_cooldown"
            logger.info("llm_chain_skip", model=entry, reason=last_skip_reason,
                         wait_s=round(_openrouter_skip_until - time.time(), 1))
            continue
        kwargs = _provider_kwargs(entry)
        if not kwargs:
            last_skip_reason = "missing_api_key"
            logger.info("llm_chain_skip", model=entry, reason=last_skip_reason)
            continue
        try:
            return litellm.completion(messages=messages, **kwargs, **call_kwargs)
        except Exception as e:
            last_exc = e
            logger.warning("llm_chain_failed", model=entry,
                            error_type=type(e).__name__, error=str(e)[:200])
            if _is_openrouter(entry) and _is_rate_limit(e):
                _openrouter_skip_until = time.time() + _OPENROUTER_COOLDOWN_S
                logger.info("openrouter_cooldown_set",
                              cooldown_until=_openrouter_skip_until)
            continue

    if last_exc:
        raise last_exc
    raise RuntimeError(
        f"No usable model in chain for tier={tier}. "
        f"Last skip reason: {last_skip_reason}. "
        f"Set OPENROUTER_API_KEY and/or VERCEL_API_KEY."
    )


def chat_stream(messages: list[dict[str, Any]], tier: str, **call_kwargs: Any):
    """Streaming variant of `chat`. Yields content chunks (strings) until the
    upstream model is done. On rate-limit *before* any token has been sent,
    walks to the next chain entry. Once a stream has started successfully,
    mid-stream failures bubble up (we can't unwind partial output)."""
    global _openrouter_skip_until
    chain = MODEL_CHAINS.get(tier)
    if not chain:
        raise ValueError(f"Unknown tier: {tier}")

    last_exc: Exception | None = None
    for entry in chain:
        if _is_openrouter(entry) and time.time() < _openrouter_skip_until:
            logger.info("llm_chain_skip", model=entry, reason="openrouter_cooldown")
            continue
        kwargs = _provider_kwargs(entry)
        if not kwargs:
            continue
        try:
            stream = litellm.completion(
                messages=messages, stream=True, **kwargs, **call_kwargs,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content or ""
                except (AttributeError, IndexError):
                    delta = ""
                if delta:
                    yield delta
            return
        except Exception as e:
            last_exc = e
            logger.warning("llm_stream_failed", model=entry,
                            error_type=type(e).__name__, error=str(e)[:200])
            if _is_openrouter(entry) and _is_rate_limit(e):
                _openrouter_skip_until = time.time() + _OPENROUTER_COOLDOWN_S
            continue

    if last_exc:
        raise last_exc
    raise RuntimeError(f"No usable streaming model in chain for tier={tier}")


def reset_openrouter_cooldown() -> None:
    """Test hook."""
    global _openrouter_skip_until
    _openrouter_skip_until = 0.0
