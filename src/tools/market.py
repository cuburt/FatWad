"""Web-grounded one-shot lookups: current macro assumptions, current asset
values. The agent calls these via litellm with OpenRouter's :online suffix —
the model browses, then returns strict JSON."""

import json
import os
import re
from typing import Any

import litellm

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
ONLINE_SUFFIX = ":online"
DEFAULT_MODEL = os.getenv("AGENTIC_MODEL", "openrouter/openai/gpt-4o-mini")


class APIKeyMissing(RuntimeError):
    """Raised when OPENROUTER_API_KEY isn't configured. Caller should surface this."""


def _resolve_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise APIKeyMissing("OPENROUTER_API_KEY is not set in the environment.")
    return key


def _online(model: str) -> str:
    return model if model.endswith(ONLINE_SUFFIX) else model + ONLINE_SUFFIX


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"Could not extract JSON: {text[:200]}")
        return json.loads(m.group(0))


def _json_call(system: str, user: str, model: str | None = None) -> dict:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    common = dict(model=_online(model or DEFAULT_MODEL), api_key=_resolve_key(),
                    api_base=OPENROUTER_BASE, messages=msgs, temperature=0.2)
    try:
        resp = litellm.completion(**common, response_format={"type": "json_object"})
    except Exception:
        resp = litellm.completion(**common)
    return _parse_json(resp.choices[0].message.content or "")


def prefill_assumptions(model: str | None = None) -> dict:
    sys = ("You are a markets analyst for a Philippine retail investor. Use the web "
            "to ground answers in TODAY's conditions. "
            "Reply with strict JSON only — no prose, no code fences.")
    user = (
        "Return JSON with these float fields (decimals between 0 and 1.5):\n"
        "- expected_return: long-term nominal PHP-denominated annualized return for a "
        "globally diversified equity portfolio held by a Philippine investor (e.g. a mix "
        "of PSEi index, US/global index funds via local brokers like COL/AAP, or peso "
        "feeder funds). Express as a decimal, e.g. 0.08.\n"
        "- savings_apy: typical TOP peso high-yield savings APY available to Philippine "
        "retail investors RIGHT NOW (e.g. CIMB, Maya, SeaBank, Tonik, GoTyme, ING). "
        "Decimal, e.g. 0.045.\n"
        "- speculative_return: defensible 10-year forward annualized return for a "
        "diversified crypto basket, in PHP terms. Decimal.\n"
        "Plus:\n"
        "- sources: array of 1-3 URL strings\n"
        "- note: rationale string, max 220 characters"
    )
    return _json_call(sys, user, model)


def prefill_asset_value(name: str, asset_type: str, purchase_date: str | None = None,
                          model: str | None = None) -> dict:
    """Look up an asset's market value (and optionally historical cost basis)
    and compute PHP totals in Python — never trust the LLM's arithmetic.

    The model returns components only — quantity, today's spot price, today's
    FX, and (if `purchase_date` is given) the historical price + FX. We do
    every multiplication ourselves. This avoids the common failure mode where
    the LLM looks up the right per-unit price but flubs the final math or
    forgets the FX conversion.
    """
    sys = (
        "You are a financial-data lookup agent for a Philippine investor. "
        "Use the web to find spot price and FX rate. "
        "Reply with strict JSON only — no prose, no code fences. "
        "Do NOT compute final peso totals yourself — the caller will multiply."
    )
    user_lines = [
        f"Asset description: '{name}'",
        f"Type: {asset_type}",
    ]
    if purchase_date:
        user_lines.append(f"User's purchase date: {purchase_date} (ISO yyyy-mm-dd)")
    user_lines.append("")
    user_lines.append("Look up the components and return JSON with these fields:")
    user_lines.append(
        "- quantity: number of units implied by the description (e.g. 3.154 from "
        "'3.154 shares of TSM', 1.5 from '1.5 BTC'). If no quantity is given, return 1."
    )
    user_lines.append(
        "- unit_price_local: spot price per unit TODAY in the asset's NATIVE trading "
        "currency (e.g. USD for TSM NYSE ADR, PHP for SM on PSE, USD for BTC)."
    )
    user_lines.append("- source_currency: ISO 4217 code (e.g. 'USD', 'PHP', 'TWD').")
    user_lines.append(
        "- fx_rate_to_php: spot FX rate FROM source_currency TO PHP today (e.g. ~58.5 "
        "for USD→PHP). If source_currency is already PHP, return 1."
    )
    if purchase_date:
        user_lines.append(
            "- unit_price_local_at_purchase: close price on the user's purchase date in "
            "the SAME source_currency. If the date was a non-trading day, use the next "
            "available close."
        )
        user_lines.append(
            "- fx_rate_to_php_at_purchase: PHP FX rate on the user's purchase date. "
            "If source_currency is PHP, return 1."
        )
    user_lines.append("- assumed_unit: short string ('per share', 'per BTC', 'per oz').")
    user_lines.append("- sources: array of 1-3 URLs you actually used.")
    user_lines.append(
        "- note: <=200 char rationale. INCLUDE the multiplication, e.g. "
        "'3.154 sh × USD 405.15 × PHP 58.5/USD'. If the ticker is ambiguous "
        "(e.g. TSM = Taiwan Semi NYSE ADR vs Taiwan 2330), state which you used."
    )
    raw = _json_call(sys, "\n".join(user_lines), model)

    quantity = _as_float(raw.get("quantity"), default=1.0)
    unit_price = _as_float(raw.get("unit_price_local"), default=0.0)
    fx = _as_float(raw.get("fx_rate_to_php"), default=1.0)
    src_ccy = (raw.get("source_currency") or "?").upper()
    current_value = round(quantity * unit_price * fx, 2)

    raw["current_value"] = current_value
    raw["computation"] = {
        "quantity": quantity,
        "unit_price_local": unit_price,
        "source_currency": src_ccy,
        "fx_rate_to_php": fx,
        "formula": f"{quantity} × {unit_price} {src_ccy} × {fx} PHP/{src_ccy} "
                    f"= PHP {current_value:,.2f}",
    }

    if purchase_date:
        pp_unit = _as_float(raw.get("unit_price_local_at_purchase"), default=0.0)
        # Default historical FX to today's rate if the model didn't return one —
        # better than silently zeroing out cost basis.
        pp_fx = _as_float(raw.get("fx_rate_to_php_at_purchase"), default=fx)
        purchase_price = round(quantity * pp_unit * pp_fx, 2)
        raw["purchase_price"] = purchase_price
        raw["purchase_date"] = purchase_date
        raw["computation_at_purchase"] = {
            "quantity": quantity,
            "unit_price_local_at_purchase": pp_unit,
            "fx_rate_to_php_at_purchase": pp_fx,
            "formula": f"{quantity} × {pp_unit} {src_ccy} × {pp_fx} PHP/{src_ccy} "
                        f"= PHP {purchase_price:,.2f}  (as of {purchase_date})",
        }
    return raw


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default
