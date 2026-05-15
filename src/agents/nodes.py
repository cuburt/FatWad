"""Node implementations for the LangGraph workflows.

Each node takes the AgentState and returns a partial dict that LangGraph
merges back in. Nodes are the only place the agent talks to the LLM, the
DB, or the tools — the graph wiring stays declarative."""

import json
import re
from typing import Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage

from src.agents import llm as _llm
from src.agents.prompts import PLANNER_PROMPT, SUMMARIZER_PROMPT, TOOL_PROMPT
from src.agents.state import AgentState
from src.safety import redact_pii_text
from src.tools import forecast as tools_forecast
from src.tools import market as tools_market
from src.tools import mutations as tools_mutations
from src.tools.advice import buy_list, rebalance_signals
from src.tools.snapshot import get_state, state_to_summary

logger = structlog.get_logger(__name__)


# ---- safety ---------------------------------------------------------------

def safety_node(state: AgentState) -> dict:
    if not state.get("messages"):
        return {}
    last = state["messages"][-1]
    sanitized = redact_pii_text(last.content)
    if sanitized != last.content:
        return {"messages": [HumanMessage(content=sanitized)]}
    return {}


# ---- snapshot -------------------------------------------------------------

def snapshot_node(state: AgentState) -> dict:
    user_id = state.get("user_id") or "user_1"
    snap = get_state(user_id)
    return {
        "snapshot": snap,
        "snapshot_text": state_to_summary(snap),
        "tool_results": [],
        "citations": [],
    }


# ---- planner --------------------------------------------------------------

VALID_STEPS = {"forecast", "scenario", "advice", "lookup", "asset_lookup",
                "mutate", "summarize"}


def planner_node(state: AgentState) -> dict:
    last_msg = _last_user_message(state)
    if not last_msg:
        return {"next_step": "summarize"}
    try:
        resp = _llm.chat(
            messages=[
                {"role": "system", "content": PLANNER_PROMPT.format(last_msg=last_msg)},
                {"role": "user", "content": last_msg},
            ],
            tier=_llm.ROUTER,
            temperature=0.1,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("planner_failed_fallback_summarize", error=str(e))
        return {"next_step": "summarize"}
    decision = content.strip().lower().split()[0] if content else "summarize"
    decision = decision.strip("'\".:` ")
    if decision not in VALID_STEPS:
        decision = "summarize"
    logger.info("planner_decision", decision=decision)
    return {"next_step": decision}


# ---- tool dispatch --------------------------------------------------------

TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "compute_compound",
            "description": "Project a portfolio over N years with monthly contributions and a fixed annual return.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_value": {"type": "number"},
                    "monthly_contribution": {"type": "number"},
                    "annual_return": {"type": "number"},
                    "years": {"type": "integer", "minimum": 1, "maximum": 80},
                },
                "required": ["start_value", "monthly_contribution", "annual_return", "years"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_freedom_date",
            "description": "Solve when investment yield covers monthly burn (financial-independence date).",
            "parameters": {
                "type": "object",
                "properties": {
                    "net_worth": {"type": "number"},
                    "monthly_burn": {"type": "number"},
                    "monthly_surplus": {"type": "number"},
                    "annual_return": {"type": "number"},
                },
                "required": ["net_worth", "monthly_burn", "monthly_surplus", "annual_return"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_scenario",
            "description": "Run an N-year projection with delta adjustments to NW, surplus, and return.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_nw": {"type": "number"},
                    "base_surplus": {"type": "number"},
                    "base_return": {"type": "number"},
                    "delta_nw": {"type": "number"},
                    "delta_surplus": {"type": "number"},
                    "delta_return": {"type": "number"},
                    "years": {"type": "integer", "minimum": 1, "maximum": 80, "default": 30},
                },
                "required": ["base_nw", "base_surplus", "base_return",
                              "delta_nw", "delta_surplus", "delta_return"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_required_monthly",
            "description": "Solve for the monthly contribution required to hit a target NW.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "number"}, "current": {"type": "number"},
                    "annual_return": {"type": "number"}, "years": {"type": "integer"},
                },
                "required": ["target", "current", "annual_return", "years"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_required_return",
            "description": "Solve for the annual return required to hit a target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "number"}, "current": {"type": "number"},
                    "monthly_contribution": {"type": "number"}, "years": {"type": "integer"},
                },
                "required": ["target", "current", "monthly_contribution", "years"],
            },
        },
    },
]


# Write tools — only available on the `mutate` planner path.
MUTATE_TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "add_asset",
            "description": "Add a new asset to the user's ledger. Call when the user "
                            "says they bought, acquired, or want to track an asset. "
                            "current_value and purchase_price are in PHP.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string",
                              "enum": ["Cash", "Real Estate", "Equity", "Crypto", "Physical"]},
                    "current_value": {"type": "number", "description": "PHP value today"},
                    "purchase_price": {"type": "number",
                                         "description": "PHP cost basis. Omit if unknown."},
                    "ai_note": {"type": "string", "description": "Optional rationale to save with the row."},
                },
                "required": ["name", "type", "current_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_settings",
            "description": "Update one or more user settings. Only pass fields the user "
                            "actually wants to change. risk_profile must be one of "
                            "Aggressive | Balanced | Defensive. Returns expected_return etc. "
                            "are decimals (0.07 = 7%).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expected_return": {"type": "number"},
                    "savings_apy": {"type": "number"},
                    "speculative_return": {"type": "number"},
                    "ghost_mode": {"type": "boolean"},
                    "risk_profile": {"type": "string",
                                       "enum": ["Aggressive", "Balanced", "Defensive"]},
                    "goal_target": {"type": "number"},
                    "goal_current_age": {"type": "integer"},
                    "goal_target_age": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_income_stream",
            "description": "Add a monthly inflow (salary, dividend, side gig). monthly is in PHP.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "monthly": {"type": "number"},
                },
                "required": ["source", "monthly"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_fixed_outflow",
            "description": "Add a recurring monthly bill (rent, insurance, subscription). monthly is in PHP.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bill": {"type": "string"},
                    "monthly": {"type": "number"},
                },
                "required": ["bill", "monthly"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_variable_burn",
            "description": "Log a weekly variable-spending total. week_iso is the Monday "
                            "of that week in YYYY-MM-DD. amount is PHP for the whole week.",
            "parameters": {
                "type": "object",
                "properties": {
                    "week_iso": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["week_iso", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_scenario",
            "description": "Add a Time-Machine what-if branch. deltas are signed PHP / decimal return.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "delta_nw": {"type": "number"},
                    "delta_surplus": {"type": "number"},
                    "delta_return": {"type": "number"},
                },
                "required": ["name"],
            },
        },
    },
]


def _project(start: float, monthly: float, ar: float, years: int) -> dict:
    bals = tools_forecast.project_net_worth(float(start), float(monthly), float(ar), int(years))
    return {"yearly_balances": [round(b, 2) for b in bals],
            "final_value": round(bals[-1], 2), "years": int(years)}


def _execute_tool(name: str, args: dict) -> dict:
    if name == "compute_compound":
        return _project(args["start_value"], args["monthly_contribution"],
                          args["annual_return"], args["years"])
    if name == "compute_freedom_date":
        return tools_forecast.freedom_date(args["net_worth"], args["monthly_burn"],
                                              args["monthly_surplus"], args["annual_return"])
    if name == "simulate_scenario":
        years = int(args.get("years", 30))
        return _project(
            args["base_nw"] + args["delta_nw"],
            max(0.0, args["base_surplus"] + args["delta_surplus"]),
            args["base_return"] + args["delta_return"], years,
        )
    if name == "compute_required_monthly":
        return {"required_monthly": round(tools_forecast.required_monthly(
            args["target"], args["current"], args["annual_return"], args["years"]), 2)}
    if name == "compute_required_return":
        v = tools_forecast.required_return(args["target"], args["current"],
                                              args["monthly_contribution"], args["years"])
        return {"required_return": round(v, 4) if v is not None else None}

    # --- write tools (only available on the `mutate` path) ----------------
    if name == "add_asset":
        return tools_mutations.add_asset(**args)
    if name == "update_settings":
        return tools_mutations.update_settings(**args)
    if name == "add_income_stream":
        return tools_mutations.add_income_stream(**args)
    if name == "add_fixed_outflow":
        return tools_mutations.add_fixed_outflow(**args)
    if name == "add_variable_burn":
        return tools_mutations.add_variable_burn(**args)
    if name == "add_scenario":
        return tools_mutations.add_scenario(**args)

    raise ValueError(f"Unknown tool: {name}")


def tool_node(state: AgentState) -> dict:
    """Dispatches based on planner's next_step. forecast/scenario use the
    LLM tool-call loop. advice runs deterministic Python. lookup hits the
    web-grounded prefill agent."""
    next_step = state.get("next_step") or "summarize"
    snapshot = state.get("snapshot") or {}
    last_msg = _last_user_message(state) or ""
    snapshot_text = state.get("snapshot_text", "")

    if next_step == "advice":
        result = _advice_tool(snapshot)
        return {"tool_results": [{"tool": "advice", "result": result}]}

    if next_step == "lookup":
        try:
            data = tools_market.prefill_assumptions()
            return {"tool_results": [{"tool": "prefill_assumptions", "result": data}],
                    "citations": list(data.get("sources") or [])}
        except Exception as e:
            return {"tool_results": [{"tool": "prefill_assumptions", "error": str(e)}]}

    if next_step == "mutate":
        # Write path: agentic loop sees read + write tools. Compute tools stay
        # available so a single chat turn can compute, then write (e.g. "add a
        # scenario that drops my surplus by 1k for 5 years").
        return _llm_tool_loop(last_msg, snapshot_text,
                                tools=TOOL_SCHEMA + MUTATE_TOOL_SCHEMA)

    if next_step == "asset_lookup":
        name, asset_type, purchase_date = _parse_asset_query(last_msg)
        if not name:
            return {"tool_results": [{"tool": "prefill_asset_value",
                                         "error": "Could not parse asset description from query."}]}
        try:
            data = tools_market.prefill_asset_value(name, asset_type,
                                                       purchase_date=purchase_date)
            return {"tool_results": [{
                "tool": "prefill_asset_value",
                "args": {"name": name, "type": asset_type,
                          "purchase_date": purchase_date},
                "result": data,
            }], "citations": list(data.get("sources") or [])}
        except Exception as e:
            return {"tool_results": [{"tool": "prefill_asset_value", "error": str(e)}]}

    # forecast or scenario -> run LLM tool loop
    return _llm_tool_loop(last_msg, snapshot_text)


_ASSET_TYPE_HINTS = {
    "Crypto": r"\b(btc|bitcoin|eth|ethereum|sol|solana|crypto|coin|usdt|usdc)\b",
    "Equity": r"\b(shares?|stock|etf|psei|voo|spy|qqq|vti|vt|vxus|index fund)\b",
    "Real Estate": r"\b(condo|house|lot|property|real estate|land)\b",
    "Physical": r"\b(gold|silver|watch|jewelry|jewellery|oz |ounce)\b",
}


def _parse_asset_query(text: str) -> tuple[str, str, str | None]:
    """Pull (name, asset_type, purchase_date) out of the user message.

    Prefers structured `Name: ...` / `Type: ...` / `PurchaseDate: yyyy-mm-dd`
    (what the frontend sends), otherwise heuristically classifies from
    keywords. `purchase_date` is None when not present.
    """
    name_m = re.search(r"(?:^|\n)\s*name\s*:\s*(.+?)\s*(?:$|\n)", text, re.IGNORECASE)
    type_m = re.search(r"(?:^|\n)\s*type\s*:\s*([A-Za-z ]+?)\s*(?:$|\n)", text, re.IGNORECASE)
    date_m = re.search(
        r"(?:^|\n)\s*purchase\s*date\s*:\s*(\d{4}-\d{2}-\d{2})\s*(?:$|\n)",
        text, re.IGNORECASE,
    )
    purchase_date = date_m.group(1) if date_m else None

    if name_m and type_m:
        return (name_m.group(1).strip(),
                type_m.group(1).strip().title(),
                purchase_date)

    lower = text.lower()
    for atype, pattern in _ASSET_TYPE_HINTS.items():
        if re.search(pattern, lower):
            return text.strip(), atype, purchase_date
    return text.strip(), "Cash", purchase_date


def _advice_tool(snapshot: dict) -> dict:
    profile = snapshot.get("risk_profile") or "Balanced"
    cf = tools_forecast.cashflow(
        snapshot.get("income", []), snapshot.get("fixed_outflow", []),
        snapshot.get("variable_burn", []),
    )
    return {
        "profile": profile,
        "buy_list": buy_list(max(0.0, cf.surplus), profile),
        "rebalance_signals": rebalance_signals(snapshot.get("assets", []), profile),
        "monthly_surplus": round(cf.surplus, 2),
    }


def _llm_tool_loop(last_msg: str, snapshot_text: str, *,
                     tools: list[dict] | None = None, max_steps: int = 4) -> dict:
    """Call the AGENTIC-tier model with `tools` (defaults to TOOL_SCHEMA),
    execute calls, collect results. The summarizer node turns this into prose."""
    tool_schema = tools if tools is not None else TOOL_SCHEMA
    messages = [
        {"role": "system", "content": TOOL_PROMPT.format(last_msg=last_msg, snapshot_text=snapshot_text)},
        {"role": "user", "content": last_msg},
    ]
    trace: list[dict] = []
    for _ in range(max_steps):
        try:
            resp = _llm.chat(
                messages=messages, tier=_llm.AGENTIC,
                tools=tool_schema, tool_choice="auto", temperature=0.2,
            )
        except Exception as e:
            return {"tool_results": [*trace, {"error": str(e)}]}
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            break
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = _execute_tool(tc.function.name, args)
            except Exception as e:
                result = {"error": str(e)}
            trace.append({"tool": tc.function.name, "args": args if 'args' in locals() else {},
                           "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                              "name": tc.function.name, "content": json.dumps(result, default=str)})
    return {"tool_results": trace}


# ---- summarizer -----------------------------------------------------------

def summarize_node(state: AgentState) -> dict:
    last_msg = _last_user_message(state) or ""
    snapshot_text = state.get("snapshot_text", "")
    tool_results = state.get("tool_results") or []

    sys = SUMMARIZER_PROMPT.format(
        snapshot_text=snapshot_text,
        tool_results=json.dumps(tool_results, default=str)[:6000],
    )
    try:
        resp = _llm.chat(
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": last_msg},
            ],
            tier=_llm.SYNTHESIS,
            temperature=0.3,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        content = f"Agent failed: {e}"
    return {"messages": [AIMessage(content=content)]}


# ---- helpers --------------------------------------------------------------

def _last_user_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return msg.content or ""
    return ""
