"""FatWad Streamlit frontend. Talks to the FastAPI backend over httpx.
No DB access here. The OpenRouter key never leaves the backend's environment."""

import json
import os
from datetime import date, datetime, timedelta
from typing import Any, Generator

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import html as _components_html

API_URL = os.getenv("API_URL", "http://localhost:8000")
ASSET_TYPES = ["Cash", "Real Estate", "Equity", "Crypto", "Physical"]

st.set_page_config(page_title="FatWad", page_icon="₱", layout="wide",
                    initial_sidebar_state="expanded")


# ---------- HTTP helpers --------------------------------------------------

def _api_get(path: str, **params: Any) -> Any:
    with httpx.Client(timeout=30.0) as c:
        r = c.get(f"{API_URL}{path}", params=params or None)
        r.raise_for_status()
        return r.json()


def _api_post(path: str, json: Any | None = None) -> Any:
    with httpx.Client(timeout=120.0) as c:
        r = c.post(f"{API_URL}{path}", json=json)
        r.raise_for_status()
        return r.json()


def _api_patch(path: str, json: Any) -> Any:
    with httpx.Client(timeout=30.0) as c:
        r = c.patch(f"{API_URL}{path}", json=json)
        r.raise_for_status()
        return r.json()


def _api_delete(path: str) -> Any:
    with httpx.Client(timeout=30.0) as c:
        r = c.delete(f"{API_URL}{path}")
        r.raise_for_status()
        return r.json()


def _api_get_bytes(path: str) -> bytes:
    with httpx.Client(timeout=60.0) as c:
        r = c.get(f"{API_URL}{path}")
        r.raise_for_status()
        return r.content


def _extract_tool_result(agent_response: dict, tool_name: str) -> dict | None:
    """Scan an /agent response for the first tool entry that matches `tool_name`
    and return its `result` dict. None if missing or errored."""
    for entry in agent_response.get("tool_results") or []:
        if entry.get("tool") == tool_name and entry.get("result"):
            return entry["result"]
    return None


def _format_note(data: dict) -> str:
    """Combine the model's rationale and the deterministic computation formula
    into a single ai_note string we persist on the asset row."""
    note = (data.get("note") or "").strip()
    comp = data.get("computation") or {}
    formula = (comp.get("formula") or "").strip()
    if note and formula:
        return f"{note}\nComputed: {formula}"
    return note or formula


def _api_stream(path: str, payload: dict) -> Generator[str, None, dict]:
    """Open an SSE stream against /ask/stream or /agent/stream and yield token
    strings. The final {done: true, session_id, ...} event is stashed onto
    `st.session_state['_last_stream_meta']` so the caller can pick it up after
    `st.write_stream` finishes."""
    st.session_state["_last_stream_meta"] = {}
    with httpx.stream("POST", f"{API_URL}{path}", json=payload, timeout=120.0) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            try:
                evt = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if "token" in evt:
                yield evt["token"]
            elif "error" in evt:
                yield f"\n\n_Error: {evt['error']}_"
                return {}
            elif evt.get("done"):
                st.session_state["_last_stream_meta"] = evt
                return evt
    return {}


# ---------- styling -------------------------------------------------------

def inject_css(ghost: bool) -> None:
    blur = "filter: blur(8px); transition: filter 120ms;" if ghost else ""
    st.markdown(
        f"""
        <style>
            html, body, [class*="css"] {{
                font-family: 'JetBrains Mono', 'IBM Plex Mono', ui-monospace, monospace;
            }}
            .stApp {{ background: #0b0b0b; color: #e8e8e8; }}
            .stApp h1, .stApp h2, .stApp h3 {{ font-weight: 800; letter-spacing: -0.02em; color: #fff; }}
            .stApp h1 {{ font-size: 3rem; line-height: 1; border-bottom: 4px solid #fff; padding-bottom: 8px; }}
            .stMetric {{ background: #141414; border: 1px solid #2a2a2a; padding: 10px 12px; }}
            [data-testid="stMetricValue"] {{ font-size: 1.6rem; font-weight: 800; }}
            [data-testid="stMetricLabel"] {{ text-transform: uppercase; letter-spacing: 0.08em; color: #9a9a9a; }}
            .stButton>button {{ background: #fff; color: #000; border: 0; border-radius: 0;
                font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }}
            .stButton>button:hover {{ background: #ffe600; color: #000; }}
            section[data-testid="stSidebar"] {{ background: #050505; border-right: 1px solid #1f1f1f; }}
            .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid #2a2a2a; }}
            .stTabs [data-baseweb="tab"] {{ background: transparent; color: #9a9a9a; border-radius: 0;
                padding: 10px 16px; text-transform: uppercase; letter-spacing: 0.08em;
                font-weight: 700; font-size: 0.85rem; }}
            .stTabs [aria-selected="true"] {{ color: #fff; border-bottom: 3px solid #ffe600; }}
            .ghost-numeric {{ {blur} }}
            .pill {{ display: inline-block; padding: 2px 8px; border: 1px solid #2a2a2a;
                font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #9a9a9a; }}
            .heat-cell {{ padding: 14px; border: 1px solid #2a2a2a; margin-bottom: 8px; }}
            .heat-glow {{ background: rgba(0, 255, 120, 0.12); border-color: #1f8f4d; }}
            .heat-dark {{ background: #0e0e0e; }}

            /* Floating CHAT FAB. _inject_chat_fab() injects this button into
               body and hides the real Streamlit button (via inline styles
               applied by JS, so we do not depend on any class-name
               convention). The clone forwards clicks to the hidden real
               button so React's onClick handler still fires. */
            button.fatwad-fab {{
                position: fixed !important;
                bottom: 28px !important;
                right: 28px !important;
                top: auto !important;
                left: auto !important;
                z-index: 999999;
                background: #ffe600;
                color: #000;
                border: 0;
                border-radius: 999px;
                padding: 16px 26px;
                font-weight: 800;
                font-size: 0.95rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                font-family: 'JetBrains Mono', 'IBM Plex Mono', ui-monospace, monospace;
                cursor: pointer;
                box-shadow: 0 8px 22px rgba(0, 0, 0, 0.45),
                              0 0 0 1px rgba(0, 0, 0, 0.6) inset;
                transition: transform 120ms ease, box-shadow 120ms ease,
                              background-color 120ms ease;
            }}
            button.fatwad-fab:hover {{
                background: #fff;
                transform: translateY(-2px);
                box-shadow: 0 14px 28px rgba(0, 0, 0, 0.55);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def money(v: float) -> str:
    if st.session_state.get("ghost_mode"):
        return "▮▮▮▮"
    sign = "-" if v < 0 else ""
    return f"{sign}₱{abs(v):,.0f}"


def pct(v: float | None) -> str:
    if v is None:
        return "—"
    if st.session_state.get("ghost_mode"):
        return "▮▮"
    return f"{v*100:.1f}%"


# ---------- data loaders --------------------------------------------------

def load_state() -> dict:
    return _api_get("/state")


# ---------- sidebar -------------------------------------------------------

def render_sidebar(state: dict) -> None:
    s = state["settings"]
    with st.sidebar:
        st.markdown("## FATWAD")
        st.caption("Manual. Brutal. Yours.")

        new_ghost = st.toggle("Ghost Mode", value=s.get("ghost_mode", False),
                                help="Blurs every dollar amount on screen.")
        st.session_state["ghost_mode"] = new_ghost
        if new_ghost != s.get("ghost_mode"):
            _api_patch("/settings", {"ghost_mode": new_ghost})

        st.markdown("---")
        st.caption("ASSUMPTIONS")
        er = st.number_input("Equity expected return", min_value=0.0, max_value=1.0,
                              value=float(s.get("expected_return", 0.07)), step=0.005, format="%.3f")
        sa = st.number_input("Cash APY", min_value=0.0, max_value=0.5,
                              value=float(s.get("savings_apy", 0.045)), step=0.005, format="%.3f")
        sp = st.number_input("Speculative expected return", min_value=0.0, max_value=2.0,
                              value=float(s.get("speculative_return", 0.15)), step=0.01, format="%.2f")
        if (er, sa, sp) != (s.get("expected_return"), s.get("savings_apy"), s.get("speculative_return")):
            _api_patch("/settings", {"expected_return": er, "savings_apy": sa, "speculative_return": sp})

        st.markdown("---")
        st.caption("AI / LLM")
        st.caption(f"Model: `{s.get('llm_model', '?')}`")
        st.caption("API key is loaded server-side from `.env` (OPENROUTER_API_KEY). "
                    "It never travels through this UI.")

        if st.button("AI PREFILL ASSUMPTIONS", use_container_width=True):
            try:
                with st.spinner("Browsing markets..."):
                    resp = _api_post("/agent", {
                        "query": "Refresh my market assumptions: equity expected return, "
                                  "cash APY, and speculative return for a Philippine investor.",
                        "persist_history": False,
                    })
                data = _extract_tool_result(resp, "prefill_assumptions")
                if not data:
                    raise RuntimeError("Agent did not return market assumptions.")
                fields = {k: float(data[k]) for k in
                           ("expected_return", "savings_apy", "speculative_return")
                           if data.get(k) is not None}
                if fields:
                    _api_patch("/settings", fields)
                st.session_state["last_prefill_note"] = data.get("note") or ""
                st.session_state["last_prefill_sources"] = data.get("sources") or []
                st.toast("Assumptions prefilled.")
                st.rerun()
            except httpx.HTTPStatusError as e:
                detail = e.response.json().get("detail", str(e)) if e.response else str(e)
                st.error(f"Prefill failed: {detail}")
            except Exception as e:
                st.error(f"Prefill failed: {e}")

        if st.session_state.get("last_prefill_note"):
            st.caption(st.session_state["last_prefill_note"])
            for url in st.session_state.get("last_prefill_sources", [])[:3]:
                st.caption(f"• {url}")

        st.markdown("---")
        st.caption(f"API: `{API_URL}`")

    # Weekly check-in nag
    last = s.get("last_checkin")
    due = True
    if last:
        try:
            due = datetime.utcnow() - datetime.fromisoformat(last.replace("Z", "")) > timedelta(days=7)
        except Exception:
            due = True
    if due:
        with st.sidebar:
            st.warning("WEEKLY CHECK-IN DUE")
            c1, c2 = st.columns(2)
            if c1.button("Mark Done", use_container_width=True):
                _api_post("/settings/checkin")
                st.rerun()
            if c2.button("Snooze", use_container_width=True):
                _api_post("/settings/checkin")
                st.rerun()


# ---------- tab: dashboard ------------------------------------------------

CASHFLOW_TYPES = {"Equity": True, "Real Estate": True, "Cash": True,
                  "Crypto": False, "Physical": False}


def tab_dashboard(state: dict) -> None:
    st.subheader("THE HEAT MAP",
                  help="Each tile is one of your assets. Glow = produces cashflow.")

    nw = sum(float(a.get("current_value", 0) or 0) for a in state["assets"])
    cb = sum(float(a.get("purchase_price", 0) or 0) for a in state["assets"])
    cf_summary = _cashflow_summary(state)
    yr = _annual_yield(state)

    a, b, c, d = st.columns(4)
    a.metric("NET WORTH", money(nw))
    b.metric("LIFETIME ROI", pct((nw - cb) / cb) if cb else "—")
    c.metric("EST. ANNUAL YIELD", money(yr))
    d.metric("MONTHLY SURPLUS", money(cf_summary["surplus"]))

    if not state["assets"]:
        st.info("Add assets to populate the heat map.")
        return

    st.markdown("---")
    cols = st.columns(3)
    for idx, asset in enumerate(state["assets"]):
        glow = CASHFLOW_TYPES.get(asset["type"], False) and float(asset.get("current_value", 0) or 0) > 0
        css_class = "heat-cell heat-glow" if glow else "heat-cell heat-dark"
        roi = ((asset["current_value"] - asset["purchase_price"]) / asset["purchase_price"]
                if asset["purchase_price"] else 0)
        cols[idx % 3].markdown(
            f"""
            <div class="{css_class}">
                <div class="pill">{asset['type'].upper()}</div>
                <h3 style="margin:6px 0;">{asset['name']}</h3>
                <div style="font-size:1.6rem;font-weight:800">{money(float(asset.get('current_value', 0) or 0))}</div>
                <div style="color:#9a9a9a;font-size:0.9rem">ROI {pct(roi)} · Basis {money(float(asset.get('purchase_price', 0) or 0))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "### BY ASSET CLASS",
        help="How your net worth is split across asset categories. "
              "A balanced wad usually doesn't have one slice over 60%.",
    )
    totals: dict[str, float] = {}
    for a in state["assets"]:
        v = float(a.get("current_value", 0) or 0)
        if v > 0:
            totals[a["type"]] = totals.get(a["type"], 0.0) + v
    if totals:
        # Hide labels in ghost mode but keep the chart shape.
        textinfo = "none" if st.session_state.get("ghost_mode") else "label+percent"
        fig = go.Figure(data=[go.Pie(
            labels=list(totals.keys()),
            values=list(totals.values()),
            hole=0.55,
            textinfo=textinfo,
            marker=dict(colors=["#ffe600", "#00d97e", "#3a86ff", "#ff5470", "#8338ec"]),
        )])
        fig.update_layout(
            plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
            font=dict(color="#e8e8e8", family="JetBrains Mono"),
            height=320, margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------- tab: assets ---------------------------------------------------

def tab_assets(state: dict) -> None:
    st.subheader("ASSET LEDGER")
    st.caption("Manual entries only. Update balances during weekly check-ins.")

    with st.expander("AI LOOKUP — let the agent fill the value"):
        a1, a2, a3, a4 = st.columns([2.5, 1.3, 1.5, 1])
        ai_name = a1.text_input("Describe asset", placeholder="e.g. 1.5 BTC, 100 shares VTI",
                                  key="ai_lookup_name")
        ai_type = a2.selectbox("Type", ASSET_TYPES, key="ai_lookup_type")
        ai_purchase_date = a3.date_input(
            "Purchase date (optional)", value=None, max_value=date.today(),
            key="ai_lookup_date",
            help="If set, the agent also looks up the historical price on that "
                  "date and fills the cost basis — so ROI is meaningful.",
        )
        if a4.button("LOOKUP & ADD", disabled=not ai_name):
            try:
                with st.spinner("Browsing for current price..."):
                    query_lines = [
                        "Look up the current PHP value of this asset.",
                        f"Name: {ai_name}",
                        f"Type: {ai_type}",
                    ]
                    if ai_purchase_date:
                        query_lines.append(f"PurchaseDate: {ai_purchase_date.isoformat()}")
                    resp = _api_post("/agent", {
                        "query": "\n".join(query_lines),
                        "persist_history": False,
                    })
                data = _extract_tool_result(resp, "prefill_asset_value")
                if not data:
                    raise RuntimeError("Agent did not return an asset value.")
                _api_post("/assets", {
                    "name": ai_name, "type": ai_type,
                    "purchase_price": float(data.get("purchase_price", 0) or 0),
                    "current_value": float(data.get("current_value", 0) or 0),
                    "ai_unit": data.get("assumed_unit"),
                    "ai_note": _format_note(data),
                    "ai_sources": data.get("sources") or [],
                })
                comp = data.get("computation") or {}
                comp_pp = data.get("computation_at_purchase") or {}
                breakdown_now = comp.get("formula", "")
                breakdown_pp = comp_pp.get("formula", "")
                msg_parts = [
                    f"Added '{ai_name}' at {money(float(data.get('current_value', 0) or 0))}."
                ]
                if data.get("purchase_price"):
                    msg_parts.append(
                        f"Cost basis @ {data.get('purchase_date', '?')}: "
                        f"{money(float(data['purchase_price']))}."
                    )
                if breakdown_now:
                    msg_parts.append(f"Now:  `{breakdown_now}`")
                if breakdown_pp:
                    msg_parts.append(f"Then: `{breakdown_pp}`")
                st.success("\n\n".join(msg_parts))
                st.rerun()
            except httpx.HTTPStatusError as e:
                detail = e.response.json().get("detail", str(e)) if e.response else str(e)
                st.error(f"Lookup failed: {detail}")
            except Exception as e:
                st.error(f"Lookup failed: {e}")

    with st.form("add_asset", clear_on_submit=True):
        c1, c2, c3, c4, c5 = st.columns([2.5, 1.5, 1.5, 1.5, 1])
        name = c1.text_input("Name")
        atype = c2.selectbox("Type", ASSET_TYPES)
        purchase = c3.number_input("Purchase price", min_value=0.0, step=100.0)
        current = c4.number_input("Current value", min_value=0.0, step=100.0)
        if c5.form_submit_button("ADD") and name:
            _api_post("/assets", {"name": name, "type": atype,
                                     "purchase_price": purchase, "current_value": current})
            st.rerun()

    if not state["assets"]:
        st.info("No assets yet.")
        return

    refreshable = [a for a in state["assets"] if a["type"] in REFRESHABLE_TYPES]
    if refreshable:
        rc1, rc2 = st.columns([1, 4])
        if rc1.button(f"REFRESH ALL ({len(refreshable)})",
                       help="Re-price every Equity/Crypto/Physical asset via the agent. "
                            "Cash and Real Estate are left alone."):
            _refresh_assets(refreshable)

    rows = [{
        "id": a["id"], "Name": a["name"], "Type": a["type"],
        "Cost Basis": a["purchase_price"], "Current Value": a["current_value"],
        # None renders as empty in NumberColumn, signalling "no cost basis yet"
        # rather than the misleading "0.0%" you'd see otherwise.
        "ROI %": (round((a["current_value"] - a["purchase_price"]) / a["purchase_price"] * 100, 2)
                   if a["purchase_price"] else None),
    } for a in state["assets"]]
    df = pd.DataFrame(rows)
    edited = st.data_editor(
        df.drop(columns=["id"]), num_rows="fixed", use_container_width=True,
        hide_index=True, disabled=["Type", "ROI %"],
        column_config={
            "Cost Basis": st.column_config.NumberColumn(format="₱%d"),
            "Current Value": st.column_config.NumberColumn(format="₱%d"),
            "ROI %": st.column_config.NumberColumn(format="%.1f%%"),
        },
        key="assets_editor",
    )
    for i, a in enumerate(state["assets"]):
        new_name = edited.iloc[i]["Name"]
        new_basis = float(edited.iloc[i]["Cost Basis"])
        new_value = float(edited.iloc[i]["Current Value"])
        if (new_name, new_basis, new_value) != (a["name"], a["purchase_price"], a["current_value"]):
            _api_patch(f"/assets/{a['id']}", {
                "name": new_name, "type": a["type"],
                "purchase_price": new_basis, "current_value": new_value,
                "ai_unit": a.get("ai_unit"), "ai_note": a.get("ai_note"),
                "ai_sources": a.get("ai_sources") or [],
            })

    st.markdown("**Per-asset actions**")
    for a in state["assets"]:
        can_refresh = a["type"] in REFRESHABLE_TYPES
        c1, c2, c3 = st.columns([4, 1, 1])
        label = f"{a['name']} ({a['type']})"
        c1.write(label)
        if c2.button("REFRESH", key=f"ref_{a['id']}", disabled=not can_refresh,
                       help=("Re-price via the agent." if can_refresh
                              else f"{a['type']} assets are priced manually.")):
            _refresh_assets([a])
        if c3.button("DELETE", key=f"del_{a['id']}"):
            _api_delete(f"/assets/{a['id']}")
            st.rerun()


REFRESHABLE_TYPES = {"Equity", "Crypto", "Physical"}


def _refresh_assets(assets: list[dict]) -> None:
    """Re-price a list of assets via the agent and PATCH the new values back.
    Sequential because web-grounded lookups are stateful per provider."""
    if not assets:
        return
    progress = st.progress(0.0, text="Starting…")
    ok = 0
    for i, a in enumerate(assets):
        progress.progress(i / len(assets), text=f"Refreshing {a['name']}…")
        try:
            resp = _api_post("/agent", {
                "query": (
                    "Look up the current PHP value of this asset.\n"
                    f"Name: {a['name']}\n"
                    f"Type: {a['type']}"
                ),
                "persist_history": False,
            })
            data = _extract_tool_result(resp, "prefill_asset_value")
            if not data or data.get("current_value") in (None, 0):
                raise RuntimeError("Agent returned no value")
            _api_patch(f"/assets/{a['id']}", {
                "name": a["name"], "type": a["type"],
                "purchase_price": a["purchase_price"],
                "current_value": float(data["current_value"]),
                "ai_unit": data.get("assumed_unit"),
                "ai_note": _format_note(data),
                "ai_sources": data.get("sources") or [],
            })
            ok += 1
        except Exception as e:
            st.warning(f"{a['name']}: {e}")
    progress.progress(1.0, text=f"Refreshed {ok}/{len(assets)}.")
    st.toast(f"Refreshed {ok}/{len(assets)} assets.")
    st.rerun()


# ---------- tab: cashflow -------------------------------------------------

def tab_cashflow(state: dict) -> None:
    st.subheader("THE CASHFLOW MATRIX")

    left, mid, right = st.columns(3)

    with left:
        st.markdown("**INFLOW STREAMS**")
        with st.form("add_income", clear_on_submit=True):
            n = st.text_input("Source"); m = st.number_input("Monthly ₱", min_value=0.0, step=1000.0)
            if st.form_submit_button("+ ADD") and n:
                _api_post("/income", {"source": n, "monthly": m}); st.rerun()
        for i in state["income"]:
            c1, c2 = st.columns([3, 1])
            c1.write(f"{i['source']} — {money(i['monthly'])}")
            if c2.button("×", key=f"di_{i['id']}"):
                _api_delete(f"/income/{i['id']}"); st.rerun()

    with mid:
        st.markdown("**FIXED OUTFLOW (THE BURN)**")
        with st.form("add_fixed", clear_on_submit=True):
            n = st.text_input("Bill"); m = st.number_input("Monthly ₱", min_value=0.0, step=500.0, key="fx")
            if st.form_submit_button("+ ADD") and n:
                _api_post("/fixed", {"bill": n, "monthly": m}); st.rerun()
        for f in state["fixed_outflow"]:
            c1, c2 = st.columns([3, 1])
            c1.write(f"{f['bill']} — {money(f['monthly'])}")
            if c2.button("×", key=f"df_{f['id']}"):
                _api_delete(f"/fixed/{f['id']}"); st.rerun()

    with right:
        st.markdown("**VARIABLE BURN (WEEKLY)**")
        with st.form("add_var", clear_on_submit=True):
            wk = st.date_input("Week of", value=date.today())
            amt = st.number_input("Total spent", min_value=0.0, step=100.0, key="va")
            if st.form_submit_button("+ ADD"):
                _api_post("/variable", {"week": wk.isoformat(), "amount": amt}); st.rerun()
        for v in state["variable_burn"][:8]:
            c1, c2 = st.columns([3, 1])
            c1.write(f"{v['week']} — {money(v['amount'])}")
            if c2.button("×", key=f"dv_{v['id']}"):
                _api_delete(f"/variable/{v['id']}"); st.rerun()

    cf = _cashflow_summary(state)
    st.markdown("---")
    a, b, c, d = st.columns(4)
    a.metric("INFLOW", money(cf["inflow"]))
    b.metric("FIXED BURN", money(cf["fixed"]))
    c.metric("VARIABLE BURN (avg)", money(cf["variable"]))
    d.metric("SURPLUS", money(cf["surplus"]),
             delta=None if cf["surplus"] >= 0 else "DEFICIT")


# ---------- floating chat dialog -----------------------------------------

SUGGESTED_PROMPTS = [
    "Summarize my position in one paragraph.",
    "Add 1.5 BTC to my ledger at ₱4,500,000.",
    "Set my goal to ₱10,000,000 by age 45.",
    "Raise my expected equity return to 8%.",
]

_WRITE_TOOLS = {
    "add_asset", "add_income_stream", "add_fixed_outflow",
    "add_variable_burn", "add_scenario", "update_settings",
}


def _summarize_writes(tool_results: list[dict]) -> list[str]:
    """Return short human-readable lines for any write-tool call this turn."""
    lines: list[str] = []
    for entry in tool_results or []:
        tool = entry.get("tool")
        if tool not in _WRITE_TOOLS:
            continue
        result = entry.get("result") or {}
        if not result.get("ok"):
            lines.append(f"✗ {tool}: {result.get('reason', 'failed')}")
            continue
        if tool == "add_asset":
            lines.append(f"✓ Added asset '{result.get('name')}' "
                         f"at {money(result.get('current_value', 0))}.")
        elif tool == "add_income_stream":
            lines.append(f"✓ Added income '{result.get('source')}' "
                         f"at {money(result.get('monthly', 0))}/mo.")
        elif tool == "add_fixed_outflow":
            lines.append(f"✓ Added bill '{result.get('bill')}' "
                         f"at {money(result.get('monthly', 0))}/mo.")
        elif tool == "add_variable_burn":
            lines.append(f"✓ Logged {money(result.get('amount', 0))} for "
                         f"week of {result.get('week')}.")
        elif tool == "add_scenario":
            lines.append(f"✓ Added scenario '{result.get('name')}'.")
        elif tool == "update_settings":
            updated = ", ".join(result.get("updated") or [])
            lines.append(f"✓ Updated settings: {updated}.")
    return lines


def _chat_body() -> None:
    """The chat UI. Rendered either inside a Streamlit dialog (preferred) or
    in the sidebar as a fallback for older Streamlit builds."""
    st.caption("Type a question OR a command. "
                "Commands like 'add 1.5 BTC at ₱4.5M' or 'set my goal to ₱10M' "
                "update the tabs underneath — close this dialog to see them.")

    history = st.session_state.setdefault("_chat_history", [])

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for line in msg.get("writes", []):
                st.success(line)

    # Suggested prompts as quick buttons (only show before the first turn).
    queued = None
    if not history:
        pcols = st.columns(len(SUGGESTED_PROMPTS))
        for col, p in zip(pcols, SUGGESTED_PROMPTS):
            if col.button(p[:34] + ("…" if len(p) > 34 else ""), help=p,
                           use_container_width=True, key=f"sp_{hash(p)}"):
                queued = p

    typed = st.chat_input("Ask or command…")
    user_msg = typed or queued
    if user_msg:
        history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(_api_stream("/agent/stream", {
                    "query": user_msg,
                    "persist_history": False,
                }))
                meta = st.session_state.get("_last_stream_meta") or {}
                writes = _summarize_writes(meta.get("tool_results") or [])
                for line in writes:
                    st.success(line)
                if meta.get("citations"):
                    st.caption("Sources: " + ", ".join(meta["citations"][:3]))
                history.append({"role": "assistant", "content": answer or "",
                                "writes": writes})
                if writes:
                    st.info("Close this dialog to refresh the tabs.")
            except httpx.HTTPStatusError as e:
                detail = e.response.json().get("detail", str(e)) if e.response else str(e)
                st.error(f"Chat failed: {detail}")
            except Exception as e:
                st.error(f"Chat failed: {e}")

    if history:
        col1, _ = st.columns([1, 4])
        if col1.button("CLEAR", use_container_width=True):
            st.session_state["_chat_history"] = []
            st.rerun()


def _inject_chat_fab() -> None:
    """Inject a clone of the CHAT button into document.body, forward its
    clicks to the real Streamlit button, and hide the real button by setting
    inline styles directly on its wrapper (don't rely on Streamlit's
    `st-key-{key}` class — version-dependent).

    Why a clone: a plain `position:fixed` on the Streamlit-rendered widget
    resolves against whichever ancestor has `transform`/`contain` — which
    Streamlit's main column usually has — so the button ends up mid-column
    instead of viewport corner. A clone parented to <body> has no
    transformed ancestors, so fixed positioning works as expected.
    """
    _components_html(
        """
<script>
(function () {
    const parentDoc = window.parent.document;
    const MARKER = "\\uD83D\\uDCAC";  // 💬 speech-balloon, surrogate pair

    function findRealButton() {
        const all = parentDoc.querySelectorAll('button');
        for (const b of all) {
            if (b.classList && b.classList.contains('fatwad-fab')) continue;
            if (b.textContent && b.textContent.indexOf(MARKER) !== -1) return b;
        }
        return null;
    }

    function findWrapper(btn) {
        // Walk up to the element-container Streamlit assigns to each widget.
        let el = btn;
        while (el && el !== parentDoc.body) {
            const t = el.dataset && el.dataset.testid;
            if (t === 'stElementContainer' || t === 'element-container') return el;
            el = el.parentElement;
        }
        return btn;
    }

    function hideWrapper(el) {
        // Inline styles guarantee the hide doesn't depend on Streamlit
        // emitting our expected CSS class.
        el.style.position = 'absolute';
        el.style.left = '-10000px';
        el.style.top = 'auto';
        el.style.width = '1px';
        el.style.height = '1px';
        el.style.minHeight = '0';
        el.style.margin = '0';
        el.style.padding = '0';
        el.style.overflow = 'hidden';
        el.style.opacity = '0';
        el.style.pointerEvents = 'none';
    }

    function ensureFab() {
        let fab = parentDoc.querySelector('button.fatwad-fab');
        if (!fab) {
            fab = parentDoc.createElement('button');
            fab.className = 'fatwad-fab';
            fab.type = 'button';
            fab.setAttribute('aria-label', 'Open chat');
            fab.textContent = MARKER + '  CHAT';
            parentDoc.body.appendChild(fab);
        }
        const real = findRealButton();
        if (real) {
            hideWrapper(findWrapper(real));
            fab.onclick = function () { real.click(); };
        }
    }

    ensureFab();

    if (!parentDoc._fatwadFabObserver) {
        parentDoc._fatwadFabObserver = new MutationObserver(ensureFab);
        parentDoc._fatwadFabObserver.observe(parentDoc.body,
            { childList: true, subtree: true });
    }
})();
</script>
        """,
        height=0,
    )


def chat_dialog() -> None:
    """Open the chat UI. Prefers st.dialog (Streamlit ≥1.37), then
    st.experimental_dialog (≥1.33), and finally falls back to rendering in
    the sidebar — so this works regardless of the installed Streamlit."""
    dialog_fn = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
    if dialog_fn is not None:
        # Some old experimental_dialog signatures don't accept `width`.
        try:
            decorator = dialog_fn("FATWAD CHAT", width="large")
        except TypeError:
            decorator = dialog_fn("FATWAD CHAT")
        @decorator
        def _modal():
            _chat_body()
        _modal()
        return

    # Final fallback: sidebar-anchored chat. Latches open so the user can
    # keep typing without re-clicking the button on every rerun.
    st.session_state["_chat_open"] = True
    with st.sidebar:
        st.markdown("---")
        st.subheader("FATWAD CHAT")
        _chat_body()
        if st.button("CLOSE CHAT", use_container_width=True):
            st.session_state["_chat_open"] = False
            st.rerun()


# ---------- tab: time machine + main character (kept compact) -------------

def tab_time_machine(state: dict) -> None:
    st.subheader("THE TIME MACHINE")
    nw = sum(float(a.get("current_value", 0) or 0) for a in state["assets"])
    cf = _cashflow_summary(state)
    er = float(state["settings"].get("expected_return", 0.07))

    a, b, c = st.columns(3)
    a.metric("NET WORTH (NOW)", money(nw))
    b.metric("MONTHLY SURPLUS", money(cf["surplus"]))
    c.metric("ASSUMED RETURN", pct(er))

    base = _project(nw, max(0.0, cf["surplus"]), er, 30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(31)), y=base, name="BASE",
                                line=dict(color="#ffe600", width=3)))

    with st.form("add_scenario", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([2.5, 1.2, 1.2, 1.2])
        sname = c1.text_input("Branch")
        d_nw = c2.number_input("ΔNW", value=0.0, step=1000.0)
        d_su = c3.number_input("ΔSurplus", value=0.0, step=100.0)
        d_re = c4.number_input("ΔReturn", value=0.0, step=0.005, format="%.3f")
        if st.form_submit_button("+ BRANCH") and sname:
            _api_post("/scenarios", {"name": sname, "delta_nw": d_nw,
                                        "delta_surplus": d_su, "delta_return": d_re})
            st.rerun()

    palette = ["#00d97e", "#ff5470", "#3a86ff", "#fb5607", "#8338ec"]
    for idx, sc in enumerate(state["scenarios"]):
        proj = _project(nw + sc["delta_nw"], max(0.0, cf["surplus"] + sc["delta_surplus"]),
                          er + sc["delta_return"], 30)
        fig.add_trace(go.Scatter(x=list(range(31)), y=proj, name=sc["name"],
                                    line=dict(color=palette[idx % len(palette)], dash="dot")))

    fig.update_layout(plot_bgcolor="#0b0b0b", paper_bgcolor="#0b0b0b",
                         font=dict(color="#e8e8e8", family="JetBrains Mono"),
                         xaxis=dict(title="YEARS", gridcolor="#1f1f1f"),
                         yaxis=dict(title="NET WORTH", gridcolor="#1f1f1f", tickprefix="₱", tickformat=",.0f"),
                         height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    for sc in state["scenarios"]:
        c1, c2 = st.columns([5, 1])
        c1.write(f"**{sc['name']}** — ΔNW {money(sc['delta_nw'])}, "
                  f"Δsurplus {money(sc['delta_surplus'])}, Δreturn {pct(sc['delta_return'])}")
        if c2.button("×", key=f"dsc_{sc['id']}"):
            _api_delete(f"/scenarios/{sc['id']}"); st.rerun()


def tab_main_character(state: dict) -> None:
    yr = _annual_yield(state)
    st.subheader("THE YIELD PUNCH",
                  help="What you 'earn' per hour just by existing. Annual yield ÷ 8,760 hours.")
    a, b, c = st.columns(3)
    a.metric("PER HOUR", money(yr / 8760))
    b.metric("PER DAY", money(yr / 365))
    c.metric("PER YEAR", money(yr))

    st.markdown("---")
    st.subheader("THE GOAL POST",
                  help="Move the slider to a target net worth. The app tells you what "
                       "monthly contribution — or what annual return — gets you there by "
                       "your target age.")
    goal = state["goal"]
    nw = sum(float(a.get("current_value", 0) or 0) for a in state["assets"])
    er = float(state["settings"].get("expected_return", 0.07))
    cf = _cashflow_summary(state)

    c1, c2, c3 = st.columns(3)
    target = c1.number_input("Target net worth (₱)", min_value=0.0,
                              value=float(goal["target"]), step=100_000.0,
                              format="%.0f",
                              help="Type the number you're aiming for. The math updates live.")
    current_age = c2.number_input("Current age", min_value=10, max_value=100,
                                    value=int(goal["current_age"]))
    target_age = c3.number_input("Target age", min_value=current_age + 1,
                                   max_value=100, value=max(int(goal["target_age"]),
                                                              current_age + 1))
    if (target != goal["target"] or current_age != goal["current_age"]
            or target_age != goal["target_age"]):
        _api_patch("/settings", {
            "goal_target": float(target),
            "goal_current_age": int(current_age),
            "goal_target_age": int(target_age),
        })

    years = target_age - current_age
    req_monthly = _required_monthly(target, nw, er, years)
    req_return = _required_return(target, nw, max(0.0, cf["surplus"]), years)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("YEARS TO TARGET", f"{years}")
    m2.metric("CURRENT NW", money(nw))
    m3.metric("REQ MONTHLY @ assumed return", money(req_monthly),
               help=f"Monthly contribution to hit {money(target)} in {years} years at "
                    f"{er*100:.1f}% annual return.")
    m4.metric("REQ ANNUAL RETURN @ current surplus",
               pct(req_return) if req_return is not None else "—",
               help=f"Annual return needed to hit {money(target)} in {years} years if you "
                    f"invest {money(cf['surplus'])} every month.")

    st.markdown("---")
    st.subheader("MANUAL EXPORT — PDF WEALTH REPORT")
    if st.button("GENERATE PDF"):
        try:
            pdf = _api_get_bytes("/report")
            st.download_button("DOWNLOAD", data=pdf,
                                  file_name=f"fatwad_report_{date.today().isoformat()}.pdf",
                                  mime="application/pdf")
        except Exception as e:
            st.error(f"Report failed: {e}")


# ---------- tab: plan -----------------------------------------------------

def tab_plan(state: dict) -> None:
    st.subheader("THE FATWAD PLAN",
                  help="Where to put your monthly surplus, based on your risk profile and "
                       "current allocation.")

    try:
        plan = _api_get("/plan")
    except Exception as e:
        st.error(f"Could not load plan: {e}")
        return

    profile = plan.get("risk_profile")
    questions = plan.get("risk_questions") or []

    with st.expander("RISK PROFILE — 5-question stress test",
                       expanded=profile is None):
        st.caption("Aggressive / Balanced / Defensive. Drives the suggested asset mix.")
        with st.form("risk_form"):
            scores: list[int] = []
            for idx, q in enumerate(questions):
                choice = st.radio(q["q"],
                                    [opt["label"] for opt in q["options"]],
                                    key=f"rq_{idx}")
                scores.append(next(opt["score"] for opt in q["options"]
                                     if opt["label"] == choice))
            if st.form_submit_button("SCORE ME"):
                try:
                    res = _api_post("/plan/score", {"answers": scores})
                    st.success(f"Profile: {res['profile']} (score {res['score']})")
                    st.rerun()
                except Exception as e:
                    st.error(f"Scoring failed: {e}")

    if not profile:
        st.info("Take the stress test to unlock the buy list.")
        return

    st.markdown(f"**Profile:** <span class='pill'>{profile}</span>",
                  unsafe_allow_html=True)
    st.metric("THIS MONTH'S SURPLUS", money(plan.get("monthly_surplus", 0)))

    st.markdown("### CAPITAL DEPLOYMENT — BUY LIST",
                 help="Suggested split of this month's surplus across asset classes.")
    bl = plan.get("buy_list") or {}
    if not bl:
        st.warning("No surplus to deploy. Cut burn or raise inflow.")
    else:
        cols = st.columns(len(bl))
        for (k, v), col in zip(bl.items(), cols):
            col.metric(k.upper(), money(v),
                        help=f"Suggested {money(v)} into {k} this month.")

    st.markdown("### REBALANCING SIGNALS",
                 help="TRIM/ADD warnings when an asset class drifts more than 10% from target.")
    sigs = plan.get("rebalance_signals") or []
    if not sigs:
        st.write("Allocation within tolerance. Hold.")
    else:
        for sig in sigs:
            st.error(sig)


def _required_monthly(target: float, current: float, ar: float, years: int) -> float:
    if years <= 0:
        return max(0.0, target - current)
    n = years * 12
    r = ar / 12
    fv = current * (1 + r) ** n
    remaining = target - fv
    if remaining <= 0:
        return 0.0
    if r == 0:
        return remaining / n
    return remaining * r / ((1 + r) ** n - 1)


def _required_return(target: float, current: float, monthly: float, years: int) -> float | None:
    if years <= 0 or current <= 0:
        return None
    lo, hi = -0.20, 0.50
    for _ in range(80):
        mid = (lo + hi) / 2
        n = years * 12
        r = mid / 12
        fv = current * (1 + r) ** n + (
            monthly * (((1 + r) ** n - 1) / r) if r else monthly * n
        )
        if fv < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------- math helpers (mirrors src/tools/forecast for display only) ----

def _cashflow_summary(state: dict) -> dict:
    inflow = sum(float(i.get("monthly", 0) or 0) for i in state["income"])
    fixed = sum(float(f.get("monthly", 0) or 0) for f in state["fixed_outflow"])
    if state["variable_burn"]:
        avg_weekly = sum(float(v.get("amount", 0) or 0)
                          for v in state["variable_burn"]) / len(state["variable_burn"])
        variable = avg_weekly * 52 / 12
    else:
        variable = 0.0
    return {"inflow": inflow, "fixed": fixed, "variable": variable,
            "surplus": inflow - fixed - variable}


def _annual_yield(state: dict) -> float:
    eq = float(state["settings"].get("expected_return", 0.07))
    cash = float(state["settings"].get("savings_apy", 0.045))
    total = 0.0
    for a in state["assets"]:
        v = float(a.get("current_value", 0) or 0)
        if a["type"] == "Equity":
            total += v * eq
        elif a["type"] == "Real Estate":
            total += v * 0.05
        elif a["type"] == "Cash":
            total += v * cash
    return total


def _project(start: float, monthly: float, ar: float, years: int) -> list[float]:
    rate = ar / 12.0
    out = [start]
    bal = start
    for m in range(1, years * 12 + 1):
        bal = bal * (1 + rate) + monthly
        if m % 12 == 0:
            out.append(bal)
    return out


# ---------- main ---------------------------------------------------------

def main() -> None:
    try:
        state = load_state()
    except Exception as e:
        st.error(f"Could not reach API at `{API_URL}`. Start with `docker compose up -d`.\n\n`{e}`")
        st.stop()

    inject_css(state["settings"].get("ghost_mode", False))
    render_sidebar(state)

    st.title("FATWAD")
    st.caption("Manual wealth simulator. No auto-sync. No fluff.")

    tabs = st.tabs(["DASHBOARD", "ASSETS", "CASHFLOW", "PLAN",
                     "TIME MACHINE", "MAIN CHARACTER"])
    with tabs[0]: tab_dashboard(state)
    with tabs[1]: tab_assets(state)
    with tabs[2]: tab_cashflow(state)
    with tabs[3]: tab_plan(state)
    with tabs[4]: tab_time_machine(state)
    with tabs[5]: tab_main_character(state)

    # Floating CHAT FAB.
    #  1. Render the real Streamlit button. The CSS in inject_css() positions
    #     it off-screen but keeps it clickable, so React's onClick still wires.
    #  2. _inject_chat_fab() creates a clone <button class="fatwad-fab"> in
    #     document.body and forwards clicks to the hidden real button. The
    #     clone has no transformed ancestors so position:fixed pins reliably.
    clicked = st.button(
        "💬  CHAT",
        key="chat_fab_button",
        help="Ask questions or issue commands like 'add 1.5 BTC at ₱4.5M' from any tab.",
    )
    _inject_chat_fab()
    if clicked or st.session_state.get("_chat_open"):
        chat_dialog()


if __name__ == "__main__":
    main()
else:
    main()
