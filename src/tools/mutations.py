"""Write tools the agent can call to mutate the user's wealth state.

Each function maps to one repository-style operation and returns a small dict
the summarizer can mention back to the user ("Added VTI", "Set goal to ₱5M").
All writes go through SQLModel so they show up in /state on the next fetch
and the tabs refresh.
"""

import os
from datetime import date as _date
from datetime import datetime
from typing import Any

from sqlmodel import Session, select

from src.db.models import (
    Asset,
    FixedOutflow,
    IncomeStream,
    Scenario,
    Settings,
    User,
    VariableBurn,
)
from src.db.session import engine

DEFAULT_USER = os.getenv("FATWAD_USER_ID", "user_1")

_SETTING_FIELDS = {
    "expected_return", "savings_apy", "speculative_return",
    "ghost_mode", "llm_model", "risk_profile",
    "goal_target", "goal_current_age", "goal_target_age",
}

ASSET_TYPES = {"Cash", "Real Estate", "Equity", "Crypto", "Physical"}


def _ensure_user(session: Session) -> None:
    if not session.exec(select(User).where(User.id == DEFAULT_USER)).first():
        session.add(User(id=DEFAULT_USER, name=DEFAULT_USER))
        session.commit()


def add_asset(name: str, type: str, current_value: float,
              purchase_price: float = 0.0, ai_note: str | None = None) -> dict:
    if type not in ASSET_TYPES:
        return {"ok": False, "reason": f"Unknown asset type: {type}. "
                                          f"Allowed: {sorted(ASSET_TYPES)}"}
    with Session(engine) as s:
        _ensure_user(s)
        row = Asset(
            user_id=DEFAULT_USER, name=name, type=type,
            current_value=float(current_value),
            purchase_price=float(purchase_price),
            ai_note=ai_note,
        )
        s.add(row); s.commit(); s.refresh(row)
        return {"ok": True, "asset_id": row.id, "name": row.name,
                 "current_value": float(row.current_value)}


def add_income_stream(source: str, monthly: float) -> dict:
    with Session(engine) as s:
        _ensure_user(s)
        row = IncomeStream(user_id=DEFAULT_USER, source=source, monthly=float(monthly))
        s.add(row); s.commit(); s.refresh(row)
        return {"ok": True, "income_id": row.id, "source": row.source,
                 "monthly": float(row.monthly)}


def add_fixed_outflow(bill: str, monthly: float) -> dict:
    with Session(engine) as s:
        _ensure_user(s)
        row = FixedOutflow(user_id=DEFAULT_USER, bill=bill, monthly=float(monthly))
        s.add(row); s.commit(); s.refresh(row)
        return {"ok": True, "outflow_id": row.id, "bill": row.bill,
                 "monthly": float(row.monthly)}


def add_variable_burn(week_iso: str, amount: float) -> dict:
    try:
        wk = _date.fromisoformat(week_iso)
    except ValueError as e:
        return {"ok": False, "reason": f"Bad week_iso ({week_iso}): {e}"}
    with Session(engine) as s:
        _ensure_user(s)
        row = VariableBurn(user_id=DEFAULT_USER, week=wk, amount=float(amount))
        s.add(row); s.commit(); s.refresh(row)
        return {"ok": True, "burn_id": row.id, "week": wk.isoformat(),
                 "amount": float(row.amount)}


def add_scenario(name: str, delta_nw: float = 0.0, delta_surplus: float = 0.0,
                 delta_return: float = 0.0) -> dict:
    with Session(engine) as s:
        _ensure_user(s)
        row = Scenario(
            user_id=DEFAULT_USER, name=name,
            delta_nw=float(delta_nw),
            delta_surplus=float(delta_surplus),
            delta_return=float(delta_return),
        )
        s.add(row); s.commit(); s.refresh(row)
        return {"ok": True, "scenario_id": row.id, "name": row.name}


def update_settings(**fields: Any) -> dict:
    fields = {k: v for k, v in fields.items() if k in _SETTING_FIELDS and v is not None}
    if not fields:
        return {"ok": False, "reason": "No valid setting fields supplied. "
                                          f"Allowed: {sorted(_SETTING_FIELDS)}"}
    with Session(engine) as s:
        _ensure_user(s)
        row = s.exec(select(Settings).where(Settings.user_id == DEFAULT_USER)).first()
        if row is None:
            row = Settings(user_id=DEFAULT_USER)
        for k, v in fields.items():
            setattr(row, k, v)
        row.updated_at = datetime.utcnow()
        s.add(row); s.commit()
        return {"ok": True, "updated": list(fields.keys())}
