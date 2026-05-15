"""Reads the user's wealth snapshot out of Postgres into the legacy state dict
that the rest of the agent + reports work against."""

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
from src.tools.forecast import (
    annual_yield,
    by_type,
    cashflow,
    total_cost_basis,
    total_net_worth,
)


def get_state(user_id: str) -> dict:
    with Session(engine) as s:
        # Ensure the User row exists before inserting a Settings row that FKs to it.
        if not s.exec(select(User).where(User.id == user_id)).first():
            s.add(User(id=user_id, name=user_id))
            s.commit()
        settings = s.exec(select(Settings).where(Settings.user_id == user_id)).first()
        if settings is None:
            settings = Settings(user_id=user_id)
            s.add(settings)
            s.commit()
            s.refresh(settings)

        assets = [_asset_dict(a) for a in s.exec(
            select(Asset).where(Asset.user_id == user_id).order_by(Asset.created_at)
        ).all()]
        income = [{"id": r.id, "source": r.source, "monthly": float(r.monthly)} for r in s.exec(
            select(IncomeStream).where(IncomeStream.user_id == user_id).order_by(IncomeStream.created_at)
        ).all()]
        fixed = [{"id": r.id, "bill": r.bill, "monthly": float(r.monthly)} for r in s.exec(
            select(FixedOutflow).where(FixedOutflow.user_id == user_id).order_by(FixedOutflow.created_at)
        ).all()]
        variable = [{"id": r.id, "week": r.week.isoformat(), "amount": float(r.amount)} for r in s.exec(
            select(VariableBurn).where(VariableBurn.user_id == user_id).order_by(VariableBurn.week.desc())
        ).all()]
        scenarios = [{"id": r.id, "name": r.name, "delta_nw": float(r.delta_nw),
                       "delta_surplus": float(r.delta_surplus), "delta_return": float(r.delta_return)}
                       for r in s.exec(select(Scenario).where(Scenario.user_id == user_id)
                                          .order_by(Scenario.created_at)).all()]

        return {
            "assets": assets,
            "income": income,
            "fixed_outflow": fixed,
            "variable_burn": variable,
            "scenarios": scenarios,
            "risk_profile": settings.risk_profile,
            "goal": {
                "target": float(settings.goal_target),
                "current_age": int(settings.goal_current_age),
                "target_age": int(settings.goal_target_age),
            },
            "settings": {
                "expected_return": float(settings.expected_return),
                "savings_apy": float(settings.savings_apy),
                "speculative_return": float(settings.speculative_return),
                "ghost_mode": bool(settings.ghost_mode),
                "llm_model": settings.llm_model,
                "last_checkin": settings.last_checkin.isoformat() if settings.last_checkin else None,
            },
        }


def state_to_summary(state: dict) -> str:
    assets = state.get("assets", [])
    nw = total_net_worth(assets)
    cb = total_cost_basis(assets)
    cf = cashflow(state.get("income", []), state.get("fixed_outflow", []),
                    state.get("variable_burn", []))
    yr = annual_yield(assets, state.get("settings", {}))
    by_t = by_type(assets)
    lines = [
        "Currency: Philippine Peso (PHP, ₱). All amounts below are in pesos.",
        f"Net worth: ₱{nw:,.0f}",
        f"Cost basis: ₱{cb:,.0f}",
        f"Monthly inflow: ₱{cf.inflow:,.0f}",
        f"Monthly burn: ₱{cf.burn:,.0f}",
        f"Monthly surplus: ₱{cf.surplus:,.0f}",
        f"Estimated annual yield: ₱{yr:,.0f}",
        f"Risk profile: {state.get('risk_profile') or 'unset'}",
    ]
    nonzero = [(k, v) for k, v in by_t.items() if v]
    if nonzero:
        lines.append("Allocation: " + ", ".join(f"{k} ₱{v:,.0f}" for k, v in nonzero))
    settings = state.get("settings", {})
    lines.append(
        f"Assumptions: equity {settings.get('expected_return', 0):.1%}, "
        f"cash {settings.get('savings_apy', 0):.1%}, "
        f"speculative {settings.get('speculative_return', 0):.1%}"
    )
    return "\n".join(lines)


def _asset_dict(a: Asset) -> dict[str, Any]:
    return {
        "id": a.id,
        "name": a.name,
        "type": a.type,
        "purchase_price": float(a.purchase_price),
        "current_value": float(a.current_value),
        "ai_unit": a.ai_unit,
        "ai_note": a.ai_note,
        "ai_sources": a.ai_sources or [],
    }
