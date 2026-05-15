"""CRUD over assets, cashflow streams, scenarios, settings + a /state aggregate.
Single user — user_id always pulled from FATWAD_USER_ID env."""

import os
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from sqlmodel import Session, select

from src.api.schemas import (
    AssetIn,
    AssetOut,
    FixedIn,
    FixedOut,
    IncomeIn,
    IncomeOut,
    ScenarioIn,
    ScenarioOut,
    SettingsOut,
    SettingsPayload,
    VariableIn,
    VariableOut,
)
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
from src.reports.pdf import build_report
from src.tools.snapshot import get_state

router = APIRouter()
DEFAULT_USER = os.getenv("FATWAD_USER_ID", "user_1")


def _ensure_user(session: Session) -> str:
    if not session.exec(select(User).where(User.id == DEFAULT_USER)).first():
        session.add(User(id=DEFAULT_USER, name=DEFAULT_USER))
        session.commit()
    return DEFAULT_USER


# ---- aggregate snapshot ---------------------------------------------------

@router.get("/state")
def state() -> dict[str, Any]:
    return get_state(DEFAULT_USER)


@router.get("/report")
def report():
    pdf = build_report(get_state(DEFAULT_USER))
    return Response(content=pdf, media_type="application/pdf",
                     headers={"Content-Disposition": 'attachment; filename="fatwad_report.pdf"'})


# ---- assets ---------------------------------------------------------------

@router.get("/assets", response_model=list[AssetOut])
def list_assets():
    with Session(engine) as s:
        _ensure_user(s)
        rows = s.exec(select(Asset).where(Asset.user_id == DEFAULT_USER).order_by(Asset.created_at)).all()
        return [AssetOut(id=r.id, name=r.name, type=r.type, purchase_price=r.purchase_price,
                          current_value=r.current_value, ai_unit=r.ai_unit, ai_note=r.ai_note,
                          ai_sources=r.ai_sources or []) for r in rows]


@router.post("/assets", response_model=AssetOut)
def create_asset(payload: AssetIn):
    with Session(engine) as s:
        _ensure_user(s)
        row = Asset(user_id=DEFAULT_USER, **payload.model_dump())
        s.add(row); s.commit(); s.refresh(row)
        return AssetOut(id=row.id, **payload.model_dump())


@router.patch("/assets/{asset_id}", response_model=AssetOut)
def update_asset(asset_id: int, payload: AssetIn):
    with Session(engine) as s:
        row = s.get(Asset, asset_id)
        if not row:
            raise HTTPException(404)
        for k, v in payload.model_dump().items():
            setattr(row, k, v)
        row.updated_at = datetime.utcnow()
        s.add(row); s.commit(); s.refresh(row)
        return AssetOut(id=row.id, **payload.model_dump())


@router.delete("/assets/{asset_id}")
def delete_asset(asset_id: int):
    with Session(engine) as s:
        row = s.get(Asset, asset_id)
        if not row:
            raise HTTPException(404)
        s.delete(row); s.commit()
        return {"ok": True}


# ---- income / fixed / variable -------------------------------------------

@router.get("/income", response_model=list[IncomeOut])
def list_income():
    with Session(engine) as s:
        _ensure_user(s)
        rows = s.exec(select(IncomeStream).where(IncomeStream.user_id == DEFAULT_USER)
                       .order_by(IncomeStream.created_at)).all()
        return [IncomeOut(id=r.id, source=r.source, monthly=r.monthly) for r in rows]


@router.post("/income", response_model=IncomeOut)
def create_income(payload: IncomeIn):
    with Session(engine) as s:
        _ensure_user(s)
        row = IncomeStream(user_id=DEFAULT_USER, **payload.model_dump())
        s.add(row); s.commit(); s.refresh(row)
        return IncomeOut(id=row.id, **payload.model_dump())


@router.delete("/income/{row_id}")
def delete_income(row_id: int):
    with Session(engine) as s:
        row = s.get(IncomeStream, row_id)
        if row:
            s.delete(row); s.commit()
        return {"ok": True}


@router.get("/fixed", response_model=list[FixedOut])
def list_fixed():
    with Session(engine) as s:
        _ensure_user(s)
        rows = s.exec(select(FixedOutflow).where(FixedOutflow.user_id == DEFAULT_USER)
                       .order_by(FixedOutflow.created_at)).all()
        return [FixedOut(id=r.id, bill=r.bill, monthly=r.monthly) for r in rows]


@router.post("/fixed", response_model=FixedOut)
def create_fixed(payload: FixedIn):
    with Session(engine) as s:
        _ensure_user(s)
        row = FixedOutflow(user_id=DEFAULT_USER, **payload.model_dump())
        s.add(row); s.commit(); s.refresh(row)
        return FixedOut(id=row.id, **payload.model_dump())


@router.delete("/fixed/{row_id}")
def delete_fixed(row_id: int):
    with Session(engine) as s:
        row = s.get(FixedOutflow, row_id)
        if row:
            s.delete(row); s.commit()
        return {"ok": True}


@router.get("/variable", response_model=list[VariableOut])
def list_variable():
    with Session(engine) as s:
        _ensure_user(s)
        rows = s.exec(select(VariableBurn).where(VariableBurn.user_id == DEFAULT_USER)
                       .order_by(VariableBurn.week.desc())).all()
        return [VariableOut(id=r.id, week=r.week, amount=r.amount) for r in rows]


@router.post("/variable", response_model=VariableOut)
def create_variable(payload: VariableIn):
    with Session(engine) as s:
        _ensure_user(s)
        row = VariableBurn(user_id=DEFAULT_USER, **payload.model_dump())
        s.add(row); s.commit(); s.refresh(row)
        return VariableOut(id=row.id, **payload.model_dump())


@router.delete("/variable/{row_id}")
def delete_variable(row_id: int):
    with Session(engine) as s:
        row = s.get(VariableBurn, row_id)
        if row:
            s.delete(row); s.commit()
        return {"ok": True}


# ---- scenarios ------------------------------------------------------------

@router.get("/scenarios", response_model=list[ScenarioOut])
def list_scenarios():
    with Session(engine) as s:
        _ensure_user(s)
        rows = s.exec(select(Scenario).where(Scenario.user_id == DEFAULT_USER)
                       .order_by(Scenario.created_at)).all()
        return [ScenarioOut(id=r.id, **{k: getattr(r, k) for k in
                 ("name", "delta_nw", "delta_surplus", "delta_return")}) for r in rows]


@router.post("/scenarios", response_model=ScenarioOut)
def create_scenario(payload: ScenarioIn):
    with Session(engine) as s:
        _ensure_user(s)
        row = Scenario(user_id=DEFAULT_USER, **payload.model_dump())
        s.add(row); s.commit(); s.refresh(row)
        return ScenarioOut(id=row.id, **payload.model_dump())


@router.delete("/scenarios/{row_id}")
def delete_scenario(row_id: int):
    with Session(engine) as s:
        row = s.get(Scenario, row_id)
        if row:
            s.delete(row); s.commit()
        return {"ok": True}


# ---- settings -------------------------------------------------------------

@router.get("/settings", response_model=SettingsOut)
def get_settings():
    with Session(engine) as s:
        _ensure_user(s)
        row = s.exec(select(Settings).where(Settings.user_id == DEFAULT_USER)).first()
        if row is None:
            row = Settings(user_id=DEFAULT_USER)
            s.add(row); s.commit(); s.refresh(row)
        return _settings_out(row)


@router.patch("/settings", response_model=SettingsOut)
def patch_settings(payload: SettingsPayload):
    with Session(engine) as s:
        _ensure_user(s)
        row = s.exec(select(Settings).where(Settings.user_id == DEFAULT_USER)).first()
        if row is None:
            row = Settings(user_id=DEFAULT_USER)
            s.add(row); s.commit(); s.refresh(row)
        for k, v in payload.model_dump(exclude_unset=True).items():
            setattr(row, k, v)
        row.updated_at = datetime.utcnow()
        s.add(row); s.commit(); s.refresh(row)
        return _settings_out(row)


@router.post("/settings/checkin", response_model=SettingsOut)
def stamp_checkin():
    with Session(engine) as s:
        _ensure_user(s)
        row = s.exec(select(Settings).where(Settings.user_id == DEFAULT_USER)).first()
        if row is None:
            row = Settings(user_id=DEFAULT_USER)
        row.last_checkin = datetime.utcnow()
        s.add(row); s.commit(); s.refresh(row)
        return _settings_out(row)


def _settings_out(row: Settings) -> SettingsOut:
    return SettingsOut(
        expected_return=row.expected_return, savings_apy=row.savings_apy,
        speculative_return=row.speculative_return, ghost_mode=row.ghost_mode,
        llm_model=row.llm_model, risk_profile=row.risk_profile,
        goal_target=row.goal_target, goal_current_age=row.goal_current_age,
        goal_target_age=row.goal_target_age, last_checkin=row.last_checkin,
    )
