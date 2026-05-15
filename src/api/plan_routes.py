"""Plan endpoints: risk profile stress test, buy list, rebalance signals.

Deterministic — no LLM. The agent's `advice` path computes the same values, but
the frontend's PLAN tab needs them synchronously without an agent round-trip.
"""

import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sqlmodel import Session, select

from src.db.models import Settings, User
from src.db.session import engine
from src.tools.advice import (
    ALLOCATION,
    RISK_QUESTIONS,
    buy_list,
    rebalance_signals,
    score_to_profile,
)
from src.tools.forecast import cashflow
from src.tools.snapshot import get_state

router = APIRouter()
DEFAULT_USER = os.getenv("FATWAD_USER_ID", "user_1")


class RiskQuestionOption(BaseModel):
    label: str
    score: int


class RiskQuestion(BaseModel):
    q: str
    options: List[RiskQuestionOption]


class PlanResponse(BaseModel):
    risk_profile: Optional[str]
    monthly_surplus: float
    buy_list: dict[str, float]
    rebalance_signals: List[str]
    risk_questions: List[RiskQuestion]
    allocation_targets: dict[str, dict[str, float]]


class ScoreRequest(BaseModel):
    answers: List[int]


class ScoreResponse(BaseModel):
    profile: str
    score: int


def _questions() -> List[RiskQuestion]:
    return [
        RiskQuestion(
            q=q["q"],
            options=[RiskQuestionOption(label=l, score=s) for (l, s) in q["options"]],
        )
        for q in RISK_QUESTIONS
    ]


@router.get("/plan", response_model=PlanResponse)
def get_plan() -> PlanResponse:
    state = get_state(DEFAULT_USER)
    profile = state.get("risk_profile")
    cf = cashflow(state["income"], state["fixed_outflow"], state["variable_burn"])
    return PlanResponse(
        risk_profile=profile,
        monthly_surplus=round(cf.surplus, 2),
        buy_list=buy_list(max(0.0, cf.surplus), profile) if profile else {},
        rebalance_signals=rebalance_signals(state["assets"], profile) if profile else [],
        risk_questions=_questions(),
        allocation_targets=ALLOCATION,
    )


@router.post("/plan/score", response_model=ScoreResponse)
def score_risk(req: ScoreRequest) -> ScoreResponse:
    total = sum(req.answers)
    profile = score_to_profile(total)
    with Session(engine) as s:
        if not s.exec(select(User).where(User.id == DEFAULT_USER)).first():
            s.add(User(id=DEFAULT_USER, name=DEFAULT_USER))
            s.commit()
        row = s.exec(select(Settings).where(Settings.user_id == DEFAULT_USER)).first()
        if row is None:
            row = Settings(user_id=DEFAULT_USER)
        row.risk_profile = profile
        row.updated_at = datetime.utcnow()
        s.add(row); s.commit()
    return ScoreResponse(profile=profile, score=total)
