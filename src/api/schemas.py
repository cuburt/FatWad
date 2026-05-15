"""Pydantic models shared by the HTTP routers."""

from datetime import date, datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    # Function-automation calls (sidebar AI buttons) set this False so the
    # one-shot prompt + answer doesn't get persisted as a chat session.
    persist_history: bool = True


class AskResponse(BaseModel):
    answer: str
    session_id: str
    citations: List[str] = []
    # Raw output from any tool the agent invoked this turn. The frontend's
    # automation buttons read this back to patch settings / add assets.
    tool_results: List[Any] = Field(default_factory=list)


class AssetIn(BaseModel):
    name: str
    type: str
    purchase_price: float = 0.0
    current_value: float = 0.0
    ai_unit: Optional[str] = None
    ai_note: Optional[str] = None
    ai_sources: List[Any] = Field(default_factory=list)


class AssetOut(AssetIn):
    id: int


class IncomeIn(BaseModel):
    source: str
    monthly: float = 0.0


class IncomeOut(IncomeIn):
    id: int


class FixedIn(BaseModel):
    bill: str
    monthly: float = 0.0


class FixedOut(FixedIn):
    id: int


class VariableIn(BaseModel):
    week: date
    amount: float = 0.0


class VariableOut(VariableIn):
    id: int


class ScenarioIn(BaseModel):
    name: str
    delta_nw: float = 0.0
    delta_surplus: float = 0.0
    delta_return: float = 0.0


class ScenarioOut(ScenarioIn):
    id: int


class SettingsPayload(BaseModel):
    expected_return: Optional[float] = None
    savings_apy: Optional[float] = None
    speculative_return: Optional[float] = None
    ghost_mode: Optional[bool] = None
    llm_model: Optional[str] = None
    risk_profile: Optional[str] = None
    goal_target: Optional[float] = None
    goal_current_age: Optional[int] = None
    goal_target_age: Optional[int] = None


class SettingsOut(BaseModel):
    expected_return: float
    savings_apy: float
    speculative_return: float
    ghost_mode: bool
    llm_model: str
    risk_profile: Optional[str] = None
    goal_target: float
    goal_current_age: int
    goal_target_age: int
    last_checkin: Optional[datetime] = None


class SessionOut(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    created_at: datetime
