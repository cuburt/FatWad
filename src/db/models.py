"""SQLModel tables. Single-user — no tenant column. No vector column either:
the agent gets context from the user's snapshot + recent message history,
not from a vector store."""

from datetime import date, datetime
from typing import Any, List, Optional

from sqlalchemy import JSON, Column, Index
from sqlmodel import Field, SQLModel

ASSET_TYPES = ["Cash", "Real Estate", "Equity", "Crypto", "Physical"]


class User(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Settings(SQLModel, table=True):
    user_id: str = Field(primary_key=True, foreign_key="user.id")
    expected_return: float = 0.07
    savings_apy: float = 0.045
    speculative_return: float = 0.15
    ghost_mode: bool = False
    llm_model: str = "openrouter/openai/gpt-4o-mini"
    risk_profile: Optional[str] = None
    goal_target: float = 1_000_000
    goal_current_age: int = 30
    goal_target_age: int = 50
    last_checkin: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Asset(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    name: str
    type: str
    purchase_price: float = 0.0
    current_value: float = 0.0
    ai_unit: Optional[str] = None
    ai_note: Optional[str] = None
    ai_sources: List[Any] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class IncomeStream(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    source: str
    monthly: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FixedOutflow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    bill: str
    monthly: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VariableBurn(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    week: date
    amount: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Scenario(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    name: str
    delta_nw: float = 0.0
    delta_surplus: float = 0.0
    delta_return: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Session(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="session.id", index=True)
    role: str  # user | assistant | system | tool
    content: str
    tool_calls: Optional[List[Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)


Index("message_session_created_idx", Message.session_id, Message.created_at)
