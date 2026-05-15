"""Engine + init_db. Creates the pgvector extension if missing, then runs
SQLModel.metadata.create_all. Idempotent — safe to call on every boot."""

import os

import structlog
from dotenv import load_dotenv
from sqlmodel import Session, SQLModel, create_engine

import src.db.models  # noqa: F401  registers tables with SQLModel metadata

load_dotenv()

logger = structlog.get_logger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://fatwad:fatwad@localhost:5432/fatwad",
)

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=1800,
)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)
