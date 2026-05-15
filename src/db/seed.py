"""Seeds the default user + settings row. Idempotent — exits early if the
user already exists. Safe to run on every boot."""

import os

import structlog
from sqlmodel import Session, select

from src.db.models import Settings, User
from src.db.session import engine, init_db

logger = structlog.get_logger(__name__)

DEFAULT_USER_ID = os.getenv("FATWAD_USER_ID", "user_1")
DEFAULT_USER_NAME = os.getenv("FATWAD_USER_NAME", "me")


def main() -> None:
    init_db()
    with Session(engine) as session:
        if session.exec(select(User).where(User.id == DEFAULT_USER_ID)).first():
            logger.info("seed_skip", user_id=DEFAULT_USER_ID)
            return
        logger.info("seed_start", user_id=DEFAULT_USER_ID)
        # Insert User and Settings in two flushes. The models don't share a
        # SQLAlchemy relationship, so the unit-of-work won't otherwise infer
        # that User must be persisted before the FK-bearing Settings row.
        session.add(User(id=DEFAULT_USER_ID, name=DEFAULT_USER_NAME))
        session.commit()
        session.add(Settings(user_id=DEFAULT_USER_ID))
        session.commit()
        logger.info("seed_complete", user_id=DEFAULT_USER_ID)


if __name__ == "__main__":
    main()
