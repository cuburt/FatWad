"""Helpers for chat sessions + messages. Routes call these so route handlers
can stay thin."""

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlmodel import Session as SQLSession
from sqlmodel import select

from src.db.models import Message, Session, User
from src.db.session import engine


def get_or_create_session(session_id: Optional[str], user_id: str) -> Session:
    with SQLSession(engine) as s:
        # Make sure the user row exists so the FK doesn't fail.
        if not s.exec(select(User).where(User.id == user_id)).first():
            s.add(User(id=user_id, name=user_id))
            s.commit()

        if session_id:
            row = s.exec(select(Session).where(Session.id == session_id)).first()
            if row:
                return row
        new_id = session_id or str(uuid.uuid4())
        row = Session(id=new_id, user_id=user_id)
        s.add(row)
        s.commit()
        s.refresh(row)
        return row


def list_sessions(user_id: str, limit: int = 50) -> list[Session]:
    with SQLSession(engine) as s:
        return list(s.exec(
            select(Session).where(Session.user_id == user_id)
            .order_by(Session.updated_at.desc()).limit(limit)
        ).all())


def list_messages(session_id: str, limit: int = 200) -> list[Message]:
    with SQLSession(engine) as s:
        return list(s.exec(
            select(Message).where(Message.session_id == session_id)
            .order_by(Message.created_at).limit(limit)
        ).all())


def add_message(session_id: str, role: str, content: str,
                 tool_calls: Optional[list[Any]] = None) -> Message:
    with SQLSession(engine) as s:
        row = Message(session_id=session_id, role=role, content=content,
                       tool_calls=tool_calls)
        s.add(row)
        # bump session.updated_at so list_sessions sorts naturally.
        sess = s.exec(select(Session).where(Session.id == session_id)).first()
        if sess:
            sess.updated_at = datetime.utcnow()
        s.commit()
        s.refresh(row)
        return row


def clear_session(session_id: str) -> None:
    with SQLSession(engine) as s:
        for m in s.exec(select(Message).where(Message.session_id == session_id)).all():
            s.delete(m)
        s.commit()
