"""Chat session listing + message history endpoints. The frontend uses these
to render the conversation list."""

import os

from fastapi import APIRouter

from src.api import sessions_db
from src.api.schemas import MessageOut, SessionOut

router = APIRouter()
DEFAULT_USER = os.getenv("FATWAD_USER_ID", "user_1")


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(user_id: str | None = None):
    rows = sessions_db.list_sessions(user_id or DEFAULT_USER)
    return [SessionOut(id=r.id, user_id=r.user_id, title=r.title,
                         created_at=r.created_at, updated_at=r.updated_at) for r in rows]


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
def list_messages(session_id: str):
    rows = sessions_db.list_messages(session_id)
    return [MessageOut(id=r.id, session_id=r.session_id, role=r.role,
                         content=r.content, created_at=r.created_at) for r in rows]


@router.delete("/sessions/{session_id}/messages")
def clear_messages(session_id: str):
    sessions_db.clear_session(session_id)
    return {"ok": True}
