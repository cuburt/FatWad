"""The two graph-invoking endpoints: /ask (read-only) and /agent (full multi-tool).
Both share the same plumbing — session resolution, message persistence, PII
scrubbing on the answer."""

import json
import os
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from src.agents import llm as _llm
from src.agents.graph import agent_app, ask_app
from src.agents.nodes import planner_node, tool_node
from src.agents.prompts import SUMMARIZER_PROMPT
from src.api.schemas import AskRequest, AskResponse
from src.api.sessions_db import add_message, get_or_create_session, list_messages
from src.safety import redact_pii_text
from src.tools.snapshot import get_state, state_to_summary

logger = structlog.get_logger(__name__)
router = APIRouter()

DEFAULT_USER = os.getenv("FATWAD_USER_ID", "user_1")


def _run(req: AskRequest, app) -> AskResponse:
    user_id = req.user_id or DEFAULT_USER

    if req.persist_history:
        session = get_or_create_session(req.session_id, user_id)
        add_message(session.id, "user", req.query)
        # Replay history so multi-turn context survives across requests.
        history = list_messages(session.id)
        msgs = [HumanMessage(content=m.content) for m in history if m.role == "user"]
        if not msgs:
            msgs = [HumanMessage(content=req.query)]
        session_id_for_response = session.id
    else:
        # One-shot automation call: don't touch the chat sessions table.
        msgs = [HumanMessage(content=req.query)]
        session_id_for_response = req.session_id or ""

    turn_input: dict[str, Any] = {
        "messages": msgs,
        "user_id": user_id,
        "session_id": session_id_for_response,
        "tool_results": [],
        "citations": [],
        "scratchpad": "",
    }
    try:
        result = app.invoke(turn_input)
    except Exception as e:
        logger.error("graph_invoke_failed", error=str(e))
        raise HTTPException(500, detail=f"Agent failed: {e}")

    answer_msgs = [m for m in result.get("messages", []) if getattr(m, "type", None) == "ai"]
    if not answer_msgs:
        raise HTTPException(500, detail="Agent returned no answer.")
    answer = redact_pii_text(answer_msgs[-1].content or "")
    citations = list(result.get("citations") or [])
    tool_results = list(result.get("tool_results") or [])

    if req.persist_history:
        add_message(session_id_for_response, "assistant", answer)
    return AskResponse(answer=answer, session_id=session_id_for_response,
                        citations=citations, tool_results=tool_results)


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return _run(req, ask_app)


@router.post("/agent", response_model=AskResponse)
def agent(req: AskRequest) -> AskResponse:
    return _run(req, agent_app)


def _build_streaming_state(req: AskRequest, with_planner: bool) -> tuple[dict, str]:
    """Run the non-streaming part of the graph by hand so we can stream the
    summarize step ourselves. Returns (state_dict, session_id_for_response)."""
    user_id = req.user_id or DEFAULT_USER

    if req.persist_history:
        session = get_or_create_session(req.session_id, user_id)
        add_message(session.id, "user", req.query)
        session_id_for_response = session.id
    else:
        session_id_for_response = req.session_id or ""

    snapshot = get_state(user_id)
    snapshot_text = state_to_summary(snapshot)
    state: dict[str, Any] = {
        "messages": [HumanMessage(content=req.query)],
        "user_id": user_id,
        "session_id": session_id_for_response,
        "snapshot": snapshot,
        "snapshot_text": snapshot_text,
        "tool_results": [],
        "citations": [],
        "scratchpad": "",
    }
    if with_planner:
        state.update(planner_node(state))
        if state.get("next_step") and state["next_step"] != "summarize":
            state.update(tool_node(state))
    return state, session_id_for_response


def _sse_stream(req: AskRequest, with_planner: bool):
    """SSE generator: emits {token: "..."} per chunk, then a final
    {done: true, session_id, citations, tool_results} event."""
    try:
        state, session_id = _build_streaming_state(req, with_planner=with_planner)
    except Exception as e:
        logger.error("stream_setup_failed", error=str(e))
        yield f"data: {json.dumps({'error': f'Setup failed: {e}'})}\n\n"
        return

    sys = SUMMARIZER_PROMPT.format(
        snapshot_text=state.get("snapshot_text", ""),
        tool_results=json.dumps(state.get("tool_results") or [], default=str)[:6000],
    )
    buf: list[str] = []
    try:
        for chunk in _llm.chat_stream(
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": req.query},
            ],
            tier=_llm.SYNTHESIS,
            temperature=0.3,
        ):
            buf.append(chunk)
            yield f"data: {json.dumps({'token': chunk})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return

    answer = redact_pii_text("".join(buf))
    if req.persist_history and session_id:
        try:
            add_message(session_id, "assistant", answer)
        except Exception as e:
            logger.warning("stream_persist_failed", error=str(e))

    yield "data: " + json.dumps({
        "done": True,
        "session_id": session_id,
        "citations": list(state.get("citations") or []),
        "tool_results": list(state.get("tool_results") or []),
    }) + "\n\n"


@router.post("/ask/stream")
def ask_stream(req: AskRequest):
    """SSE stream of the /ask flow: safety -> snapshot -> stream(summarize).
    No planner, no tools — fast conversational answer."""
    return StreamingResponse(_sse_stream(req, with_planner=False),
                              media_type="text/event-stream")


@router.post("/agent/stream")
def agent_stream(req: AskRequest):
    """SSE stream of the /agent flow: snapshot -> planner -> tool -> stream(summarize)."""
    return StreamingResponse(_sse_stream(req, with_planner=True),
                              media_type="text/event-stream")
