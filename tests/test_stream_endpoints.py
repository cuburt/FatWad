"""SSE streaming tests for /ask/stream and /agent/stream.

Patches the LLM client so no network is required and the planner is stubbed
to keep the /agent/stream path deterministic.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

if "src.db.session" not in sys.modules:
    _db_path = Path(tempfile.mkdtemp(prefix="fatwad-stream-")) / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"

from fastapi.testclient import TestClient  # noqa: E402

try:
    from src.db.session import init_db
    from src.main import app
    init_db()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"DB unavailable: {exc}", allow_module_level=True)

client = TestClient(app)


def _parse_sse(body: str) -> list[dict]:
    events: list[dict] = []
    for line in body.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[len("data: "):]))
            except json.JSONDecodeError:
                pass
    return events


def _fake_stream(*_a, **_kw):
    """Yield three tokens; mimics chat_stream's contract."""
    for tok in ["Hello", " ", "world"]:
        yield tok


def test_ask_stream_emits_tokens_then_done():
    with patch("src.api.agent_routes._llm.chat_stream", side_effect=_fake_stream):
        with client.stream("POST", "/ask/stream",
                            json={"query": "hi", "persist_history": False}) as r:
            assert r.status_code == 200
            body = "".join(chunk for chunk in r.iter_text())

    events = _parse_sse(body)
    tokens = [e["token"] for e in events if "token" in e]
    assert tokens == ["Hello", " ", "world"]
    done = [e for e in events if e.get("done")]
    assert len(done) == 1


def test_agent_stream_runs_planner_then_streams():
    # Force the planner toward 'summarize' so tool_node isn't invoked
    # (avoids needing market.py / chat for that path).
    from src.agents import nodes as _nodes

    def _stub_planner(state):
        return {"next_step": "summarize"}

    with patch.object(_nodes, "planner_node", side_effect=_stub_planner), \
            patch("src.api.agent_routes.planner_node", side_effect=_stub_planner), \
            patch("src.api.agent_routes._llm.chat_stream", side_effect=_fake_stream):
        with client.stream("POST", "/agent/stream",
                            json={"query": "hi", "persist_history": False}) as r:
            assert r.status_code == 200
            body = "".join(chunk for chunk in r.iter_text())

    events = _parse_sse(body)
    assert any("token" in e for e in events)
    assert any(e.get("done") for e in events)


def test_ask_stream_persists_assistant_message():
    """When persist_history=True, the final aggregated answer should land in
    the session message log."""
    with patch("src.api.agent_routes._llm.chat_stream", side_effect=_fake_stream):
        with client.stream("POST", "/ask/stream",
                            json={"query": "what's my position?"}) as r:
            body = "".join(chunk for chunk in r.iter_text())

    events = _parse_sse(body)
    done = next(e for e in events if e.get("done"))
    sid = done["session_id"]
    assert sid

    msgs = client.get(f"/sessions/{sid}/messages").json()
    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
    assert assistant_msgs
    assert assistant_msgs[-1]["content"] == "Hello world"
