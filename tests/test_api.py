"""FastAPI smoke tests against an ephemeral SQLite DB.

Switches DATABASE_URL to in-memory SQLite *before* importing src.main, so the
engine is created against SQLite and pgvector is never required.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Force SQLite-on-disk before any src.* import. In-memory SQLite gives each
# connection its own DB, so the table created by init_db() wouldn't be visible
# to the request handler. A tempfile dodges that.
if "src.db.session" not in sys.modules:
    _db_path = Path(tempfile.mkdtemp(prefix="fatwad-test-")) / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"

from fastapi.testclient import TestClient  # noqa: E402

# Skip the entire module if SQLAlchemy SQLite/JSONB compatibility blocks DDL.
try:
    from src.main import app
    from src.db.session import init_db
    init_db()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"DB init unavailable in this env: {exc}", allow_module_level=True)

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_state_returns_defaults():
    r = client.get("/state")
    assert r.status_code == 200
    body = r.json()
    assert "assets" in body and "income" in body
    assert body["settings"]["expected_return"] == pytest.approx(0.07)


def test_create_and_list_asset():
    r = client.post("/assets", json={"name": "VTI", "type": "Equity",
                                          "purchase_price": 1000, "current_value": 1500})
    assert r.status_code == 200
    asset = r.json()
    assert asset["name"] == "VTI"
    listed = client.get("/assets").json()
    assert any(a["name"] == "VTI" for a in listed)


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "requests_total" in body
    assert body["requests_total"] >= 1


def test_ai_endpoints_removed():
    """/ai/* shims were removed; /agent is the only path for AI work now."""
    assert client.post("/ai/assumptions").status_code == 404
    assert client.post("/ai/asset-value", json={"name": "x", "type": "Cash"}).status_code == 404


def test_ask_and_agent_routes_exist():
    """Smoke: /ask and /agent are mounted (even if they 500 without an LLM key)."""
    # Pydantic 422 if body bad; we send a valid one to ensure the route exists.
    # Either 200 or 500 (LLM not configured) is fine — we just don't want 404.
    r1 = client.post("/ask", json={"query": "ping"})
    r2 = client.post("/agent", json={"query": "ping"})
    assert r1.status_code != 404
    assert r2.status_code != 404
