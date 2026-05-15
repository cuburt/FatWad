"""Tests for /plan and /plan/score endpoints."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Share the same SQLite-on-disk DB module test_api uses so we don't fight over
# DATABASE_URL set at import time.
if "src.db.session" not in sys.modules:
    _db_path = Path(tempfile.mkdtemp(prefix="fatwad-plan-")) / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"

from fastapi.testclient import TestClient  # noqa: E402

try:
    from src.db.session import init_db
    from src.main import app
    init_db()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"DB unavailable: {exc}", allow_module_level=True)

client = TestClient(app)


def test_get_plan_returns_questions_and_targets():
    r = client.get("/plan")
    assert r.status_code == 200
    body = r.json()
    assert len(body["risk_questions"]) == 5
    assert set(body["allocation_targets"].keys()) == {"Aggressive", "Balanced", "Defensive"}
    # Empty profile before user takes the test.
    assert body["risk_profile"] in (None, "Balanced", "Aggressive", "Defensive")


def test_score_persists_profile():
    # All max-score answers should map to Aggressive.
    r = client.post("/plan/score", json={"answers": [3, 3, 3, 3, 3]})
    assert r.status_code == 200
    assert r.json()["profile"] == "Aggressive"

    # GET /plan should now reflect the saved profile.
    r2 = client.get("/plan")
    assert r2.json()["risk_profile"] == "Aggressive"


def test_score_low_picks_defensive():
    r = client.post("/plan/score", json={"answers": [0, 0, 0, 0, 0]})
    assert r.json()["profile"] == "Defensive"


def test_plan_includes_buy_list_when_profile_set():
    # Need a profile + a surplus. Score the profile, add income, fixed below it.
    client.post("/plan/score", json={"answers": [3, 2, 2, 2, 3]})  # Aggressive
    client.post("/income", json={"source": "Salary", "monthly": 100_000})
    client.post("/fixed", json={"bill": "Rent", "monthly": 30_000})

    body = client.get("/plan").json()
    assert body["monthly_surplus"] > 0
    # buy_list keys should match the ALLOCATION dict for the profile.
    assert set(body["buy_list"].keys()) >= {"Equity", "Cash"}
    total = sum(body["buy_list"].values())
    assert abs(total - body["monthly_surplus"]) < 0.5  # rounding only
