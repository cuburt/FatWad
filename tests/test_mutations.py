"""Write-tool tests. Run against a fresh tempfile SQLite so each invocation
verifies the round-trip Tool call → DB write → /state read.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

if "src.db.session" not in sys.modules:
    _db_path = Path(tempfile.mkdtemp(prefix="fatwad-mut-")) / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"

from fastapi.testclient import TestClient  # noqa: E402

try:
    from src.db.session import init_db
    from src.main import app
    from src.tools import mutations
    from src.agents.nodes import _execute_tool
    init_db()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"DB unavailable: {exc}", allow_module_level=True)

client = TestClient(app)


def test_add_asset_writes_and_state_reflects_it():
    # Unique name so we don't collide with assets created by other test files
    # (they share the same SQLite file when run together).
    NAME = "MUT_TEST_VTI"
    initial = client.get("/state").json()
    initial_count = len(initial["assets"])

    result = mutations.add_asset(name=NAME, type="Equity",
                                    current_value=500_000, purchase_price=400_000)
    assert result["ok"] is True
    assert result["name"] == NAME

    after = client.get("/state").json()
    assert len(after["assets"]) == initial_count + 1
    new = next(a for a in after["assets"] if a["name"] == NAME)
    assert new["current_value"] == pytest.approx(500_000)
    assert new["purchase_price"] == pytest.approx(400_000)


def test_add_asset_rejects_unknown_type():
    out = mutations.add_asset(name="Junk", type="Bogus", current_value=1)
    assert out["ok"] is False
    assert "Unknown asset type" in out["reason"]


def test_update_settings_writes_subset_only():
    pre = client.get("/settings").json()
    out = mutations.update_settings(expected_return=0.08, goal_target=5_000_000)
    assert out["ok"] is True
    assert set(out["updated"]) == {"expected_return", "goal_target"}

    post = client.get("/settings").json()
    assert post["expected_return"] == pytest.approx(0.08)
    assert post["goal_target"] == pytest.approx(5_000_000)
    # Other fields preserved.
    assert post["savings_apy"] == pre["savings_apy"]


def test_update_settings_filters_unknown_fields():
    out = mutations.update_settings(not_a_field=1, expected_return=0.07)
    assert out["ok"] is True
    assert out["updated"] == ["expected_return"]


def test_update_settings_with_no_valid_fields_returns_no_op():
    out = mutations.update_settings(garbage=1, also_garbage=2)
    assert out["ok"] is False


def test_add_income_and_fixed_and_variable():
    inc = mutations.add_income_stream(source="Salary", monthly=120_000)
    fix = mutations.add_fixed_outflow(bill="Rent", monthly=30_000)
    var = mutations.add_variable_burn(week_iso="2026-05-04", amount=5_000)
    for r in (inc, fix, var):
        assert r["ok"] is True

    state = client.get("/state").json()
    assert any(i["source"] == "Salary" for i in state["income"])
    assert any(f["bill"] == "Rent" for f in state["fixed_outflow"])
    assert any(v["week"] == "2026-05-04" for v in state["variable_burn"])


def test_add_scenario():
    out = mutations.add_scenario(name="Quit job at 40",
                                    delta_surplus=-100_000, delta_return=-0.02)
    assert out["ok"] is True
    state = client.get("/state").json()
    assert any(s["name"] == "Quit job at 40" for s in state["scenarios"])


def test_execute_tool_dispatches_to_mutations():
    """The agent's _execute_tool hands off to mutations by name."""
    out = _execute_tool("add_asset",
                          {"name": "TEST_DISPATCH", "type": "Cash",
                           "current_value": 1000.0})
    assert out["ok"] is True
    assert out["name"] == "TEST_DISPATCH"
