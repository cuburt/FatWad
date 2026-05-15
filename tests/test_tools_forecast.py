"""Forecast math tests. Pure functions; no DB."""

from src.tools.forecast import (
    annual_yield,
    asset_roi,
    cashflow,
    freedom_date,
    project_net_worth,
    required_monthly,
    total_net_worth,
)


def test_total_net_worth():
    assets = [{"current_value": 100}, {"current_value": 200}]
    assert total_net_worth(assets) == 300.0


def test_asset_roi_zero_basis():
    assert asset_roi({"purchase_price": 0, "current_value": 100}) == 0.0


def test_asset_roi_positive():
    assert asset_roi({"purchase_price": 100, "current_value": 150}) == 0.5


def test_cashflow_combines():
    cf = cashflow(
        income=[{"monthly": 10000}],
        fixed=[{"monthly": 2500}],
        variable=[{"amount": 600}],
    )
    assert cf.inflow == 10000
    assert cf.fixed == 2500
    assert cf.variable == pytest_approx(600 * 52 / 12, 0.01)
    assert cf.surplus == cf.inflow - cf.fixed - cf.variable


def pytest_approx(target: float, tol_rel: float = 0.01):
    class _A:
        def __eq__(self, other):
            return abs(other - target) / max(abs(target), 1e-9) <= tol_rel
    return _A()


def test_project_grows_with_contributions():
    bals = project_net_worth(start_value=0, monthly_contribution=1000,
                                annual_return=0.07, years=10)
    assert len(bals) == 11
    assert bals[-1] > bals[0]


def test_required_monthly_zero_when_already_above_target():
    assert required_monthly(target=1000, current=2000, annual_return=0.07, years=10) == 0.0


def test_freedom_date_unreachable_with_zero_return():
    out = freedom_date(net_worth=0, monthly_burn=1000, monthly_surplus=100,
                         blended_return=0.0)
    assert out["reached"] is False
    assert out["needed_nw"] == 1000 * 12 / 0.04


def test_annual_yield_only_counts_yielding_types():
    assets = [
        {"type": "Equity", "current_value": 1000},
        {"type": "Crypto", "current_value": 1000},
        {"type": "Physical", "current_value": 1000},
    ]
    settings = {"expected_return": 0.07, "savings_apy": 0.045}
    # Only Equity contributes; Crypto and Physical don't.
    assert annual_yield(assets, settings) == 70.0
