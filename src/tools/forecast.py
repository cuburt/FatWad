"""Pure-math forecast tools the agent calls. No I/O, no LLM.

Mirrors the source repo's `src/tools/scheduler.py` shape: each public function
is a tool the agent can call. Kept dependency-free so unit tests don't need a
DB or network."""

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class CashflowSummary:
    inflow: float
    fixed: float
    variable: float
    surplus: float

    @property
    def burn(self) -> float:
        return self.fixed + self.variable


def total_net_worth(assets: list[dict]) -> float:
    return sum(float(a.get("current_value", 0) or 0) for a in assets)


def total_cost_basis(assets: list[dict]) -> float:
    return sum(float(a.get("purchase_price", 0) or 0) for a in assets)


def asset_roi(asset: dict) -> float:
    pp = float(asset.get("purchase_price", 0) or 0)
    cv = float(asset.get("current_value", 0) or 0)
    if pp <= 0:
        return 0.0
    return (cv - pp) / pp


def by_type(assets: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for a in assets:
        t = a.get("type", "Cash")
        out[t] = out.get(t, 0.0) + float(a.get("current_value", 0) or 0)
    return out


def cashflow(income: list[dict], fixed: list[dict], variable: list[dict]) -> CashflowSummary:
    inflow = sum(float(i.get("monthly", 0) or 0) for i in income)
    fx = sum(float(f.get("monthly", 0) or 0) for f in fixed)
    if variable:
        avg_weekly = sum(float(v.get("amount", 0) or 0) for v in variable) / len(variable)
        vb_monthly = avg_weekly * 52 / 12
    else:
        vb_monthly = 0.0
    return CashflowSummary(inflow=inflow, fixed=fx, variable=vb_monthly, surplus=inflow - fx - vb_monthly)


def annual_yield(assets: list[dict], settings: dict) -> float:
    eq = float(settings.get("expected_return", 0.07))
    cash = float(settings.get("savings_apy", 0.045))
    total = 0.0
    for a in assets:
        v = float(a.get("current_value", 0) or 0)
        t = a.get("type", "Cash")
        if t == "Equity":
            total += v * eq
        elif t == "Real Estate":
            total += v * 0.05
        elif t == "Cash":
            total += v * cash
    return total


def project_net_worth(start_value: float, monthly_contribution: float,
                       annual_return: float, years: int) -> list[float]:
    monthly_rate = annual_return / 12.0
    months = years * 12
    out = [start_value]
    bal = start_value
    for m in range(1, months + 1):
        bal = bal * (1 + monthly_rate) + monthly_contribution
        if m % 12 == 0:
            out.append(bal)
    return out


def freedom_date(net_worth: float, monthly_burn: float, monthly_surplus: float,
                  blended_return: float) -> dict:
    annual_burn = monthly_burn * 12
    if blended_return <= 0:
        return {"reached": False, "years": None, "date": None, "needed_nw": annual_burn / 0.04}
    needed = annual_burn / blended_return
    if net_worth >= needed:
        return {"reached": True, "years": 0, "date": date.today().isoformat(), "needed_nw": needed}
    monthly_rate = blended_return / 12
    bal = net_worth
    months = 0
    cap = 12 * 80
    while bal < needed and months < cap:
        bal = bal * (1 + monthly_rate) + monthly_surplus
        months += 1
    if bal < needed:
        return {"reached": False, "years": None, "date": None, "needed_nw": needed}
    target = date.today() + timedelta(days=int(months * 30.4375))
    return {"reached": True, "years": months / 12, "date": target.isoformat(), "needed_nw": needed}


def required_monthly(target: float, current: float, annual_return: float, years: int) -> float:
    if years <= 0:
        return max(0.0, target - current)
    n = years * 12
    r = annual_return / 12
    fv_current = current * (1 + r) ** n
    remaining = target - fv_current
    if remaining <= 0:
        return 0.0
    if r == 0:
        return remaining / n
    return remaining * r / ((1 + r) ** n - 1)


def required_return(target: float, current: float, monthly_contribution: float,
                     years: int) -> float | None:
    if years <= 0 or current <= 0:
        return None
    lo, hi = -0.20, 0.50
    for _ in range(80):
        mid = (lo + hi) / 2
        n = years * 12
        r = mid / 12
        fv = current * (1 + r) ** n + (
            monthly_contribution * (((1 + r) ** n - 1) / r) if r else monthly_contribution * n
        )
        if fv < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
