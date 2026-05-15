"""Deterministic forecast eval harness. Runs the pure-math tools against
known-good answers and writes the result to baseline.json.

Skipped on every /metrics scrape — we cache the last result on disk to
avoid re-running on a hot path.

Run:  python -m evals.run_evals
"""

import json
import time
from pathlib import Path

from src.tools.forecast import (
    freedom_date,
    project_net_worth,
    required_monthly,
    required_return,
)

CASES = [
    {
        "name": "compound_10k_at_7_for_10y",
        "tool": "project_net_worth",
        "args": {"start_value": 10_000, "monthly_contribution": 0,
                  "annual_return": 0.07, "years": 10},
        "expected_final": 20_096.61,  # 10000 * (1 + 0.07/12)^120
        "tolerance": 0.01,
    },
    {
        "name": "compound_50k_plus_3k_mo_at_7_for_20y",
        "tool": "project_net_worth",
        "args": {"start_value": 50_000, "monthly_contribution": 3_000,
                  "annual_return": 0.07, "years": 20},
        "expected_final": 1_761_823,  # within rounding
        "tolerance": 0.01,
    },
    {
        "name": "required_monthly_to_1M_in_20y_at_7_from_50k",
        "tool": "required_monthly",
        "args": {"target": 1_000_000, "current": 50_000,
                  "annual_return": 0.07, "years": 20},
        "expected": 1_532.01,
        "tolerance": 0.05,
    },
    {
        "name": "freedom_date_50k_5k_burn_3k_surplus_at_7",
        "tool": "freedom_date",
        "args": {"net_worth": 50_000, "monthly_burn": 5_000,
                  "monthly_surplus": 3_000, "blended_return": 0.07},
        "expected_needed_nw": 857_142.86,
        "tolerance": 0.005,
    },
]


def _run_one(case: dict) -> dict:
    name = case["tool"]
    args = case["args"]
    if name == "project_net_worth":
        bals = project_net_worth(**args)
        return {"final": round(bals[-1], 2), "years": len(bals) - 1}
    if name == "required_monthly":
        return {"required_monthly": round(required_monthly(**args), 2)}
    if name == "required_return":
        v = required_return(**args)
        return {"required_return": round(v, 4) if v is not None else None}
    if name == "freedom_date":
        return freedom_date(**args)
    raise ValueError(name)


def main() -> int:
    out_path = Path(__file__).resolve().parent / "baseline.json"
    rows = []
    failed = 0
    started = time.time()
    for case in CASES:
        result = _run_one(case)
        ok = True
        if "expected_final" in case:
            ok = abs(result["final"] - case["expected_final"]) / case["expected_final"] <= case["tolerance"]
        elif "expected" in case:
            ok = abs(result[next(iter(result))] - case["expected"]) / case["expected"] <= case["tolerance"]
        elif "expected_needed_nw" in case:
            ok = abs(result["needed_nw"] - case["expected_needed_nw"]) / case["expected_needed_nw"] <= case["tolerance"]
        rows.append({"name": case["name"], "ok": ok, "result": result})
        if not ok:
            failed += 1
    summary = {
        "ran_at": time.time(),
        "duration_s": round(time.time() - started, 4),
        "total": len(CASES),
        "passed": len(CASES) - failed,
        "failed": failed,
        "cases": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
