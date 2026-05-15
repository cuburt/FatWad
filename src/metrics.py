"""In-process counters incremented from middleware in src/main.py."""

from collections import defaultdict
from threading import Lock
from typing import Any

_lock = Lock()
_counts: dict[str, int] = defaultdict(int)
_latency_ms: list[float] = []
_errors: int = 0


def record_request(latency_ms: float, status: int, route: str) -> None:
    global _errors
    with _lock:
        _counts[f"requests_total:{route}"] += 1
        if status >= 500:
            _errors += 1
        # cap to last N to avoid unbounded growth
        _latency_ms.append(latency_ms)
        if len(_latency_ms) > 5000:
            del _latency_ms[:2500]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def snapshot() -> dict[str, Any]:
    with _lock:
        total = sum(_counts.values())
        return {
            "requests_total": total,
            "by_route": dict(_counts),
            "errors_total": _errors,
            "latency_ms_p50": _percentile(_latency_ms, 50),
            "latency_ms_p95": _percentile(_latency_ms, 95),
            "latency_ms_p99": _percentile(_latency_ms, 99),
        }
