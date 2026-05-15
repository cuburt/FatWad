"""Lightweight in-process metrics. Counters live in a module dict so every
worker sees its own — fine for a single-process Streamlit demo."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter

from src.metrics import snapshot

router = APIRouter()


@router.get("/metrics")
def metrics() -> dict[str, Any]:
    return {"ts": datetime.utcnow().isoformat(), **snapshot()}
