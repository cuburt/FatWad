"""FastAPI entry point. Runs init_db on startup, mounts the route groups,
and times every request into the metrics counters."""

import time

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from src.api.agent_routes import router as agent_router
from src.api.metrics_routes import router as metrics_router
from src.api.plan_routes import router as plan_router
from src.api.resource_routes import router as resource_router
from src.api.session_routes import router as session_router
from src.db.seed import main as seed_main
from src.metrics import record_request
from src.safety import redact_pii_processor

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        redact_pii_processor,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()

app = FastAPI(title="FatWad — Wealth Simulator Agent", version="1.0.0")


@app.on_event("startup")
def on_startup():
    try:
        seed_main()
        logger.info("startup_complete")
    except Exception as e:
        # Postgres may not be up at boot in a docker-compose race. Routes that
        # need the DB will fail until it comes up.
        logger.warning("startup_db_unavailable", error=str(e))


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        record_request(elapsed_ms, status if 'status' in locals() else 500,
                        route=request.url.path)
    return response


@app.get("/health")
def health() -> dict:
    return {"ok": True}


# Routers
app.include_router(agent_router, tags=["agent"])
app.include_router(session_router, tags=["sessions"])
app.include_router(resource_router, tags=["resources"])
app.include_router(plan_router, tags=["plan"])
app.include_router(metrics_router, tags=["metrics"])
