from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from starlette.middleware.base import BaseHTTPMiddleware

from app.api import auth, picks, proof, results
from app.config import settings
from app.csrf import csrf_dispatch
from app.db import get_session
from app.exception_handlers import add_exception_handlers
from app.models import TaskRun
from app.tasks import ping

app = FastAPI(title="NFL Pick'em API")

# Middleware order: the LAST add_middleware is outermost. CORS must be outermost
# so its headers are attached to every response — including a CSRF 403 — so add
# CSRF first (inner) and CORS last (outer).
app.add_middleware(BaseHTTPMiddleware, dispatch=csrf_dispatch)

# Explicit origins + credentials so cookie auth works cross-origin. A wildcard
# "*" is invalid with allow_credentials=True (browsers reject it).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

add_exception_handlers(app)
app.include_router(auth.router)
app.include_router(proof.router)
app.include_router(picks.router)
app.include_router(results.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ping")
def trigger_ping(message: str = "pong") -> dict:
    """Enqueue the fake Celery task and return its async id."""
    result = ping.delay(message)
    return {"task_id": result.id, "message": message}


@app.get("/api/task-runs")
def list_task_runs(session: Session = Depends(get_session)) -> list[TaskRun]:
    """List rows the worker has written — proves both ends share the DB."""
    return list(session.exec(select(TaskRun).order_by(TaskRun.id.desc())).all())
