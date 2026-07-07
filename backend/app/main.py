from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.api import (
    admin,
    auth,
    calendar,
    config,
    current_week,
    picks,
    results,
    slate,
)
from app.config import settings
from app.csrf import csrf_dispatch
from app.exception_handlers import add_exception_handlers

# Loud demo-mode banner (belt-and-suspenders for "never on silently",
# T-sf0-02). Logged at import/startup so EVERY API process surfaces the demo
# state, not just the seed CLI. A no-op when the flag is OFF (the prod path),
# so prod startup is unchanged.
if settings.is_demo_data:
    import logging

    _demo_line = "=" * 72
    logging.getLogger("uvicorn.error").warning(
        "\n%s\n  WARNING: IS_DEMO_DATA ON — this API is serving the FAKE "
        "time-shifted 2025 DEMO season.\n  This is NOT production. Disable "
        "IS_DEMO_DATA before go-live.\n%s",
        _demo_line,
        _demo_line,
    )

app = FastAPI(title="NFL Pick'em API", version="1.1.8")

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
app.include_router(picks.router)
app.include_router(results.router)
app.include_router(current_week.router)
app.include_router(slate.router)
app.include_router(calendar.router)
app.include_router(config.router)
app.include_router(admin.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
