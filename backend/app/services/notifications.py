"""Best-effort Discord event publisher (QT-1 — the load-bearing transport).

This is the backend SIDE of the Discord notification pipe: the FastAPI process
serializes a small event and PUBLISHes it to a shared Redis channel. The bot
process subscribes (see :mod:`app.bot.notifier`), renders, and posts it into a
Discord channel. Backend never imports ``discord``; the bot owns all rendering.

v1 event schema
---------------
Every event is a JSON object::

    {
      "v": 1,                       # schema version (int)
      "type": "<str>",              # e.g. "user.login"
      "targets": ["logger"|"chat"], # which Discord surfaces should render it
      ...display-data-only fields   # e.g. "actor": "<display_name>"
    }

HARD RULE: only DISPLAY data is ever published — never passwords, tokens, emails,
session cookies, or any secret. The event crosses a trust boundary into Discord,
so the builders below carry display fields only (T-kd8-02).

Publisher contract
------------------
``publish_event`` is BEST-EFFORT: the entire body (client construction + publish)
is wrapped in try/except. On any failure it logs a structlog warning and returns
normally — it MUST NEVER raise. A Redis hiccup must not break a login (T-kd8-01).
The publisher uses the SYNCHRONOUS redis client because it runs inside the request
thread; the async client is the subscriber's concern.
"""

from __future__ import annotations

import json

import structlog

from app.config import settings

logger = structlog.get_logger(__name__)

# The single Redis pub/sub channel for cross-process notification events.
EVENTS_CHANNEL = "pickem:events"


def login_event(display_name: str) -> dict:
    """Build the v1 ``user.login`` event — a pure function, no I/O.

    Returns exactly ``{"v": 1, "type": "user.login", "targets": ["logger"],
    "actor": display_name}``. ``actor`` is the user's DISPLAY name only.
    """
    return {
        "v": 1,
        "type": "user.login",
        "targets": ["logger"],
        "actor": display_name,
    }


def _redis_client():
    """Construct a synchronous redis client from ``settings.redis_url``.

    Isolated as a tiny seam so tests can monkeypatch it without touching a real
    socket, and so the URL is never hardcoded (reuse the celery broker setting).
    """
    import redis

    return redis.Redis.from_url(settings.redis_url)


def publish_event(event: dict) -> None:
    """PUBLISH ``event`` (as JSON) to :data:`EVENTS_CHANNEL` — best-effort.

    Wraps the whole operation (client build + publish) in try/except: on ANY
    failure it logs a structlog warning and returns normally. NEVER raises, so a
    Redis outage can never break the caller (e.g. the login route).
    """
    try:
        client = _redis_client()
        client.publish(EVENTS_CHANNEL, json.dumps(event))
    except Exception:
        logger.warning(
            "notification_publish_failed",
            event_type=event.get("type"),
            channel=EVENTS_CHANNEL,
        )
