"""The ONE fetch-and-cache shell behind every outbound HTTP seam (#182).

Cache-first GET, ONE ``httpx.AsyncClient`` construction, ONE Redis read/write pair, ONE
never-raises contract. :mod:`app.services.espn_extra`, :mod:`app.services.live_odds` and
:mod:`app.services.weather` are thin delegations, so a cache or timeout fix lands in one
place instead of four. This module imports NOTHING from ``app``: each caller keeps its own
``_redis_client`` seam and passes it in, which is also what keeps the tests able to patch
that seam on the calling module.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

# Explicit timeout so a hung upstream response cannot block the bot loop.
DEFAULT_TIMEOUT = 10.0


async def _cache_read(key: str, *, label: str, redis_client: Callable[[], Any]) -> dict | None:
    """Return the cached payload at ``key``, else ``None``.

    FAIL-OPEN: any Redis/JSON error warns and returns ``None``, so the caller fetches live.
    """
    try:
        client = redis_client()
        try:
            raw = await client.get(key)
        finally:
            await client.aclose()
        if raw is None:
            return None
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        logger.warning(f"{label}_cache_get_failed", key=key, exc_info=True)
        return None


async def _cache_write(
    key: str, payload: dict, *, ttl_seconds: int, label: str, redis_client: Callable[[], Any]
) -> None:
    """Best-effort write of ``payload`` under ``key`` + ``ttl_seconds``.

    FAIL-OPEN: any Redis/JSON error warns and returns — a cache outage must NOT block a
    fetch that already succeeded.
    """
    try:
        client = redis_client()
        try:
            await client.set(key, json.dumps(payload), ex=ttl_seconds)
        finally:
            await client.aclose()
    except Exception:
        logger.warning(f"{label}_cache_set_failed", key=key, exc_info=True)


# ``headers`` defaults to None because ESPN's edge returns 403 for a branded User-Agent
# (PR #171, a live outage): an ESPN caller passes nothing and httpx sends its own default.
# Only the Open-Meteo caller passes an agent. Do NOT make a custom UA the default here.
async def fetch_cached(
    url: str,
    *,
    cache_key: str,
    ttl_seconds: int,
    label: str,
    redis_client: Callable[[], Any],
    headers: dict[str, str] | None = None,
) -> dict | None:
    """Cache-first GET of ``url`` returning the parsed JSON dict, or ``None``.

    A cache HIT returns the cached payload with NO HTTP call; a MISS performs EXACTLY one
    GET and best-effort writes the raw payload back under ``cache_key``. NEVER raises: any
    HTTP/timeout/non-200/parse error degrades to ``None`` (the caller shows a fixed degrade
    line, never an invented fact) and a Redis outage on the read or the write fails open. A
    payload that is not a dict returns ``None`` and is NOT cached. ``label`` prefixes every
    structlog event, so each caller keeps its own ``<label>_cache_get_failed`` /
    ``_cache_set_failed`` / ``_fetch_non_200`` / ``_fetch_failed`` names.
    """
    cached = await _cache_read(cache_key, label=label, redis_client=redis_client)
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(url, headers=headers)
        if response.status_code != 200:
            logger.warning(f"{label}_fetch_non_200", status_code=response.status_code)
            return None
        payload = response.json()
    except Exception:
        logger.warning(f"{label}_fetch_failed", key=cache_key, exc_info=True)
        return None

    if not isinstance(payload, dict):
        return None

    await _cache_write(
        cache_key, payload, ttl_seconds=ttl_seconds, label=label, redis_client=redis_client
    )
    return payload
