"""On-demand live-odds seam for the @mention prediction intent (260710-mpw).

The prediction intent labels its line the **current market** line — fetched live at
ask-time so it stays contemporaneous with live injuries/weather — NOT the frozen
pick'em spread. This is that best-effort live-line seam, mirroring the impure-shell /
pure-never-raising-core posture of :mod:`app.services.espn_extra`:

* IMPURE: :func:`fetch_live_odds` — a best-effort async shell that first consults a
  short-TTL Redis cache (keyed by season+week — one cached page serves every game that
  week), else GETs the public site scoreboard over ``httpx`` and writes the raw payload
  back. It NEVER raises: any HTTP/timeout/non-200/parse error degrades to ``None`` (the
  caller falls back to the frozen spread, relabelled — never bails).
* PURE: :func:`select_live_odds_for_event` — reuses
  :func:`app.scoreboard.espn.select_odds_item` + :func:`app.scoreboard.espn.normalize_odds`
  (the SAME parser the ingest poller uses — the design's REUSE mandate; odds parsing is
  never re-implemented) to index a site-scoreboard payload by event id and return one
  event's :class:`~app.scoreboard.types.ScoreboardOdds`, or ``None``. Defensive on every
  field; never raises — this is what the offline unit tests exercise.

SSRF note (T-mpw-01): the scoreboard URL carries ONLY int ``season``/``week`` resolved
from our own DB — never user text — the same SSRF-safe posture as ``espn_extra``'s
event-id input. This module imports NO ``discord``; the qa.py brain imports THIS seam
for the HTTP + cache, staying itself httpx-free.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import structlog

from app.config import settings
from app.scoreboard.espn import SITE_SCOREBOARD_URL, normalize_odds, select_odds_item
from app.scoreboard.types import ScoreboardOdds

logger = structlog.get_logger(__name__)

# A plain outbound-only UA + explicit timeout (mirror the espn_extra QA seam, not the
# 20s ingest timeout — a hung ESPN response must not block the bot loop).
_USER_AGENT = "nfl-pickem-qa/1.0 (dev tooling; httpx)"
DEFAULT_TIMEOUT = 10.0

# Short Redis cache: line movement is the whole point of a LIVE line, so keep the TTL
# short — but a flurry of asks on the same week is still ONE upstream call.
LIVE_ODDS_CACHE_TTL_SECONDS = 300


def _cache_key(season: int, week: int) -> str:
    """The Redis key for one week's cached site-scoreboard payload.

    Keyed by season+week only (the page carries every game that week), so repeat asks
    for DIFFERENT games in the same week reuse ONE cached page.
    """
    return f"qa:live_odds:scoreboard:{season}:{week}"


# ---------------------------------------------------------------------------
# Pure parsing (no network — unit-tested offline)
# ---------------------------------------------------------------------------


def _index_odds_by_event(payload: Any) -> dict[str, ScoreboardOdds]:
    """Build an ``{event_id -> ScoreboardOdds}`` index from a site-scoreboard payload.

    Pure and never-raising: reuses :func:`select_odds_item` + :func:`normalize_odds`
    over each event's ``competitions[0].odds[]``. An event with no usable odds is simply
    omitted; a non-dict payload / missing ``events`` yields an empty index.
    """
    payload = payload if isinstance(payload, dict) else {}
    events = payload.get("events")
    events = events if isinstance(events, list) else []
    index: dict[str, ScoreboardOdds] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        event_id = event.get("id")
        if event_id is None:
            continue
        competitions = event.get("competitions")
        competition = competitions[0] if isinstance(competitions, list) and competitions else {}
        competition = competition if isinstance(competition, dict) else {}
        odds = normalize_odds(select_odds_item(competition.get("odds")))
        if odds is not None:
            index[str(event_id)] = odds
    return index


def select_live_odds_for_event(payload: Any, event_id: int) -> ScoreboardOdds | None:
    """Return the normalized live odds for ``event_id`` from ``payload``, or ``None``.

    Pure and never-raising: indexes the payload via :func:`_index_odds_by_event` and
    returns the entry matching ``str(event_id)`` (ESPN reports event ids as strings; our
    DB event id is an int). ``None`` when the event is absent or carries no odds.
    """
    return _index_odds_by_event(payload).get(str(event_id))


# ---------------------------------------------------------------------------
# Impure shell (best-effort HTTP + short Redis cache — never raises)
# ---------------------------------------------------------------------------


def _redis_client():
    """Build an async Redis client from ``settings.redis_url`` (single seam).

    Isolated so tests monkeypatch it without a real socket (mirrors
    :func:`app.services.espn_extra._redis_client`). A fresh client per call keeps it
    bound to the calling event loop.
    """
    import redis.asyncio as aioredis

    return aioredis.Redis.from_url(settings.redis_url)


async def _cache_get(season: int, week: int) -> dict | None:
    """Return the cached scoreboard payload for ``season``/``week``, else ``None`` (FAIL-OPEN).

    Best-effort: any Redis/JSON error logs a warning and returns ``None`` so the caller
    degrades to a live fetch. A missing key also returns ``None``.
    """
    try:
        client = _redis_client()
        try:
            raw = await client.get(_cache_key(season, week))
        finally:
            await client.aclose()
        if raw is None:
            return None
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        logger.warning("live_odds_cache_get_failed", season=season, week=week, exc_info=True)
        return None


async def _cache_set(season: int, week: int, payload: dict) -> None:
    """Best-effort write of ``payload`` under the week key + TTL.

    FAIL-OPEN: any Redis/JSON error logs a warning and returns normally — a cache outage
    must NOT block the fetch that already succeeded.
    """
    try:
        client = _redis_client()
        try:
            await client.set(
                _cache_key(season, week),
                json.dumps(payload),
                ex=LIVE_ODDS_CACHE_TTL_SECONDS,
            )
        finally:
            await client.aclose()
    except Exception:
        logger.warning("live_odds_cache_set_failed", season=season, week=week, exc_info=True)


async def fetch_live_odds(season: int, week: int, event_id: int) -> ScoreboardOdds | None:
    """Fetch the live market odds for ``event_id`` — best-effort (mirrors ``fetch_injuries``).

    Cache-first: on a cache HIT the cached scoreboard page is reused WITHOUT any HTTP
    call. On a MISS it performs ONE ``httpx`` GET to :data:`SITE_SCOREBOARD_URL` (explicit
    ~10s timeout, ``_USER_AGENT`` header), parses the JSON, and best-effort caches the raw
    page. Either way it returns the target event's normalized
    :class:`~app.scoreboard.types.ScoreboardOdds` via :func:`select_live_odds_for_event`,
    or ``None``. NEVER raises: any HTTP/timeout/non-200/parse error degrades to ``None``
    (the caller falls back to the frozen spread, relabelled); a Redis outage fails open.
    The URL carries ONLY int ``season``/``week`` — never user text (T-mpw-01).
    """
    payload = await _cache_get(season, week)
    if payload is None:
        url = SITE_SCOREBOARD_URL.format(season=season, week=week)
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                response = await client.get(url, headers={"User-Agent": _USER_AGENT})
            if response.status_code != 200:
                logger.warning("live_odds_fetch_non_200", status_code=response.status_code)
                return None
            fetched = response.json()
        except Exception:
            logger.warning("live_odds_fetch_failed", season=season, week=week, exc_info=True)
            return None
        if not isinstance(fetched, dict):
            return None
        payload = fetched
        await _cache_set(season, week, payload)

    return select_live_odds_for_event(payload, event_id)
