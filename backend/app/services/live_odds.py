"""On-demand live-odds seam for the @mention prediction intent (260710-mpw).

The prediction intent labels its line the **current market** line — fetched live at
ask-time so it stays contemporaneous with live injuries/weather — NOT the frozen
pick'em spread. This is that best-effort live-line seam, mirroring the impure-shell /
pure-never-raising-core posture of :mod:`app.services.espn_extra`:

* IMPURE: :func:`fetch_live_odds` — a thin delegation to
  :func:`app.services.http_cache.fetch_cached`, which owns the contract (cache-first, one
  GET, never raises, fail-open Redis). This seam supplies the site-scoreboard URL and a
  season+week key, so one cached page serves every game that week. A degrade returns
  ``None`` and the caller falls back to the frozen spread, relabelled — never bails.
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

from typing import Any

import structlog

from app.config import settings
from app.scoreboard.espn import SITE_SCOREBOARD_URL, normalize_odds, select_odds_item
from app.scoreboard.types import ScoreboardOdds
from app.services import http_cache

logger = structlog.get_logger(__name__)

# One source of truth for the timeout — the shared shell owns the value.
DEFAULT_TIMEOUT = http_cache.DEFAULT_TIMEOUT

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


async def fetch_live_odds(season: int, week: int, event_id: int) -> ScoreboardOdds | None:
    """Return the live market odds for ``event_id``, or ``None`` — best-effort.
    The contract lives in :func:`app.services.http_cache.fetch_cached`. The season+week key
    means ONE cached page serves every game that week, and the URL carries only those two
    ints — never user text (T-mpw-01). NO headers: this ESPN edge 403s a branded agent.
    """
    # ``_redis_client`` is read from the module HERE, at call time, so the tests' patch
    # of this module's seam still takes effect (a default argument would defeat it).
    payload = await http_cache.fetch_cached(
        SITE_SCOREBOARD_URL.format(season=season, week=week),
        cache_key=_cache_key(season, week),
        ttl_seconds=LIVE_ODDS_CACHE_TTL_SECONDS,
        label="live_odds",
        redis_client=_redis_client,
    )
    if payload is None:
        return None
    return select_live_odds_for_event(payload, event_id)
