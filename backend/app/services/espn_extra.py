"""On-demand ESPN "extras" adapter — live injury report (Path B slice 1, 260709-u0z).

The first Path-B seam (outside-intelligence): when a league member @mentions the bot
with an injuries question, the bot fetches the game ``summary`` from ESPN RIGHT THEN,
caches the raw payload briefly in Redis, and parses the asked team's injury block into
a deterministic per-player fact list. NO new DB tables, NO Celery-beat poller — freshness
matters most for injuries, so on-demand is always fresh (design:
``.planning/notes/discord-query-bot-path-b-design.md``).

Design — impure shell / pure never-raising core (mirrors :mod:`app.scoreboard.espn`):

* IMPURE: :func:`fetch_injuries` — a best-effort async shell that first consults a
  short-TTL Redis cache, else GETs the ``summary`` endpoint over ``httpx`` and writes
  the raw payload back to the cache. It NEVER raises: any HTTP/timeout/non-200 degrades
  to ``None`` (the caller shows a fixed "couldn't pull the injury report" line, never an
  invented injury), and a Redis outage FAILS OPEN (a cache miss degrades to a live fetch).
* PURE: :func:`parse_injuries` — takes the already-parsed ``summary`` dict + a canonical
  team abbreviation and returns the per-player fact list for THAT team (``[]`` when the
  team is present with no injuries, ``None`` when the top-level shape is unusable or the
  team block is absent). Defensive on EVERY field (isinstance guards, ``.get``, degrade
  to ``None``); never raises — this is what the offline unit tests exercise.

This module imports NO ``discord`` and lives on the Discord-free side: the qa.py brain
imports THIS seam for the HTTP+cache, staying itself HTTP-free.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirror the scoreboard adapter's endpoint/UA/timeout conventions)
# ---------------------------------------------------------------------------

# The public, no-auth ESPN game ``summary`` endpoint (SAME host family as the
# scoreboard we already poll). One call carries BOTH teams' injuries (+ game news).
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}"

# A plain outbound-only UA (mirror scoreboard ``_USER_AGENT``); no credentials sent.
_USER_AGENT = "nfl-pickem-qa/1.0 (dev tooling; httpx)"

# Explicit timeout so a hung/slow ESPN response cannot block the bot loop.
DEFAULT_TIMEOUT = 10.0

# Short Redis cache: freshness matters for injuries, but ~10 min cushions repeat asks
# so a flurry of questions on the same game is ONE upstream call.
INJURIES_CACHE_TTL_SECONDS = 600


def _cache_key(event_id: int) -> str:
    """The Redis key for one event's cached ``summary`` payload."""
    return f"qa:injuries:summary:{event_id}"


# ---------------------------------------------------------------------------
# Pure parsing (no network — unit-tested offline)
# ---------------------------------------------------------------------------


def _first_str(*values: Any) -> str | None:
    """Return the first non-empty string among ``values``, else ``None``."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_one_injury(injury: Any) -> dict[str, str | None] | None:
    """Normalize one ``injuries[].injuries[]`` entry into a per-player fact dict.

    Defensive on every field (``isinstance`` guards + ``.get``); degrades a missing
    field to ``None`` and never raises. Returns ``None`` for an unusable (non-dict)
    entry so the caller can skip it. Reads status defensively from EITHER the
    top-level ``status`` OR ``type.name`` (whichever the unofficial shape carries),
    the athlete display name / position abbreviation, the body part + return date
    from ``details``, and the injury-level ``date`` as the "as-of" freshness stamp.
    """
    if not isinstance(injury, dict):
        return None

    athlete = injury.get("athlete")
    athlete = athlete if isinstance(athlete, dict) else {}
    position = athlete.get("position")
    position = position if isinstance(position, dict) else {}

    type_obj = injury.get("type")
    type_obj = type_obj if isinstance(type_obj, dict) else {}

    details = injury.get("details")
    details = details if isinstance(details, dict) else {}

    return {
        "status": _first_str(injury.get("status"), type_obj.get("name")),
        "display_name": _first_str(athlete.get("displayName")),
        "position": _first_str(position.get("abbreviation")),
        "body_part": _first_str(details.get("type")),
        "return_date": _first_str(details.get("returnDate")),
        "date": _first_str(injury.get("date")),
    }


def parse_injuries(payload: Any, team_abbr: str) -> list[dict[str, str | None]] | None:
    """Extract ONLY ``team_abbr``'s per-player injury facts from a ``summary`` payload.

    Pure and never-raising (mirrors ``normalize_event``/``normalize_odds``):

    * Returns a list of per-player fact dicts (status / display_name / position /
      body_part / return_date / as-of ``date``, each ``None`` when absent) for the
      matched team's injury block.
    * Returns ``[]`` when the team's block is present but lists no injuries.
    * Returns ``None`` when the top-level shape is unusable (non-dict payload, or
      ``injuries`` is not a list) OR the asked team's block is not present — a
      distinct signal so the caller degrades to "couldn't pull the report" rather
      than falsely announcing "no injuries" for a team it could not locate.

    One ``summary`` call carries BOTH teams, so this filters to the block whose
    ``team.abbreviation`` (upper-cased) equals ``team_abbr`` (upper-cased).
    """
    if not isinstance(payload, dict):
        return None
    blocks = payload.get("injuries")
    if not isinstance(blocks, list):
        return None

    needle = team_abbr.strip().upper()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        team = block.get("team")
        team = team if isinstance(team, dict) else {}
        abbr = team.get("abbreviation")
        if not isinstance(abbr, str) or abbr.strip().upper() != needle:
            continue
        # Matched the asked team's block. Parse its (possibly empty) injury list.
        entries = block.get("injuries")
        entries = entries if isinstance(entries, list) else []
        players: list[dict[str, str | None]] = []
        for entry in entries:
            parsed = _parse_one_injury(entry)
            if parsed is not None:
                players.append(parsed)
        return players

    # The asked team's block was not present — degrade (never invent "no injuries").
    return None


# ---------------------------------------------------------------------------
# Impure shell (best-effort HTTP + short Redis cache — never raises)
# ---------------------------------------------------------------------------


def _redis_client():
    """Build an async Redis client from ``settings.redis_url`` (single seam).

    Isolated as a tiny seam so tests monkeypatch it without touching a real socket
    and so the URL is never hardcoded (reuse the broker setting, mirroring
    :func:`app.services.notifications._redis_client`). A fresh client per call keeps
    the client bound to the calling event loop (these reads happen a few times a
    week, so no pool churn concern).
    """
    import redis.asyncio as aioredis

    return aioredis.Redis.from_url(settings.redis_url)


async def _cache_get(event_id: int) -> dict | None:
    """Return the cached ``summary`` payload for ``event_id``, or ``None`` (FAIL-OPEN).

    Best-effort: any Redis/JSON error logs a warning and returns ``None`` so the
    caller degrades to a live fetch. A missing key also returns ``None``.
    """
    try:
        client = _redis_client()
        try:
            raw = await client.get(_cache_key(event_id))
        finally:
            await client.aclose()
        if raw is None:
            return None
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        logger.warning("injuries_cache_get_failed", event_id=event_id, exc_info=True)
        return None


async def _cache_set(event_id: int, payload: dict) -> None:
    """Best-effort write of ``payload`` to the cache under the event key + TTL.

    FAIL-OPEN: any Redis/JSON error logs a warning and returns normally — a cache
    outage must NOT block the fetch that already succeeded.
    """
    try:
        client = _redis_client()
        try:
            await client.set(
                _cache_key(event_id),
                json.dumps(payload),
                ex=INJURIES_CACHE_TTL_SECONDS,
            )
        finally:
            await client.aclose()
    except Exception:
        logger.warning("injuries_cache_set_failed", event_id=event_id, exc_info=True)


async def fetch_injuries(espn_event_id: int) -> dict | None:
    """Fetch the raw ESPN ``summary`` payload for ``espn_event_id`` — best-effort.

    Cache-first: on a cache HIT the cached payload is returned WITHOUT any HTTP call.
    On a MISS it performs ONE ``httpx`` GET to the ``summary`` endpoint (explicit
    ~10s timeout, mirroring ``llm_client._chat_completion``), returns the parsed JSON
    dict, and best-effort writes the raw payload back to the cache. NEVER raises: any
    HTTP/timeout/non-200/parse error degrades to ``None`` (the caller shows a fixed
    degrade line, never an invented injury); a Redis outage on either the read or the
    write fails open (the read degrades to a live fetch, the write is skipped).
    """
    cached = await _cache_get(espn_event_id)
    if cached is not None:
        return cached

    url = SUMMARY_URL.format(event_id=espn_event_id)
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(url, headers={"User-Agent": _USER_AGENT})
        if response.status_code != 200:
            logger.warning("injuries_fetch_non_200", status_code=response.status_code)
            return None
        payload = response.json()
    except Exception:
        logger.warning("injuries_fetch_failed", event_id=espn_event_id, exc_info=True)
        return None

    if not isinstance(payload, dict):
        return None

    await _cache_set(espn_event_id, payload)
    return payload
