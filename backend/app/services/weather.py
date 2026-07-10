"""On-demand Open-Meteo weather adapter — game-time forecast (Path B slice 2, 260710-29v).

The second Path-B seam (outside-intelligence): when a league member @mentions the bot
with a weather question, the bot resolves the asked team's current-week game to its HOME
stadium, fetches the hourly forecast from Open-Meteo RIGHT THEN, caches the raw payload
briefly in Redis, and indexes the hourly arrays by the kickoff hour into a deterministic
temp/wind/precip fact. NO new DB tables, NO Celery-beat poller — freshness improves near
kickoff, so on-demand + a short cache is always fresh enough (design:
``.planning/notes/discord-query-bot-path-b-design.md``).

Design — impure shell / pure never-raising core (mirrors :mod:`app.services.espn_extra`):

* STATIC: :data:`STADIUMS` — a hand-authored 32-row ``home_abbr -> Stadium`` table
  (lat/lon + an ``indoor`` flag + name). ESPN's ``venue`` carries NO coordinates, so
  this table is the ONLY coordinate source. The ``indoor`` flag lets a dome/retractable
  game short-circuit the fetch entirely (weather is a non-factor indoors). :func:`lookup_stadium`
  is a pure, case-insensitive lookup.
* PURE: :func:`parse_forecast` — takes the already-parsed Open-Meteo payload + a kickoff
  ``datetime`` and returns ``{temperature_f, wind_mph, precip_in, hour}`` indexed by the
  kickoff hour (a naive kickoff is assumed UTC; the hour is floored and matched against
  ``hourly.time``). Defensive on EVERY field (isinstance guards, ``.get``): a missing
  single metric degrades to ``None`` (never invented) while a usable hour still returns
  the dict; an unusable shape / an absent hour / all-missing metrics returns ``None``.
  Never raises — this is what the offline unit tests exercise.
* IMPURE: :func:`fetch_forecast` — a best-effort async shell that first consults a
  short-TTL Redis cache (location-scoped key), else GETs the Open-Meteo ``forecast``
  endpoint over ``httpx`` (no API key) and writes the raw payload back to the cache. It
  NEVER raises: any HTTP/timeout/non-200 degrades to ``None`` (the caller shows a fixed
  degrade line, never an invented forecast), and a Redis outage FAILS OPEN.

This module imports NO ``discord`` and lives on the Discord-free side: the qa.py brain
imports THIS seam for the coordinate lookup + HTTP + cache, staying itself HTTP-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirror the espn_extra adapter's endpoint/UA/timeout conventions)
# ---------------------------------------------------------------------------

# The public, no-auth Open-Meteo hourly forecast endpoint. US-friendly units
# (°F / mph / inch), GMT time base so the hourly ``time[]`` keys are UTC-aligned
# (``utc_offset_seconds`` 0), and a wide 16-day window so more games fall inside the
# hourly horizon. ``latitude``/``longitude`` come ONLY from the static STADIUMS table
# (a canonical home abbr resolved from our DB) — NEVER from user text (T-29v-02).
FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&hourly=temperature_2m,precipitation,wind_speed_10m"
    "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
    "&timezone=GMT&forecast_days=16"
)

# A plain outbound-only UA (mirror espn_extra ``_USER_AGENT``); no credentials sent.
_USER_AGENT = "nfl-pickem-qa/1.0 (dev tooling; httpx)"

# Explicit timeout so a hung/slow Open-Meteo response cannot block the bot loop.
DEFAULT_TIMEOUT = 10.0

# Short Redis cache: forecast improves near kickoff, so a ~30 min TTL cushions repeat
# asks for the same stadium into ONE upstream call without going stale.
WEATHER_CACHE_TTL_SECONDS = 1800


# ---------------------------------------------------------------------------
# Static stadium table (hand-authored — accuracy matters; a wrong lat/lon silently
# returns the WRONG city's weather). ESPN ``venue`` carries no coordinates, so this
# is the ONLY coordinate source. ``indoor`` is True for fixed domes, fixed roofs, and
# retractable-roof venues that are usually closed — those short-circuit the fetch.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stadium:
    """A single NFL home stadium: display name, coordinates, and an indoor flag."""

    name: str
    lat: float
    lon: float
    indoor: bool


# Keyed by the HOME team's canonical abbreviation (matches app/seeds/teams.py, the same
# abbreviations notifications_read resolves from ``Team.abbreviation``). LAR + LAC both
# play at SoFi (both indoor, same coordinates). The 11 indoor venues: ATL, NO, DET, MIN,
# LV, LAR, LAC, ARI, DAL, HOU, IND.
STADIUMS: dict[str, Stadium] = {
    "ATL": Stadium("Mercedes-Benz Stadium", 33.7554, -84.4008, True),
    "BUF": Stadium("Highmark Stadium", 42.7738, -78.7870, False),
    "CHI": Stadium("Soldier Field", 41.8623, -87.6167, False),
    "CIN": Stadium("Paycor Stadium", 39.0955, -84.5161, False),
    "CLE": Stadium("Huntington Bank Field", 41.5061, -81.6995, False),
    "DAL": Stadium("AT&T Stadium", 32.7473, -97.0945, True),
    "DEN": Stadium("Empower Field at Mile High", 39.7439, -105.0201, False),
    "DET": Stadium("Ford Field", 42.3400, -83.0456, True),
    "GB": Stadium("Lambeau Field", 44.5013, -88.0622, False),
    "TEN": Stadium("Nissan Stadium", 36.1665, -86.7713, False),
    "IND": Stadium("Lucas Oil Stadium", 39.7601, -86.1639, True),
    "KC": Stadium("Arrowhead Stadium", 39.0489, -94.4839, False),
    "LV": Stadium("Allegiant Stadium", 36.0909, -115.1833, True),
    "LAR": Stadium("SoFi Stadium", 33.9535, -118.3392, True),
    "MIA": Stadium("Hard Rock Stadium", 25.9580, -80.2389, False),
    "MIN": Stadium("U.S. Bank Stadium", 44.9736, -93.2575, True),
    "NE": Stadium("Gillette Stadium", 42.0909, -71.2643, False),
    "NO": Stadium("Caesars Superdome", 29.9511, -90.0812, True),
    "NYG": Stadium("MetLife Stadium", 40.8135, -74.0745, False),
    "NYJ": Stadium("MetLife Stadium", 40.8135, -74.0745, False),
    "PHI": Stadium("Lincoln Financial Field", 39.9008, -75.1675, False),
    "ARI": Stadium("State Farm Stadium", 33.5276, -112.2626, True),
    "PIT": Stadium("Acrisure Stadium", 40.4468, -80.0158, False),
    "LAC": Stadium("SoFi Stadium", 33.9535, -118.3392, True),
    "SF": Stadium("Levi's Stadium", 37.4030, -121.9700, False),
    "SEA": Stadium("Lumen Field", 47.5952, -122.3316, False),
    "TB": Stadium("Raymond James Stadium", 27.9759, -82.5033, False),
    "WSH": Stadium("Northwest Stadium", 38.9077, -76.8645, False),
    "CAR": Stadium("Bank of America Stadium", 35.2258, -80.8528, False),
    "JAX": Stadium("EverBank Stadium", 30.3239, -81.6373, False),
    "BAL": Stadium("M&T Bank Stadium", 39.2780, -76.6227, False),
    "HOU": Stadium("NRG Stadium", 29.6847, -95.4107, True),
}


def lookup_stadium(home_abbr: str) -> Stadium | None:
    """Resolve a HOME-team abbreviation to its :class:`Stadium`, or ``None``.

    Pure and case/whitespace-insensitive: upper-cases + strips the key and returns the
    matching row, or ``None`` when the abbreviation is not one of the 32 (or is blank).
    """
    if not isinstance(home_abbr, str):
        return None
    return STADIUMS.get(home_abbr.strip().upper())


# ---------------------------------------------------------------------------
# Pure parsing (no network — unit-tested offline)
# ---------------------------------------------------------------------------


def _as_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC (a naive datetime is ASSUMED UTC).

    Mirrors ``notifications_read._as_aware``: a naive kickoff read back from SQLite
    is treated as UTC; a tz-aware kickoff is converted to UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _numeric_at(values: Any, index: int) -> float | int | None:
    """Return the numeric value at ``index`` in ``values``, else ``None``.

    Defensive: ``values`` must be a list long enough to hold ``index`` and the entry
    must be a real number (``bool`` is explicitly rejected — a JSON true/false is NOT
    a metric). Anything else degrades to ``None`` (never invented).
    """
    if not isinstance(values, list) or index >= len(values):
        return None
    value = values[index]
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def parse_forecast(payload: Any, kickoff_dt: datetime) -> dict | None:
    """Extract the kickoff-hour forecast from an Open-Meteo ``forecast`` payload.

    Pure and never-raising (mirrors ``espn_extra.parse_injuries``):

    * Returns ``{temperature_f, wind_mph, precip_in, hour}`` with each metric read at
      the index whose ``hourly.time[]`` entry equals the kickoff hour key (the kickoff
      normalized to UTC, floored to the hour, formatted ``"%Y-%m-%dT%H:00"`` to match
      Open-Meteo's ``timezone=GMT`` output). A single missing/short/non-numeric metric
      degrades to ``None`` (never invented) as long as at least one metric is present.
    * Returns ``None`` when the top-level shape is unusable (non-dict payload, or
      ``hourly`` is not a dict with a list ``time``), when the kickoff hour is not in
      ``hourly.time`` (degrade — never guess a neighboring hour), or when all three
      metrics are absent for that hour.
    """
    if not isinstance(payload, dict):
        return None
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        return None
    times = hourly.get("time")
    if not isinstance(times, list):
        return None

    hour_key = (
        _as_utc(kickoff_dt).replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
    )
    if hour_key not in times:
        return None
    index = times.index(hour_key)

    temperature_f = _numeric_at(hourly.get("temperature_2m"), index)
    wind_mph = _numeric_at(hourly.get("wind_speed_10m"), index)
    precip_in = _numeric_at(hourly.get("precipitation"), index)

    if temperature_f is None and wind_mph is None and precip_in is None:
        return None

    return {
        "temperature_f": temperature_f,
        "wind_mph": wind_mph,
        "precip_in": precip_in,
        "hour": hour_key,
    }


# ---------------------------------------------------------------------------
# Impure shell (best-effort HTTP + short Redis cache — never raises)
# ---------------------------------------------------------------------------


def _cache_key(lat: float, lon: float) -> str:
    """The Redis key for one LOCATION's cached forecast payload.

    Location-scoped (the payload carries all hours for the stadium), so the kickoff is
    NOT part of the key — repeat asks for the same stadium reuse the one cached fetch.
    """
    return f"qa:weather:forecast:{lat:.2f}:{lon:.2f}"


def _redis_client():
    """Build an async Redis client from ``settings.redis_url`` (single seam).

    Isolated as a tiny seam so tests monkeypatch it without touching a real socket
    (mirror :func:`app.services.espn_extra._redis_client`). A fresh client per call
    keeps it bound to the calling event loop (these reads happen a few times a week).
    """
    import redis.asyncio as aioredis

    return aioredis.Redis.from_url(settings.redis_url)


async def _cache_get(lat: float, lon: float) -> dict | None:
    """Return the cached forecast payload for a location, or ``None`` (FAIL-OPEN).

    Best-effort: any Redis/JSON error logs a warning and returns ``None`` so the
    caller degrades to a live fetch. A missing key also returns ``None``.
    """
    try:
        client = _redis_client()
        try:
            raw = await client.get(_cache_key(lat, lon))
        finally:
            await client.aclose()
        if raw is None:
            return None
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        logger.warning("weather_cache_get_failed", lat=lat, lon=lon, exc_info=True)
        return None


async def _cache_set(lat: float, lon: float, payload: dict) -> None:
    """Best-effort write of ``payload`` to the cache under the location key + TTL.

    FAIL-OPEN: any Redis/JSON error logs a warning and returns normally — a cache
    outage must NOT block the fetch that already succeeded.
    """
    try:
        client = _redis_client()
        try:
            await client.set(
                _cache_key(lat, lon),
                json.dumps(payload),
                ex=WEATHER_CACHE_TTL_SECONDS,
            )
        finally:
            await client.aclose()
    except Exception:
        logger.warning("weather_cache_set_failed", lat=lat, lon=lon, exc_info=True)


async def fetch_forecast(lat: float, lon: float) -> dict | None:
    """Fetch the raw Open-Meteo forecast payload for a location — best-effort.

    Cache-first: on a cache HIT the cached payload is returned WITHOUT any HTTP call.
    On a MISS it performs ONE ``httpx`` GET to the Open-Meteo ``forecast`` endpoint
    (explicit ~10s timeout), returns the parsed JSON dict, and best-effort writes the
    raw payload back to the cache. NEVER raises: any HTTP/timeout/non-200/parse error
    degrades to ``None`` (the caller shows a fixed degrade line, never an invented
    forecast); a Redis outage on either the read or the write fails open (the read
    degrades to a live fetch, the write is skipped). ``lat``/``lon`` come ONLY from the
    static STADIUMS table — never from user text (SSRF-safe, T-29v-02).
    """
    cached = await _cache_get(lat, lon)
    if cached is not None:
        return cached

    url = FORECAST_URL.format(lat=lat, lon=lon)
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(url, headers={"User-Agent": _USER_AGENT})
        if response.status_code != 200:
            logger.warning("weather_fetch_non_200", status_code=response.status_code)
            return None
        payload = response.json()
    except Exception:
        logger.warning("weather_fetch_failed", lat=lat, lon=lon, exc_info=True)
        return None

    if not isinstance(payload, dict):
        return None

    await _cache_set(lat, lon, payload)
    return payload
