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
* IMPURE: :func:`fetch_forecast` — a thin delegation to
  :func:`app.services.http_cache.fetch_cached`, which owns the contract (cache-first on a
  location-scoped key, one GET, never raises, fail-open Redis). This seam supplies the
  Open-Meteo URL (no API key) and its own branded ``User-Agent``.

This module imports NO ``discord`` and lives on the Discord-free side: the qa.py brain
imports THIS seam for the coordinate lookup + HTTP + cache, staying itself HTTP-free.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import structlog

from app.config import settings
from app.services import http_cache

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
    "&hourly=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,precipitation_probability"
    "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
    "&timezone=GMT&forecast_days=16"
)

# A plain outbound-only UA (mirror espn_extra ``_USER_AGENT``); no credentials sent.
_USER_AGENT = "nfl-pickem-qa/1.0 (dev tooling; httpx)"

# One source of truth for the timeout — the shared shell owns the value.
DEFAULT_TIMEOUT = http_cache.DEFAULT_TIMEOUT
# Issue #283: a first forecast GET took the whole 10 s and the answer said "unavailable";
# the retry a minute later took 2.8 s. Two 5 s tries keep the same worst case.
_FORECAST_TIMEOUT_SECONDS = 5.0

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
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


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

    * Returns ``{temperature_f, wind_mph, precip_in, wind_gust_mph, precip_chance_pct,
      hour}`` with each metric read at the index whose ``hourly.time[]`` entry equals the kickoff hour key (the kickoff
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
        # Issue #286: a storm question needs the gusts and the chance, not only the amount.
        "wind_gust_mph": _numeric_at(hourly.get("wind_gusts_10m"), index),
        "precip_chance_pct": _numeric_at(hourly.get("precipitation_probability"), index),
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


async def fetch_forecast(lat: float, lon: float) -> dict | None:
    """Fetch the raw Open-Meteo forecast payload for a location — best-effort.
    The contract lives in :func:`app.services.http_cache.fetch_cached`. Open-Meteo gets
    this module's branded ``User-Agent``; ``lat``/``lon`` come ONLY from the static
    STADIUMS table, never from user text (SSRF-safe, T-29v-02).
    """
    # ``_redis_client`` is read from the module HERE, at call time, so the tests' patch
    # of this module's seam still takes effect (a default argument would defeat it).
    return await http_cache.fetch_cached(
        FORECAST_URL.format(lat=lat, lon=lon),
        cache_key=_cache_key(lat, lon),
        ttl_seconds=WEATHER_CACHE_TTL_SECONDS,
        label="weather",
        redis_client=_redis_client,
        headers={"User-Agent": _USER_AGENT},
        timeout=_FORECAST_TIMEOUT_SECONDS,
        attempts=2,
    )


# ---------------------------------------------------------------------------
# Neutral sites (issue #288): BAL vs DAL in Rio got AT&T Stadium's dome line.
# ---------------------------------------------------------------------------

VENUE_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/venues/{venue_id}"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search?name={name}&count=10"
# A venue's roof and a city's coordinates do not change within a season.
VENUE_CACHE_TTL_SECONDS = 30 * 86400
_VENUE_ID_RE = re.compile(r"[0-9]{1,9}")
# ESPN writes the country its own way; Open-Meteo writes the full English name.
_COUNTRY_ALIASES = {
    "usa": "united states",
    "us": "united states",
    "uk": "united kingdom",
    "england": "united kingdom",
    "scotland": "united kingdom",
    "wales": "united kingdom",
}


def _fold(text: str) -> str:
    """Lower-case with accents removed, so "Sao Paulo" matches "São Paulo"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip().casefold()


def parse_neutral_venue(summary: Any) -> dict | None:
    """The venue of a neutral-site game from an ESPN ``summary``, else ``None``.

    ``None`` means the game is NOT marked neutral (or the payload is unusable), so the
    home team's stadium applies. A neutral game always returns a dict; a missing field is
    ``None`` and the caller then gives no forecast rather than the home stadium's.
    """
    if not isinstance(summary, dict):
        return None
    header = summary.get("header")
    competitions = header.get("competitions") if isinstance(header, dict) else None
    first = competitions[0] if isinstance(competitions, list) and competitions else None
    if not isinstance(first, dict) or first.get("neutralSite") is not True:
        return None
    info = summary.get("gameInfo")
    venue = info.get("venue") if isinstance(info, dict) else None
    venue = venue if isinstance(venue, dict) else {}
    address = venue.get("address")
    address = address if isinstance(address, dict) else {}

    def _text(value: Any) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    venue_id = _text(venue.get("id"))
    return {
        "id": venue_id if venue_id and _VENUE_ID_RE.fullmatch(venue_id) else None,
        "name": _text(venue.get("fullName")),
        "city": _text(address.get("city")),
        "country": _text(address.get("country")),
    }


def pick_geocode(payload: Any, country: str | None) -> tuple[float, float] | None:
    """The first geocoding result in ``country``; ``None`` when none matches."""
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not country:
        return None
    wanted = _COUNTRY_ALIASES.get(_fold(country), _fold(country))
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("country"), str):
            continue
        if _fold(result["country"]) != wanted:
            continue
        lat, lon = result.get("latitude"), result.get("longitude")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return float(lat), float(lon)
    return None


async def _fetch_long(url: str, cache_key: str) -> dict | None:
    return await http_cache.fetch_cached(
        url,
        cache_key=cache_key,
        ttl_seconds=VENUE_CACHE_TTL_SECONDS,
        label="weather_venue",
        redis_client=_redis_client,
        headers={"User-Agent": _USER_AGENT},
        timeout=_FORECAST_TIMEOUT_SECONDS,
    )


async def _neutral_venue(espn_event_id: Any) -> tuple[bool, Stadium | None]:
    """``(is_neutral, stadium)``; a neutral venue that cannot be placed is ``(True, None)``."""
    if espn_event_id is None:
        return False, None
    from app.services import espn_extra

    venue = parse_neutral_venue(await espn_extra.fetch_game_summary(espn_event_id))
    if venue is None:
        return False, None
    name = venue["name"]
    # The Super Bowl is usually at an NFL stadium the table already holds.
    for stadium in STADIUMS.values():
        if name and _fold(stadium.name) == _fold(name):
            return True, stadium
    if not (name and venue["id"] and venue["city"]):
        return True, None
    detail = await _fetch_long(
        VENUE_URL.format(venue_id=venue["id"]), f"qa:weather:venue:{venue['id']}"
    )
    indoor = detail.get("indoor") if isinstance(detail, dict) else None
    if not isinstance(indoor, bool):
        return True, None
    city = _fold(venue["city"])
    geocode = await _fetch_long(
        GEOCODE_URL.format(name=quote(venue["city"])), f"qa:weather:geocode:{city}"
    )
    place = pick_geocode(geocode, venue["country"])
    if place is None:
        return True, None
    return True, Stadium(name, place[0], place[1], indoor)


async def resolve_stadium(home_abbr: str | None, espn_event_id: Any = None) -> Stadium | None:
    """The stadium a game is played at: a neutral site's venue, else the home team's.

    ``None`` means no forecast: an unknown home team, or a neutral venue that could not be
    placed. A neutral game never falls back to the home team's stadium. When the ESPN
    summary cannot be read, the home stadium applies, as it did before issue #288.
    """
    is_neutral, venue = await _neutral_venue(espn_event_id)
    if is_neutral:
        return venue
    return lookup_stadium(home_abbr or "")
