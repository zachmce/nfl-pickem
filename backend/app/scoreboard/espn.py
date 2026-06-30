"""The real ESPN site-scoreboard adapter for the ``ScoreboardSource`` port.

This is the production source: it fetches one regular-season week from the public
ESPN *site* scoreboard endpoint over the standard library ``urllib`` and
normalizes the denormalized JSON into :class:`~app.scoreboard.types.ScoreboardGame`
value objects. It satisfies :class:`~app.scoreboard.port.ScoreboardSource`
structurally (no inheritance).

Design — impure shell / pure core (mirrors ``backend/scripts/gen_2025_fixture.py``
and the sibling pure services):

* IMPURE: :func:`_http_get_json` (a thin stdlib-``urllib`` getter with a plain
  User-Agent and an explicit timeout) and
  :meth:`EspnScoreboardSource.fetch_week` (builds the URL, fetches, delegates).
  On any HTTP/URL error it raises :class:`~app.scoreboard.port.ScoreboardFetchError`
  (including the URL + reason) — it never silently returns an empty list.
* PURE: :func:`normalize_scoreboard` and its per-event / per-competitor / odds
  helpers take the already-parsed scoreboard dict and return the normalized
  games with NO network access — these are what the offline unit tests exercise.
  The pure path is defensive: it never trusts the shape and never raises, missing
  or malformed fields degrade to ``None``.

This deliberately re-implements (does NOT import) the proven calling/normalizing
patterns from ``backend/scripts/gen_2025_fixture.py`` and
``backend/app/seeds/fixture_2025.py`` — ``app/`` must not depend on ``scripts/``,
and the pure services must not couple to the seed beyond shared conventions.

Odds source note: unlike the fixture generator (which pulled historical odds from
the ESPN *core* API), the LIVE site path carries odds INLINE on
``competitions[0].odds[]`` — DraftKings on upcoming games, absent on completed
games. This adapter reads that inline site shape (see
``.planning/notes/espn-ingestion-strategy.md`` "Odds object shape").

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

from app.models import GameStatus
from app.scoreboard.port import ScoreboardFetchError
from app.scoreboard.types import ScoreboardGame, ScoreboardOdds, ScoreboardTeam

# ---------------------------------------------------------------------------
# Constants (mirror the gen script's site endpoint + UA + timeout conventions)
# ---------------------------------------------------------------------------

SITE_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    "?dates={season}&seasontype=2&week={week}"
)

# A plain UA avoids the occasional default-client block without impersonating a
# browser. No credentials are ever sent — these are public, outbound-only GETs.
_USER_AGENT = "nfl-pickem-scoreboard/1.0 (dev tooling; stdlib urllib)"

# Explicit timeout so a hung/slow ESPN response cannot block indefinitely.
DEFAULT_TIMEOUT = 20.0

# Map ESPN status type name -> our GameStatus (mirror fixture_2025._STATUS_NAME_MAP).
_STATUS_NAME_MAP = {
    "STATUS_FINAL": GameStatus.FINAL,
    "STATUS_IN_PROGRESS": GameStatus.IN_PROGRESS,
}

# Preferred odds provider on the live site path. Resolved by NAME first; provider
# ids drift across endpoints/time so we never hardcode the id (see
# espn-ingestion-strategy.md "Provider selection").
_PREFERRED_PROVIDER_NAME = "DraftKings"


# ---------------------------------------------------------------------------
# HTTP layer (impure — not exercised by the offline unit tests)
# ---------------------------------------------------------------------------


def _http_get_json(url: str, timeout: float = DEFAULT_TIMEOUT) -> Any:
    """GET ``url`` and parse the JSON body.

    Raises :class:`ScoreboardFetchError` (including the URL and reason) on any
    HTTP or URL/network error so the caller never confuses a failed fetch with an
    empty week.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise ScoreboardFetchError(f"HTTP {exc.code} fetching {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ScoreboardFetchError(f"network error fetching {url}: {exc.reason}") from exc
    return json.loads(body)


# ---------------------------------------------------------------------------
# Pure normalization (no network — unit-tested offline)
# ---------------------------------------------------------------------------


def _to_int_or_none(value: Any) -> int | None:
    """Best-effort int coercion; returns None for missing/blank/garbage."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _to_float_or_none(value: Any) -> float | None:
    """Best-effort float coercion; returns None for missing/blank/garbage."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _parse_kickoff(value: Any) -> datetime | None:
    """Parse an ESPN ISO kickoff string into a tz-aware datetime, else None.

    Mirrors ``fixture_2025._parse_kickoff``: normalizes a trailing ``Z`` to
    ``+00:00`` so :meth:`datetime.fromisoformat` yields a tz-aware value. Never
    raises — malformed input degrades to ``None``.
    """
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _map_status(status_type: Any) -> GameStatus:
    """Map an ESPN ``status.type`` object to a :class:`GameStatus` (defensive).

    Mirrors ``fixture_2025._map_status``: match the status NAME first
    (``STATUS_FINAL`` / ``STATUS_IN_PROGRESS``), then fall back to the ``state``
    (``"in"`` -> IN_PROGRESS), else SCHEDULED. Live ESPN also uses ``"pre"`` /
    ``"post"`` states which both fall through to the name-based result.
    """
    if not isinstance(status_type, dict):
        return GameStatus.SCHEDULED
    name = status_type.get("name")
    if name in _STATUS_NAME_MAP:
        return _STATUS_NAME_MAP[name]
    if status_type.get("state") == "in":
        return GameStatus.IN_PROGRESS
    return GameStatus.SCHEDULED


def _normalize_competitor(competitor: Any) -> ScoreboardTeam:
    """Normalize one site-scoreboard competitor into a :class:`ScoreboardTeam`."""
    competitor = competitor if isinstance(competitor, dict) else {}
    team = competitor.get("team")
    team = team if isinstance(team, dict) else {}
    team_id = team.get("id")
    return ScoreboardTeam(
        espn_team_id=str(team_id) if team_id is not None else None,
        abbreviation=team.get("abbreviation"),
        score=_to_int_or_none(competitor.get("score")),
    )


def select_odds_item(odds_items: Any) -> dict[str, Any] | None:
    """Pick the preferred odds line from a site ``competition.odds[]`` list.

    Preference order (drift-proof — never hardcode a provider id):
    1. ``provider.name == "DraftKings"``
    2. ``provider.priority == 1``
    3. the first usable item present

    Returns ``None`` when there are no usable items.
    """
    if not isinstance(odds_items, list):
        return None

    priority_one: dict[str, Any] | None = None
    first_present: dict[str, Any] | None = None
    for item in odds_items:
        if not isinstance(item, dict):
            continue
        if first_present is None:
            first_present = item
        provider = item.get("provider")
        provider = provider if isinstance(provider, dict) else {}
        if provider.get("name") == _PREFERRED_PROVIDER_NAME:
            return item
        if priority_one is None and provider.get("priority") == 1:
            priority_one = item
    return priority_one or first_present


def _team_id_from_side(side: Any) -> str | None:
    """Extract the inline team id from a site ``home/awayTeamOdds`` side block."""
    side = side if isinstance(side, dict) else {}
    team = side.get("team")
    team = team if isinstance(team, dict) else {}
    team_id = team.get("id")
    return str(team_id) if team_id is not None else None


def normalize_odds(odds_item: Any) -> ScoreboardOdds | None:
    """Normalize one site inline odds item into :class:`ScoreboardOdds`.

    Reads the SITE inline shape (``provider.name``, ``spread``, ``overUnder``,
    ``home/awayTeamOdds`` with inline ``team.id`` + ``favorite``/``underdog``
    flags). Missing/partial fields degrade to ``None``; never raises. The
    ``spread`` is carried RAW (signed home-relative) — no abs() here.
    """
    if not isinstance(odds_item, dict):
        return None

    provider = odds_item.get("provider")
    provider = provider if isinstance(provider, dict) else {}
    provider_name = provider.get("name")
    # Capture the provider id from the SAME provider dict the name comes from
    # (the item ``select_odds_item`` chose) — NEVER from a hardcoded/other source
    # (provider ids drift; see espn-ingestion-strategy.md "Provider selection").
    # Coerce to str when present (mirrors the ``str(team_id)`` pattern); a
    # missing/None id degrades to None, just like the name.
    provider_id_raw = provider.get("id")
    provider_id = str(provider_id_raw) if provider_id_raw is not None else None

    spread = _to_float_or_none(odds_item.get("spread"))
    total = _to_float_or_none(odds_item.get("overUnder"))

    away = odds_item.get("awayTeamOdds")
    home = odds_item.get("homeTeamOdds")
    favorite_team_id: str | None = None
    underdog_team_id: str | None = None
    for side in (away, home):
        side_dict = side if isinstance(side, dict) else {}
        team_id = _team_id_from_side(side_dict)
        if side_dict.get("favorite") is True:
            favorite_team_id = team_id
        if side_dict.get("underdog") is True:
            underdog_team_id = team_id

    return ScoreboardOdds(
        provider=provider_name,
        provider_id=provider_id,
        spread=spread,
        total=total,
        favorite_team_id=favorite_team_id,
        underdog_team_id=underdog_team_id,
    )


def normalize_event(event: Any, season: int, week: int) -> ScoreboardGame | None:
    """Normalize a single site-scoreboard event into a :class:`ScoreboardGame`.

    Returns ``None`` for an unusable (non-dict) event. Reads kickoff/status/
    competitors/odds from ``competitions[0]`` (falling back to the event for
    date/status), distinguishing home/away by ``homeAway``. Never raises on a
    malformed shape — fields degrade to ``None``/defaults.
    """
    if not isinstance(event, dict):
        return None

    competitions = event.get("competitions")
    competition = competitions[0] if isinstance(competitions, list) and competitions else {}
    competition = competition if isinstance(competition, dict) else {}

    kickoff = competition.get("date") or event.get("date")

    status_obj = competition.get("status") or event.get("status") or {}
    status_obj = status_obj if isinstance(status_obj, dict) else {}
    status = _map_status(status_obj.get("type"))

    competitors = competition.get("competitors")
    competitors = competitors if isinstance(competitors, list) else []
    home: ScoreboardTeam | None = None
    away: ScoreboardTeam | None = None
    for competitor in competitors:
        side = competitor.get("homeAway") if isinstance(competitor, dict) else None
        if side == "home":
            home = _normalize_competitor(competitor)
        elif side == "away":
            away = _normalize_competitor(competitor)
    # An event with no competitors still yields placeholder empty teams so the
    # returned value object is well-formed (defensive; never raise).
    if home is None:
        home = ScoreboardTeam(espn_team_id=None, abbreviation=None, score=None)
    if away is None:
        away = ScoreboardTeam(espn_team_id=None, abbreviation=None, score=None)

    odds = normalize_odds(select_odds_item(competition.get("odds")))

    event_id = event.get("id")
    return ScoreboardGame(
        espn_event_id=str(event_id) if event_id is not None else None,
        season=season,
        week=week,
        kickoff_at=_parse_kickoff(kickoff),
        status=status,
        home=home,
        away=away,
        odds=odds,
    )


def normalize_scoreboard(payload: Any, season: int, week: int) -> list[ScoreboardGame]:
    """Normalize a parsed site-scoreboard dict into a list of games.

    Pure: takes the already-parsed JSON, returns normalized games. Skips any
    event that does not normalize. Never raises on a malformed payload — a
    missing/odd ``events`` field yields an empty list.
    """
    payload = payload if isinstance(payload, dict) else {}
    events = payload.get("events")
    events = events if isinstance(events, list) else []
    games: list[ScoreboardGame] = []
    for event in events:
        game = normalize_event(event, season, week)
        if game is not None:
            games.append(game)
    return games


# ---------------------------------------------------------------------------
# The adapter (thin impure shell over the pure normalizer)
# ---------------------------------------------------------------------------


class EspnScoreboardSource:
    """Production :class:`~app.scoreboard.port.ScoreboardSource` backed by ESPN.

    Structurally satisfies the port. :meth:`fetch_week` is the thin impure shell:
    it builds the site URL, fetches over stdlib ``urllib`` (raising
    :class:`ScoreboardFetchError` on failure), then delegates to the pure
    :func:`normalize_scoreboard`.
    """

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    def fetch_week(self, season: int, week: int) -> list[ScoreboardGame]:
        url = SITE_SCOREBOARD_URL.format(season=season, week=week)
        payload = _http_get_json(url, timeout=self._timeout)
        return normalize_scoreboard(payload, season, week)
