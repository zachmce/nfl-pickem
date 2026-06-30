"""Dev/test-only one-time 2025 NFL regular-season fixture generator.

This script performs a single offline-importable data pull of the 2025 NFL
regular season (seasontype=2, weeks 1..18) from public ESPN endpoints and writes
one normalized JSON fixture. Its sole purpose is to produce ground-truth 2025
data (real final scores plus ESPN BET stand-in odds) so the future scoring
engine can be tested *before* the production Game/Odds schema exists.

This is deliberately NOT part of any production ingest path. It lives under
``backend/scripts/`` (not ``backend/app/``) and imports nothing from the
application package, so it cannot couple to or drift with production code.

Why two ESPN endpoints:
- The site scoreboard gives denormalized games/scores/status/teams in one call
  per week. For *completed* 2025 games it carries NO odds at all.
- The core API retains historical ESPN BET lines per competition, which we fetch
  per game and attach. Those lines are honestly labeled ``ESPN BET`` (NOT real
  DraftKings) because 2025 has no DraftKings lines anywhere.

Run::

    cd backend && python -m scripts.gen_2025_fixture

Optional flags: ``--weeks 1-2`` (quick smoke run), ``--out PATH``,
``--delay SECONDS``. Re-running overwrites the fixture cleanly (idempotent).

Design: impure I/O (network, file writes) is isolated from pure normalization
functions so the normalization logic is unit-testable offline against captured
JSON shapes with no network access.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SITE_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    "?dates={year}&seasontype=2&week={week}"
)
# Single-line so host + the "odds" path segment stay greppable together.
CORE_ODDS_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{event_id}/competitions/{competition_id}/odds"  # noqa: E501

# ESPN BET provider id observed in core-API historical odds (id drifts across
# endpoints, so we resolve by name first and only fall back to this id).
ESPN_BET_PROVIDER_ID = 58
ESPN_BET_PROVIDER_NAME = "ESPN BET"

# A plain UA avoids the occasional default-client block without impersonating a
# browser. No credentials are ever sent — these are public, outbound-only GETs.
_USER_AGENT = "nfl-pickem-fixture-generator/1.0 (dev tooling; stdlib urllib)"

# Matches the numeric team id in a core-API team $ref URL, e.g.
# ".../teams/26?lang=en&region=us" -> "26".
_TEAM_REF_ID_RE = re.compile(r"/teams/(\d+)")


# ---------------------------------------------------------------------------
# HTTP layer (impure — not exercised by the offline unit tests)
# ---------------------------------------------------------------------------


def http_get_json(url: str, timeout: float = 20.0) -> Any:
    """GET ``url`` and parse the JSON body.

    Raises a clear RuntimeError (including the URL and status) on HTTP/URL
    errors so callers can decide whether the failure is fatal.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} fetching {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"network error fetching {url}: {exc.reason}") from exc
    return json.loads(body)


def fetch_scoreboard_week(year: int, week: int, timeout: float = 20.0) -> dict[str, Any]:
    """Fetch one week of the site scoreboard. A failure here is fatal."""
    url = SITE_SCOREBOARD_URL.format(year=year, week=week)
    return http_get_json(url, timeout=timeout)


def fetch_event_odds(
    event_id: str, competition_id: str, timeout: float = 20.0
) -> dict[str, Any] | None:
    """Fetch core-API odds for one competition.

    Returns ``None`` on any HTTP/URL error — odds are simply unavailable for that
    game and the run must continue rather than crash.
    """
    url = CORE_ODDS_URL.format(event_id=event_id, competition_id=competition_id)
    try:
        return http_get_json(url, timeout=timeout)
    except RuntimeError:
        return None


# ---------------------------------------------------------------------------
# Pure normalization (no network — unit-tested in tests/test_gen_2025_fixture.py)
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


def parse_team_id_from_ref(team_obj: Any) -> str | None:
    """Extract the numeric team id from a team object.

    Handles both the core-API ``$ref`` form (team carries a ``$ref`` URL ending
    in ``/teams/{id}``) and the inline form (team carries an ``id``). Returns the
    id as a string, or ``None`` for malformed/missing input — never raises.
    """
    if not isinstance(team_obj, dict):
        return None
    ref = team_obj.get("$ref")
    if isinstance(ref, str):
        match = _TEAM_REF_ID_RE.search(ref)
        if match:
            return match.group(1)
    inline_id = team_obj.get("id")
    if inline_id is not None:
        return str(inline_id)
    return None


def select_odds_item(odds_response: Any) -> dict[str, Any] | None:
    """Pick the ESPN BET odds line from a core-API odds response.

    Preference order:
    1. ``provider.name == "ESPN BET"``
    2. fallback: ``provider.id == 58`` (compared as both string and int)

    Returns ``None`` when neither is present or the response has no usable items.
    """
    if not isinstance(odds_response, dict):
        return None
    items = odds_response.get("items")
    if not isinstance(items, list):
        return None

    fallback: dict[str, Any] | None = None
    for item in items:
        if not isinstance(item, dict):
            continue
        provider = item.get("provider")
        if not isinstance(provider, dict):
            continue
        name = provider.get("name")
        if name == ESPN_BET_PROVIDER_NAME:
            return item
        provider_id = provider.get("id")
        if fallback is None and (
            provider_id == ESPN_BET_PROVIDER_ID or provider_id == str(ESPN_BET_PROVIDER_ID)
        ):
            fallback = item
    return fallback


def normalize_odds(odds_item: Any) -> dict[str, Any] | None:
    """Normalize one ESPN BET odds item into our flat odds shape.

    Returns ``{provider, spread, total, favorite_team_id, underdog_team_id}`` or
    ``None`` if the item is unusable. Prefers the ``close`` sub-line for
    spread/total when present, else the top-level values. Missing/partial fields
    degrade gracefully to ``None`` and never raise.
    """
    if not isinstance(odds_item, dict):
        return None

    close = odds_item.get("close")
    close = close if isinstance(close, dict) else {}

    # spread: prefer close.spread, else top-level spread.
    spread = _to_float_or_none(close.get("spread"))
    if spread is None:
        spread = _to_float_or_none(odds_item.get("spread"))

    # total: prefer close.total / close.overUnder, else top-level overUnder.
    total = _to_float_or_none(close.get("total"))
    if total is None:
        total = _to_float_or_none(close.get("overUnder"))
    if total is None:
        total = _to_float_or_none(odds_item.get("overUnder"))

    away = odds_item.get("awayTeamOdds")
    home = odds_item.get("homeTeamOdds")
    away = away if isinstance(away, dict) else {}
    home = home if isinstance(home, dict) else {}

    favorite_team_id: str | None = None
    underdog_team_id: str | None = None
    for side in (away, home):
        team_id = parse_team_id_from_ref(side.get("team"))
        if side.get("favorite") is True:
            favorite_team_id = team_id
        if side.get("underdog") is True:
            underdog_team_id = team_id

    return {
        "provider": ESPN_BET_PROVIDER_NAME,
        "spread": spread,
        "total": total,
        "favorite_team_id": favorite_team_id,
        "underdog_team_id": underdog_team_id,
    }


def _normalize_competitor(competitor: Any) -> dict[str, Any]:
    """Normalize a site-scoreboard competitor into {team_id, abbreviation, score}."""
    competitor = competitor if isinstance(competitor, dict) else {}
    team = competitor.get("team")
    team = team if isinstance(team, dict) else {}
    team_id = team.get("id")
    return {
        "team_id": str(team_id) if team_id is not None else None,
        "abbreviation": team.get("abbreviation"),
        "score": _to_int_or_none(competitor.get("score")),
    }


def normalize_game(event: Any, week: int) -> dict[str, Any] | None:
    """Normalize a single site-scoreboard event into our flat game shape.

    The returned ``odds`` field is initialized to ``None`` here; the orchestrator
    fills it later from the core API. Returns ``None`` for an unusable event.
    The competition id is derived from ``competitions[0].id`` and is NOT assumed
    to equal the event id.
    """
    if not isinstance(event, dict):
        return None

    competitions = event.get("competitions")
    competition = competitions[0] if isinstance(competitions, list) and competitions else {}
    competition = competition if isinstance(competition, dict) else {}

    # kickoff: competition date if present, else event date.
    kickoff = competition.get("date") or event.get("date")

    status_obj = competition.get("status") or event.get("status") or {}
    status_obj = status_obj if isinstance(status_obj, dict) else {}
    status_type = status_obj.get("type")
    status_type = status_type if isinstance(status_type, dict) else {}
    status = {
        "state": status_type.get("state"),
        "completed": bool(status_type.get("completed")),
        "name": status_type.get("name"),
    }

    # competitors[]: home/away distinguished by homeAway.
    competitors = competition.get("competitors")
    competitors = competitors if isinstance(competitors, list) else []
    home: dict[str, Any] | None = None
    away: dict[str, Any] | None = None
    for competitor in competitors:
        side = competitor.get("homeAway") if isinstance(competitor, dict) else None
        if side == "home":
            home = _normalize_competitor(competitor)
        elif side == "away":
            away = _normalize_competitor(competitor)

    return {
        "espn_event_id": str(event.get("id")) if event.get("id") is not None else None,
        "competition_id": str(competition.get("id")) if competition.get("id") is not None else None,
        "week": week,
        "kickoff": kickoff,
        "status": status,
        "home": home,
        "away": away,
        "odds": None,
    }


def build_fixture_dict(games: list[dict[str, Any]], counts: dict[str, int]) -> dict[str, Any]:
    """Assemble the final fixture document with a self-describing metadata block."""
    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "season": 2025,
            "season_type": 2,
            "source": (
                "ESPN site scoreboard (games/scores/status/teams) + ESPN core API "
                "(historical odds), pulled once for offline test data"
            ),
            "odds_provider": ESPN_BET_PROVIDER_NAME,
            "note": (
                "Odds are ESPN BET stand-in lines, NOT real DraftKings lines. "
                "2025 has no DraftKings odds available anywhere."
            ),
            "games_total": counts.get("games_total", 0),
            "games_with_odds": counts.get("games_with_odds", 0),
            "games_without_odds": counts.get("games_without_odds", 0),
        },
        "games": games,
    }


# ---------------------------------------------------------------------------
# Orchestration (impure)
# ---------------------------------------------------------------------------


def generate(
    year: int = 2025,
    weeks: list[int] | None = None,
    delay: float = 0.4,
) -> dict[str, Any]:
    """Pull every requested week and assemble the full fixture dict.

    For each week the scoreboard is fetched (fatal on failure), each event is
    normalized, then each game's odds are fetched from the core API, the ESPN BET
    line selected and normalized, and attached (or left ``None``). A polite
    ``delay`` is observed between odds requests.
    """
    if weeks is None:
        weeks = list(range(1, 19))

    games: list[dict[str, Any]] = []
    games_total = 0
    games_with_odds = 0
    games_without_odds = 0

    for week in weeks:
        scoreboard = fetch_scoreboard_week(year, week)
        events = scoreboard.get("events")
        events = events if isinstance(events, list) else []
        week_games = 0
        for event in events:
            game = normalize_game(event, week)
            if game is None:
                continue
            games_total += 1
            week_games += 1

            event_id = game.get("espn_event_id")
            competition_id = game.get("competition_id")
            odds: dict[str, Any] | None = None
            if event_id and competition_id:
                odds_response = fetch_event_odds(event_id, competition_id)
                odds = normalize_odds(select_odds_item(odds_response))
                time.sleep(delay)
            game["odds"] = odds
            if odds is not None:
                games_with_odds += 1
            else:
                games_without_odds += 1
            games.append(game)

        print(
            f"[week {week:>2}] {week_games} games "
            f"(running totals: {games_total} games, "
            f"{games_with_odds} with odds, {games_without_odds} without)",
            file=sys.stderr,
        )

    counts = {
        "games_total": games_total,
        "games_with_odds": games_with_odds,
        "games_without_odds": games_without_odds,
    }
    print(
        f"[done] {games_total} games total; "
        f"{games_with_odds} with ESPN BET odds, {games_without_odds} without",
        file=sys.stderr,
    )
    return build_fixture_dict(games, counts)


def write_fixture(fixture: dict[str, Any], out_path: Path) -> Path:
    """Pretty-print the fixture JSON to ``out_path``, overwriting any existing file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(fixture, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return out_path.resolve()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_weeks(spec: str) -> list[int]:
    """Parse a weeks spec like ``1-18``, ``3``, or ``1,2,5`` into a list of ints."""
    weeks: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
            weeks.extend(range(start, end + 1))
        else:
            weeks.append(int(part))
    return weeks


def _default_out_path() -> Path:
    """Default fixture path resolved relative to backend/ so it works from there."""
    # This file lives at backend/scripts/gen_2025_fixture.py; backend/ is two up.
    backend_dir = Path(__file__).resolve().parent.parent
    return backend_dir / "tests" / "fixtures" / "nfl_2025_regular_season.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a one-time 2025 NFL regular-season test fixture from ESPN "
            "(dev/test only; not a production ingest path)."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: backend/tests/fixtures/nfl_2025_regular_season.json).",
    )
    parser.add_argument(
        "--weeks",
        type=str,
        default="1-18",
        help="Weeks to pull, e.g. '1-18', '3', or '1,2,5' (default: 1-18).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="Polite delay in seconds between per-game odds requests (default: 0.4).",
    )
    args = parser.parse_args(argv)

    out_path = args.out if args.out is not None else _default_out_path()
    weeks = _parse_weeks(args.weeks)

    fixture = generate(year=2025, weeks=weeks, delay=args.delay)
    resolved = write_fixture(fixture, out_path)

    meta = fixture["metadata"]
    print(f"Wrote fixture to: {resolved}")
    print(
        f"Games: {meta['games_total']} "
        f"({meta['games_with_odds']} with odds, {meta['games_without_odds']} without)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
