"""Offline, idempotent seed that imports the packaged 2025 NFL season data.

This loads ``backend/app/seeds/data/nfl_2025_regular_season.json`` into the
database as :class:`~app.models.Week` and :class:`~app.models.Game` rows,
including the embedded ESPN BET odds snapshot. It gives the scoring engine real
2025 ground-truth data (final scores + ESPN BET stand-in lines) to run against
**without any ESPN HTTP**.

This is the offline season-data seed, distinct from the future production
live-ingest worker:

* It performs **no network I/O of any kind** — it only reads a local JSON file.
* Importing this module is side-effect-free: it opens no DB connection and reads
  no files at import time. All work happens inside :func:`import_fixture_2025` /
  :func:`main`.
* It requires the team table to be seeded first (run ``python -m
  app.seeds.teams``); ``Game`` rows FK to ``team.id`` and team identity is
  resolved by ``espn_team_id``. Importing against an unseeded team table raises
  :class:`TeamsNotSeededError` rather than inserting orphan FKs.

Run it from the ``backend/`` directory::

    cd backend
    python -m app.seeds.fixture_2025

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use ``python3 -m app.seeds.fixture_2025`` or the venv
> interpreter ``.venv/bin/python -m app.seeds.fixture_2025`` if ``python`` is not
> found.

Idempotency: every upsert is keyed on a **stable natural key** (never the
surrogate PK) — ``Game`` on its unique ``espn_event_id`` and ``Week`` on
``(season, week)``. Re-running leaves exactly the same row counts and re-applies
the same field values.

A note on the odds: the fixture's ``spread`` is signed relative to the home team
(negative = home favored). The :class:`~app.models.Game` model stores ``spread``
as a **positive** half-point magnitude the favorite must cover, with direction
carried by ``favorite_team_id`` / ``underdog_team_id``. We therefore store
``abs(spread)`` and never persist a negative value. The lines are labeled
honestly as ``odds_provider = "ESPN BET"`` (they are ESPN BET stand-ins, NOT
DraftKings lines).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from sqlmodel import Session, select

from app.models import Game, GameStatus, Team, Week

# Resolve the season-data file relative to this module so the seed works
# regardless of the current working directory. This file lives at
# ``backend/app/seeds/fixture_2025.py`` and the data ships alongside it at
# ``backend/app/seeds/data/nfl_2025_regular_season.json``.
FIXTURE_PATH: Path = Path(__file__).parent / "data" / "nfl_2025_regular_season.json"

ODDS_PROVIDER = "ESPN BET"

# Map the fixture's ESPN status name -> our GameStatus. Only STATUS_FINAL occurs
# in this fixture, but map defensively for future fixtures.
_STATUS_NAME_MAP = {
    "STATUS_FINAL": GameStatus.FINAL,
    "STATUS_IN_PROGRESS": GameStatus.IN_PROGRESS,
}


class TeamsNotSeededError(RuntimeError):
    """Raised when a fixture team_id has no matching seeded ``Team`` row.

    The team table must be populated before importing games. Run
    ``python -m app.seeds.teams`` first.
    """


@dataclass(frozen=True)
class ImportResult:
    """Summary of an import run."""

    week_count: int
    game_count: int


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_kickoff(value: str | None) -> datetime | None:
    """Parse the fixture's ISO kickoff (e.g. ``2025-09-05T00:20Z``).

    The trailing ``Z`` (and the missing seconds) are normalized so the result is
    timezone-aware via ``datetime.fromisoformat``.
    """
    if not value:
        return None
    normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
    return datetime.fromisoformat(normalized)


def _map_status(status: dict | None) -> GameStatus:
    """Map the fixture ``status`` block to a :class:`GameStatus` (defensive)."""
    if not status:
        return GameStatus.SCHEDULED
    name = status.get("name")
    if name in _STATUS_NAME_MAP:
        return _STATUS_NAME_MAP[name]
    if status.get("state") == "in":
        return GameStatus.IN_PROGRESS
    return GameStatus.SCHEDULED


def _load_fixture(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _build_team_map(session: Session) -> dict[int, int]:
    """Map ``espn_team_id`` (int) -> ``team.id`` for all seeded teams."""
    return {team.espn_team_id: team.id for team in session.exec(select(Team)).all()}


def _resolve_team(team_map: dict[int, int], espn_team_id_str: str) -> int:
    """Resolve a fixture (string) team_id to a seeded ``team.id`` PK.

    Raises :class:`TeamsNotSeededError` if the team is not present, naming the
    missing ``espn_team_id`` and instructing the caller to seed teams first.
    """
    espn_team_id = int(espn_team_id_str)
    team_id = team_map.get(espn_team_id)
    if team_id is None:
        raise TeamsNotSeededError(
            f"No seeded Team for espn_team_id={espn_team_id}. "
            "Run `python -m app.seeds.teams` before importing the fixture."
        )
    return team_id


def import_fixture_2025(session: Session, *, path: Path | None = None) -> ImportResult:
    """Idempotently import the 2025 NFL fixture into Week + Game rows.

    Reads the packaged season-data JSON (offline — no network), upserts one ``Week`` per
    distinct ``(season, week)`` and one ``Game`` per ``espn_event_id``, resolving
    home/away (and favorite/underdog) FKs by ``espn_team_id``. Games carrying an
    ``odds`` object get the ESPN BET snapshot frozen (positive-magnitude spread,
    total, favorite/underdog FKs, ``odds_provider='ESPN BET'``,
    ``odds_frozen=True``); games with ``odds: null`` keep all odds fields NULL and
    ``odds_frozen=False``. Commits once at the end.

    :param session: an open SQLModel session.
    :param path: optional override for the data file path (defaults to
        :data:`FIXTURE_PATH`); used by tests to point at the same packaged file.
    :raises TeamsNotSeededError: if any referenced team is not seeded.
    :returns: an :class:`ImportResult` summarizing rows present after the run.
    """
    fixture = _load_fixture(path or FIXTURE_PATH)
    metadata = fixture["metadata"]
    games = fixture["games"]

    # Use the fixture's own season; do not hardcode 2025.
    season = int(metadata["season"])

    team_map = _build_team_map(session)
    captured_at = _utcnow()

    # 1. Upsert distinct weeks keyed on (season, week); build (s, w) -> week.id.
    week_id_by_key: dict[tuple[int, int], int] = {}
    for week_number in sorted({int(g["week"]) for g in games}):
        existing_week = session.exec(
            select(Week).where(Week.season == season, Week.week == week_number)
        ).first()
        if existing_week is None:
            week = Week(season=season, week=week_number)
            session.add(week)
            # Flush so the surrogate PK is assigned for Game.week_id below.
            session.flush()
        else:
            week = existing_week
        week_id_by_key[(season, week_number)] = week.id

    # 2. Upsert games keyed on the unique espn_event_id.
    for raw in games:
        espn_event_id = int(raw["espn_event_id"])
        week_number = int(raw["week"])

        home_team_id = _resolve_team(team_map, raw["home"]["team_id"])
        away_team_id = _resolve_team(team_map, raw["away"]["team_id"])

        game = session.exec(select(Game).where(Game.espn_event_id == espn_event_id)).first()
        if game is None:
            game = Game(espn_event_id=espn_event_id)
            session.add(game)

        game.espn_competition_id = int(raw["competition_id"])
        game.week_id = week_id_by_key[(season, week_number)]
        game.season = season
        game.week = week_number
        game.home_team_id = home_team_id
        game.away_team_id = away_team_id
        game.kickoff_at = _parse_kickoff(raw.get("kickoff"))
        game.status = _map_status(raw.get("status"))
        game.home_score = raw["home"].get("score")
        game.away_score = raw["away"].get("score")

        odds = raw.get("odds")
        if odds is not None:
            # Store positive magnitude; direction lives in favorite/underdog FKs.
            game.spread = Decimal(str(abs(odds["spread"])))
            game.total = Decimal(str(odds["total"]))
            game.favorite_team_id = _resolve_team(team_map, odds["favorite_team_id"])
            game.underdog_team_id = _resolve_team(team_map, odds["underdog_team_id"])
            game.odds_provider = ODDS_PROVIDER
            game.odds_frozen = True
            game.odds_captured_at = captured_at
        else:
            game.spread = None
            game.total = None
            game.favorite_team_id = None
            game.underdog_team_id = None
            game.odds_provider = None
            game.odds_frozen = False
            game.odds_captured_at = None

    session.commit()

    week_count = len(session.exec(select(Week)).all())
    game_count = len(session.exec(select(Game)).all())
    return ImportResult(week_count=week_count, game_count=game_count)


def main() -> None:
    """CLI entry point: open a task session, import the fixture, print a summary."""
    # Imported here (not at module top) so importing this module never builds the
    # Postgres engine in app.db — keeps the module import side-effect-free.
    from app.db import task_session

    with task_session() as session:
        result = import_fixture_2025(session)
    print(f"Imported 2025 fixture: {result.week_count} weeks, {result.game_count} games.")


if __name__ == "__main__":
    main()
