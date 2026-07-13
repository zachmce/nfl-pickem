"""Idempotent startup upsert for the historical NFL game corpus.

This is the ALWAYS-LOADED startup seeder (mirrors :mod:`app.seeds.bot_picks` /
:mod:`app.seeds.teams` in style — importing it is side-effect-free, it operates
on a passed-in session and commits once). It reads the committed CSV artifact
``data/historical_games.csv`` (final games 1999 -> last completed season) and
upserts one :class:`~app.models.HistoricalGame` per game.

The upsert NEVER touches the network. The out-of-band regeneration of the CSV
artifact lives in :mod:`app.seeds.historical_gen` (dev-only, run once per season).

Team resolution is a two-hop lookup that FAILS LOUD on any miss:
``nflverse abbreviation`` -> :data:`NFLVERSE_ABBR_TO_ESPN` -> ``espn_team_id`` ->
seeded ``Team.id``. A malformed/unmapped row raises rather than silently dropping
a game (threat T-260713how-01).

Run it from the ``backend/`` directory::

    cd backend
    .venv/bin/python -m app.seeds.historical_games

Sign conventions (see :class:`~app.models.HistoricalGame`): ``result`` is COMPUTED
here as ``home_score - away_score`` (never read from a column), and ``spread_line``
keeps nflverse's home-perspective sign (positive => home favored).
"""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from sqlmodel import Session, select

from app.models import HistoricalGame, Team
from app.seeds.teams import NFL_TEAMS

# ---------------------------------------------------------------------------
# The single source of truth mapping a nflverse team abbreviation to its ESPN
# team id. The upsert resolves through it at READ time; the regeneration script
# (:mod:`app.seeds.historical_gen`) validates every abbreviation against it at
# WRITE time — so the two scripts can never drift.
#
# The 32 canonical entries are derived from ``teams.NFL_TEAMS`` (so a change to
# the team table flows through automatically). Layered on top are the nflverse
# abbreviations that DIFFER from ESPN's:
#
#   * ``WAS`` — nflverse spells Washington ``WAS``; ESPN/app uses ``WSH``.
#   * ``OAK`` — pre-2020 Oakland Raiders; same franchise as ``LV`` (espn 13).
#   * ``SD``  — pre-2017 San Diego Chargers; same franchise as ``LAC`` (espn 24).
#   * ``STL`` — pre-2016 St. Louis Rams; same franchise as ``LAR`` (espn 14).
#   * ``LA``  — nflverse's code for the (2016+) Los Angeles Rams; the app/ESPN
#     abbreviation is ``LAR``. Mapping both ``LA`` and ``LAR`` to espn 14 keeps
#     the pre/post-2016 Rams continuity intact.
#
# The canonical ``LAR``/``LAC``/``LV`` (from NFL_TEAMS) are kept too, so whichever
# spelling nflverse emits for a given season resolves. A truly unknown
# abbreviation is NEVER guessed — it fails loud in both scripts.
# ---------------------------------------------------------------------------
NFLVERSE_ABBR_TO_ESPN: dict[str, int] = {
    abbreviation: espn_team_id for espn_team_id, abbreviation, _display in NFL_TEAMS
}
NFLVERSE_ABBR_TO_ESPN.update(
    {
        "WAS": 28,  # nflverse Washington spelling (ESPN/app uses WSH)
        "OAK": 13,  # Oakland Raiders -> LV
        "SD": 24,  # San Diego Chargers -> LAC
        "STL": 14,  # St. Louis Rams -> LAR
        "LA": 14,  # nflverse Los Angeles Rams code -> LAR (pre/post-2016 continuity)
    }
)

# Default committed artifact (mirrors fixture_2025.FIXTURE_PATH).
ARTIFACT_PATH: Path = Path(__file__).parent / "data" / "historical_games.csv"


def load_historical_rows(path: Path | None = None) -> list[dict[str, str]]:
    """Read the committed CSV artifact into raw string-dict rows.

    Uses :class:`csv.DictReader`, so each row is a ``{column: str}`` dict exactly
    as :func:`seed_historical_games` expects. Defaults to :data:`ARTIFACT_PATH`;
    tests may pass their own path (or inject rows directly and never touch disk).
    """
    with open(path or ARTIFACT_PATH, newline="") as fh:
        return list(csv.DictReader(fh))


def _parse_decimal(cell: str | None) -> Decimal | None:
    """Parse a CSV cell into a Decimal, treating empty/None as SQL NULL."""
    if cell is None or cell == "":
        return None
    return Decimal(cell)


def seed_historical_games(session: Session, *, rows: Iterable[dict[str, str]] | None = None) -> int:
    """Idempotently upsert historical games from the artifact (or injected rows).

    When ``rows`` is ``None`` the committed artifact is loaded from disk; tests
    inject a small slice so they never touch disk or the network. A
    ``{espn_team_id -> Team.id}`` index is built from the seeded ``Team`` rows
    (mirrors bot_picks' ``event_to_game`` index).

    For each row the home/away abbreviation is resolved through
    :data:`NFLVERSE_ABBR_TO_ESPN` and then the team index to a ``Team.id`` — a miss
    at EITHER hop raises :class:`ValueError` naming the offending abbreviation or
    unseeded ``espn_team_id`` (FAIL LOUD; a game is never silently dropped).
    ``result`` is COMPUTED as ``home_score - away_score`` so the sign logic is
    owned and tested here.

    IDEMPOTENT (mirrors bot_picks / teams check-then-skip): a row whose
    ``nflverse_game_id`` already exists is skipped, so a re-run inserts nothing new.
    Commits once. Returns the total ``historical_game`` row count after the run.
    """
    if rows is None:
        rows = load_historical_rows()

    # espn_team_id -> Team.id for the seeded teams.
    team_index = {
        t.espn_team_id: t.id for t in session.exec(select(Team)).all() if t.id is not None
    }

    # Existing natural keys, for the idempotency guard.
    existing_ids = {gid for gid in session.exec(select(HistoricalGame.nflverse_game_id)).all()}

    def _resolve(abbr: str) -> int:
        espn_id = NFLVERSE_ABBR_TO_ESPN.get(abbr)
        if espn_id is None:
            raise ValueError(f"Unmapped nflverse team abbreviation: {abbr!r}")
        team_id = team_index.get(espn_id)
        if team_id is None:
            raise ValueError(f"Team abbreviation {abbr!r} -> espn_team_id {espn_id} is not seeded")
        return team_id

    for row in rows:
        nflverse_game_id = row["nflverse_game_id"]
        if nflverse_game_id in existing_ids:
            continue  # already persisted — skip (idempotent re-seed)

        home_team_id = _resolve(row["home_team"])
        away_team_id = _resolve(row["away_team"])
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        spread_line = _parse_decimal(row["spread_line"])
        if spread_line is None:
            raise ValueError(f"Missing spread_line for game {nflverse_game_id!r}")

        session.add(
            HistoricalGame(
                nflverse_game_id=nflverse_game_id,
                season=int(row["season"]),
                week=int(row["week"]),
                game_type=row["game_type"],
                gameday=date.fromisoformat(row["gameday"]),
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_score=home_score,
                away_score=away_score,
                result=home_score - away_score,  # home margin (owned + tested here)
                spread_line=spread_line,
                total_line=_parse_decimal(row.get("total_line")),
            )
        )
        existing_ids.add(nflverse_game_id)

    session.commit()
    return len(session.exec(select(HistoricalGame)).all())


def main() -> None:
    """CLI entry point: open a task session, upsert the artifact, print a summary."""
    # Imported here (not at module top) so importing this module never builds the
    # Postgres engine in app.db — keeps the module import side-effect-free.
    from app.db import task_session

    with task_session() as session:
        count = seed_historical_games(session)
    print(f"Seeded historical games: {count} rows present.")


if __name__ == "__main__":
    main()
