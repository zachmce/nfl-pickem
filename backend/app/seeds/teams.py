"""Idempotent production seeder for the 32-team NFL reference table.

The ``Team`` model is production reference data: ``Game`` rows FK to ``team.id``,
so the team table must be populated **before** any game ingest. Unlike the
dev-only fixture generator in ``backend/scripts/``, this seeder lives under
``app/`` because it writes production reference data.

The seeder is **idempotent** — it upserts each canonical team keyed on the unique
``espn_team_id`` column (never on the surrogate PK). Running it repeatedly leaves
exactly 32 rows and corrects any drifted abbreviation/display_name back to the
canonical value.

Run it from the ``backend/`` directory::

    cd backend
    python -m app.seeds.teams

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use ``python3 -m app.seeds.teams`` or the venv
> interpreter ``.venv/bin/python -m app.seeds.teams`` if ``python`` is not found.

Importing this module is side-effect-free: it opens no DB connection and performs
no I/O. All database work happens inside :func:`seed_teams` / :func:`main`.

The canonical ESPN ids and abbreviations below are verified against
``backend/tests/fixtures/nfl_2025_regular_season.json``. Note the gap: ESPN has
no team id 31 or 32; the ids jump from 30 (JAX) to 33 (BAL) and 34 (HOU). The
fixture does not carry display names, so the full display names here come from
this static canonical table.
"""

from __future__ import annotations

from sqlmodel import Session, select

from app.models import Team

# Canonical 32-team reference table: (espn_team_id, abbreviation, display_name).
# Verified against backend/tests/fixtures/nfl_2025_regular_season.json.
NFL_TEAMS: tuple[tuple[int, str, str], ...] = (
    (1, "ATL", "Atlanta Falcons"),
    (2, "BUF", "Buffalo Bills"),
    (3, "CHI", "Chicago Bears"),
    (4, "CIN", "Cincinnati Bengals"),
    (5, "CLE", "Cleveland Browns"),
    (6, "DAL", "Dallas Cowboys"),
    (7, "DEN", "Denver Broncos"),
    (8, "DET", "Detroit Lions"),
    (9, "GB", "Green Bay Packers"),
    (10, "TEN", "Tennessee Titans"),
    (11, "IND", "Indianapolis Colts"),
    (12, "KC", "Kansas City Chiefs"),
    (13, "LV", "Las Vegas Raiders"),
    (14, "LAR", "Los Angeles Rams"),
    (15, "MIA", "Miami Dolphins"),
    (16, "MIN", "Minnesota Vikings"),
    (17, "NE", "New England Patriots"),
    (18, "NO", "New Orleans Saints"),
    (19, "NYG", "New York Giants"),
    (20, "NYJ", "New York Jets"),
    (21, "PHI", "Philadelphia Eagles"),
    (22, "ARI", "Arizona Cardinals"),
    (23, "PIT", "Pittsburgh Steelers"),
    (24, "LAC", "Los Angeles Chargers"),
    (25, "SF", "San Francisco 49ers"),
    (26, "SEA", "Seattle Seahawks"),
    (27, "TB", "Tampa Bay Buccaneers"),
    (28, "WSH", "Washington Commanders"),
    (29, "CAR", "Carolina Panthers"),
    (30, "JAX", "Jacksonville Jaguars"),
    (33, "BAL", "Baltimore Ravens"),
    (34, "HOU", "Houston Texans"),
)


def seed_teams(session: Session) -> int:
    """Idempotently upsert all 32 canonical NFL teams, keyed on espn_team_id.

    For each canonical record, look up the existing ``Team`` by its unique
    ``espn_team_id``. If found, correct its ``abbreviation``/``display_name`` to
    the canonical values; otherwise insert a new row. The upsert is keyed on
    ``espn_team_id`` (the stable ESPN identity), never on the surrogate PK the
    seeder does not own. Commits once at the end.

    Returns the number of canonical teams processed (32).
    """
    for espn_team_id, abbreviation, display_name in NFL_TEAMS:
        existing = session.exec(select(Team).where(Team.espn_team_id == espn_team_id)).first()
        if existing is None:
            session.add(
                Team(
                    espn_team_id=espn_team_id,
                    abbreviation=abbreviation,
                    display_name=display_name,
                )
            )
        else:
            existing.abbreviation = abbreviation
            existing.display_name = display_name
            session.add(existing)

    session.commit()
    return len(NFL_TEAMS)


def main() -> None:
    """CLI entry point: open a task session, seed teams, print a summary."""
    # Imported here (not at module top) so importing this module never builds the
    # Postgres engine in app.db — keeps the module import side-effect-free.
    from app.db import task_session

    with task_session() as session:
        count = seed_teams(session)
    print(f"Seeded {count} NFL teams.")


if __name__ == "__main__":
    main()
