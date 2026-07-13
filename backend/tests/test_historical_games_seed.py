"""Offline unit tests for the historical-games startup upsert.

Four behaviors are covered against an in-memory SQLite database (no Postgres, no
network — ``app.db`` is deliberately NOT imported because it builds a Postgres
engine at import time):

1. **Idempotency** — seeding the same rows twice leaves the row count unchanged
   and creates no duplicate ``nflverse_game_id``.
2. **Team-map completeness** — every value in ``NFLVERSE_ABBR_TO_ESPN`` is a
   seeded ``Team``'s ``espn_team_id``; the required franchise/spelling aliases
   ``OAK/SD/STL/LA/WAS`` are all present as keys; and every home/away abbreviation
   in the sample rows resolves ``abbr -> espn_team_id -> Team.id``.
3. **Parse/sign** — the stored ``result`` equals ``home_score - away_score`` for a
   home win (positive) AND an away win (negative), ``spread_line`` keeps its
   home-perspective sign, and ``total_line`` is ``None`` for an empty CSV cell.
4. **Fail-loud** — a row with an unmapped abbreviation raises rather than being
   silently dropped.

Plus a pure assertion that the committed artifact's header matches the expected
trimmed columns (reads only the first line; no network).

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && python -m unittest tests.test_historical_games_seed -v

No pytest dependency is required (none is configured for this project).
"""

from __future__ import annotations

import unittest
from decimal import Decimal

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import HistoricalGame, Team
from app.seeds.historical_games import (
    ARTIFACT_PATH,
    NFLVERSE_ABBR_TO_ESPN,
    seed_historical_games,
)
from app.seeds.teams import seed_teams

# The exact trimmed artifact header (mirrors historical_gen.ARTIFACT_HEADER).
_EXPECTED_HEADER = (
    "nflverse_game_id,season,week,game_type,gameday,home_team,away_team,"
    "home_score,away_score,spread_line,total_line"
)

# ~4 rows shaped like csv.DictReader output (every value is a string). Covers a
# home-favored home win, a home-favored away win (negative result), an empty
# total_line, and franchise-alias abbreviations (SD/OAK).
SAMPLE_ROWS: list[dict[str, str]] = [
    {
        # home-favored home win: positive spread, positive result.
        "nflverse_game_id": "2011_01_DEN_KC",
        "season": "2011",
        "week": "1",
        "game_type": "REG",
        "gameday": "2011-09-11",
        "home_team": "KC",
        "away_team": "DEN",
        "home_score": "27",
        "away_score": "10",
        "spread_line": "3.5",
        "total_line": "45.5",
    },
    {
        # home-favored AWAY win: positive spread, NEGATIVE result.
        "nflverse_game_id": "2011_02_PHI_DAL",
        "season": "2011",
        "week": "2",
        "game_type": "REG",
        "gameday": "2011-09-18",
        "home_team": "DAL",
        "away_team": "PHI",
        "home_score": "17",
        "away_score": "24",
        "spread_line": "2.5",
        "total_line": "49",
    },
    {
        # empty total_line cell -> stored as NULL.
        "nflverse_game_id": "2011_03_SEA_SF",
        "season": "2011",
        "week": "3",
        "game_type": "REG",
        "gameday": "2011-09-25",
        "home_team": "SF",
        "away_team": "SEA",
        "home_score": "20",
        "away_score": "13",
        "spread_line": "6",
        "total_line": "",
    },
    {
        # BOTH teams are franchise aliases: SD -> LAC (24), OAK -> LV (13).
        "nflverse_game_id": "2011_04_OAK_SD",
        "season": "2011",
        "week": "4",
        "game_type": "REG",
        "gameday": "2011-10-02",
        "home_team": "SD",
        "away_team": "OAK",
        "home_score": "13",
        "away_score": "20",
        "spread_line": "-3",
        "total_line": "40",
    },
]


class SeedHistoricalGamesTests(unittest.TestCase):
    """Upsert behavior against an in-memory SQLite database."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)
        # Teams must be seeded first so the abbr -> espn_team_id -> Team.id resolves.
        with Session(self.engine) as session:
            seed_teams(session)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _count(self, session: Session) -> int:
        return len(session.exec(select(HistoricalGame)).all())

    def _by_id(self, session: Session, nflverse_game_id: str) -> HistoricalGame:
        row = session.exec(
            select(HistoricalGame).where(HistoricalGame.nflverse_game_id == nflverse_game_id)
        ).first()
        assert row is not None
        return row

    # --- (a) idempotency ---------------------------------------------------
    def test_second_run_is_idempotent_no_duplicates(self) -> None:
        with Session(self.engine) as session:
            first = seed_historical_games(session, rows=SAMPLE_ROWS)
            self.assertEqual(first, len(SAMPLE_ROWS))
            second = seed_historical_games(session, rows=SAMPLE_ROWS)
            self.assertEqual(second, len(SAMPLE_ROWS))
            self.assertEqual(self._count(session), len(SAMPLE_ROWS))

            ids = [r.nflverse_game_id for r in session.exec(select(HistoricalGame)).all()]
            self.assertEqual(len(ids), len(set(ids)))

    # --- (b) team-map completeness ----------------------------------------
    def test_map_values_are_all_seeded_espn_ids(self) -> None:
        with Session(self.engine) as session:
            seeded_espn_ids = {t.espn_team_id for t in session.exec(select(Team)).all()}
        for abbr, espn_id in NFLVERSE_ABBR_TO_ESPN.items():
            self.assertIn(espn_id, seeded_espn_ids, f"{abbr} -> {espn_id} not seeded")

    def test_required_aliases_present(self) -> None:
        for alias in ("OAK", "SD", "STL", "LA", "WAS"):
            self.assertIn(alias, NFLVERSE_ABBR_TO_ESPN)

    def test_sample_abbrs_resolve_through_two_hops(self) -> None:
        with Session(self.engine) as session:
            seed_historical_games(session, rows=SAMPLE_ROWS)
            team_ids = {t.id for t in session.exec(select(Team)).all()}
            for row in session.exec(select(HistoricalGame)).all():
                self.assertIn(row.home_team_id, team_ids)
                self.assertIn(row.away_team_id, team_ids)

    # --- (c) parse / sign --------------------------------------------------
    def test_result_sign_and_line_parsing(self) -> None:
        with Session(self.engine) as session:
            seed_historical_games(session, rows=SAMPLE_ROWS)

            # Positive result: home win, home-favored spread stays positive.
            home_win = self._by_id(session, "2011_01_DEN_KC")
            self.assertEqual(home_win.result, 27 - 10)
            self.assertGreater(home_win.result, 0)
            self.assertEqual(home_win.spread_line, Decimal("3.5"))
            self.assertEqual(home_win.total_line, Decimal("45.5"))

            # Negative result: away win, spread still home-perspective positive.
            away_win = self._by_id(session, "2011_02_PHI_DAL")
            self.assertEqual(away_win.result, 17 - 24)
            self.assertLess(away_win.result, 0)
            self.assertEqual(away_win.spread_line, Decimal("2.5"))

            # Empty total_line cell -> NULL.
            no_total = self._by_id(session, "2011_03_SEA_SF")
            self.assertIsNone(no_total.total_line)

            # Away-favored (negative home spread) alias game.
            alias_game = self._by_id(session, "2011_04_OAK_SD")
            self.assertEqual(alias_game.spread_line, Decimal("-3"))
            self.assertEqual(alias_game.result, 13 - 20)

    # --- (d) fail-loud -----------------------------------------------------
    def test_unmapped_abbreviation_raises(self) -> None:
        bogus = dict(SAMPLE_ROWS[0])
        bogus["nflverse_game_id"] = "2011_99_XXX_KC"
        bogus["away_team"] = "XXX"  # not in NFLVERSE_ABBR_TO_ESPN
        with Session(self.engine) as session:
            with self.assertRaises(ValueError):
                seed_historical_games(session, rows=[bogus])


class ArtifactHeaderTests(unittest.TestCase):
    """Pure assertion about the committed artifact header (no network)."""

    def test_artifact_header_matches_expected_columns(self) -> None:
        with open(ARTIFACT_PATH, newline="") as fh:
            header = fh.readline().rstrip("\r\n")
        self.assertEqual(header, _EXPECTED_HEADER)


if __name__ == "__main__":
    unittest.main()
