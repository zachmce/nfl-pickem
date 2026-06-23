"""Offline end-to-end tests for the 2025 NFL fixture importer.

These tests exercise :mod:`app.seeds.fixture_2025` against the REAL fixture file
(``backend/tests/fixtures/nfl_2025_regular_season.json``) — the same file the
production importer targets. No synthetic mini-fixture is used.

Everything runs offline:

* an in-memory SQLite engine is constructed inside the test (no Postgres),
* ``app.db`` is deliberately NOT imported (it builds a Postgres engine at import
  time),
* there is no network access of any kind.

Teams are seeded first (the importer requires the team table populated, since
``Game`` rows FK to ``team.id``), then the fixture is imported and asserted.

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && python -m unittest tests.test_import_fixture_2025 -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use ``python3 -m unittest ...`` or the venv
> interpreter ``.venv/bin/python -m unittest ...``.

No pytest dependency is required (none is configured for this project).
"""

from __future__ import annotations

import json
import unittest
from decimal import Decimal

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Game, Team, Week
from app.seeds.fixture_2025 import (
    FIXTURE_PATH,
    TeamsNotSeededError,
    import_fixture_2025,
)
from app.seeds.teams import seed_teams


def _load_fixture() -> dict:
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


class FixtureImporterTests(unittest.TestCase):
    """End-to-end import of the real fixture into an in-memory SQLite db."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _load_fixture()
        cls.metadata = cls.fixture["metadata"]
        cls.games = cls.fixture["games"]
        cls.expected_games = cls.metadata["games_total"]
        cls.expected_weeks = len({g["week"] for g in cls.games})

    def setUp(self) -> None:
        # Fresh in-memory db per test; no Postgres, no app.db import.
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    # -- helpers -----------------------------------------------------------

    def _team_id_by_espn(self, session: Session, espn_team_id: int) -> int:
        team = session.exec(
            select(Team).where(Team.espn_team_id == espn_team_id)
        ).first()
        assert team is not None, f"team espn_team_id={espn_team_id} missing"
        return team.id

    def _first_game_with_odds(self) -> dict:
        for g in self.games:
            if g.get("odds") is not None:
                return g
        self.fail("fixture has no game with odds")

    def _first_game_without_odds(self) -> dict:
        for g in self.games:
            if g.get("odds") is None:
                return g
        self.fail("fixture has no game without odds")

    # -- tests -------------------------------------------------------------

    def test_imports_18_weeks_and_272_games(self) -> None:
        with Session(self.engine) as session:
            seed_teams(session)
            import_fixture_2025(session)

            week_count = len(session.exec(select(Week)).all())
            game_count = len(session.exec(select(Game)).all())

            self.assertEqual(week_count, self.expected_weeks)
            self.assertEqual(week_count, 18)
            self.assertEqual(game_count, self.expected_games)
            self.assertEqual(game_count, 272)

    def test_team_fks_resolved_for_known_game(self) -> None:
        sample = self.games[0]
        with Session(self.engine) as session:
            seed_teams(session)
            import_fixture_2025(session)

            game = session.exec(
                select(Game).where(
                    Game.espn_event_id == int(sample["espn_event_id"])
                )
            ).first()
            self.assertIsNotNone(game)

            expected_home = self._team_id_by_espn(
                session, int(sample["home"]["team_id"])
            )
            expected_away = self._team_id_by_espn(
                session, int(sample["away"]["team_id"])
            )
            self.assertEqual(game.home_team_id, expected_home)
            self.assertEqual(game.away_team_id, expected_away)
            self.assertEqual(game.home_score, sample["home"]["score"])
            self.assertEqual(game.away_score, sample["away"]["score"])

    def test_odds_snapshot_frozen_and_labeled(self) -> None:
        sample = self._first_game_with_odds()
        odds = sample["odds"]
        with Session(self.engine) as session:
            seed_teams(session)
            import_fixture_2025(session)

            game = session.exec(
                select(Game).where(
                    Game.espn_event_id == int(sample["espn_event_id"])
                )
            ).first()
            self.assertIsNotNone(game)

            # Positive magnitude regardless of the fixture's signed value.
            self.assertIsNotNone(game.spread)
            self.assertGreater(game.spread, 0)
            self.assertEqual(game.spread, Decimal(str(abs(odds["spread"]))))
            self.assertEqual(game.total, Decimal(str(odds["total"])))

            expected_fav = self._team_id_by_espn(
                session, int(odds["favorite_team_id"])
            )
            expected_dog = self._team_id_by_espn(
                session, int(odds["underdog_team_id"])
            )
            self.assertEqual(game.favorite_team_id, expected_fav)
            self.assertEqual(game.underdog_team_id, expected_dog)

            self.assertEqual(game.odds_provider, "ESPN BET")
            self.assertTrue(game.odds_frozen)
            self.assertIsNotNone(game.odds_captured_at)

    def test_game_without_odds_has_null_unfrozen_odds(self) -> None:
        sample = self._first_game_without_odds()
        with Session(self.engine) as session:
            seed_teams(session)
            import_fixture_2025(session)

            game = session.exec(
                select(Game).where(
                    Game.espn_event_id == int(sample["espn_event_id"])
                )
            ).first()
            self.assertIsNotNone(game)

            self.assertIsNone(game.spread)
            self.assertIsNone(game.total)
            self.assertIsNone(game.favorite_team_id)
            self.assertIsNone(game.underdog_team_id)
            self.assertIsNone(game.odds_provider)
            self.assertFalse(game.odds_frozen)

    def test_reimport_is_idempotent(self) -> None:
        with Session(self.engine) as session:
            seed_teams(session)
            import_fixture_2025(session)
            import_fixture_2025(session)

            week_count = len(session.exec(select(Week)).all())
            game_count = len(session.exec(select(Game)).all())
            self.assertEqual(week_count, 18)
            self.assertEqual(game_count, 272)

    def test_requires_teams_seeded(self) -> None:
        with Session(self.engine) as session:
            # No teams seeded — importer must refuse to insert orphan FKs.
            with self.assertRaises(TeamsNotSeededError):
                import_fixture_2025(session)


if __name__ == "__main__":
    unittest.main()
