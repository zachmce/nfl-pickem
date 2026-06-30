"""Offline unit tests for the production NFL team seeder.

Two groups are covered:

1. **Canonical-table integrity** (pure, no DB): the ``NFL_TEAMS`` table has
   exactly 32 entries with unique espn ids and abbreviations, and every record
   carries a non-empty display_name/abbreviation within the model's length limits.
2. **Idempotent upsert** (in-memory SQLite, no Postgres/network): seeding twice
   yields exactly 32 rows, and a row with a stale abbreviation/display_name is
   corrected back to canonical on the next run.

These tests never touch the network or Postgres. The in-memory SQLite engine is
constructed inside the test — ``app.db`` is deliberately NOT imported because it
builds a Postgres engine at import time.

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && python -m unittest tests.test_seed_teams -v

No pytest dependency is required (none is configured for this project).
"""

from __future__ import annotations

import unittest

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Team
from app.seeds.teams import NFL_TEAMS, seed_teams

# Mirror the model's column limits (app.models.Team).
_MAX_ABBREVIATION = 10
_MAX_DISPLAY_NAME = 100


class CanonicalTableTests(unittest.TestCase):
    """Pure assertions about NFL_TEAMS — no database required."""

    def test_has_exactly_32_entries(self) -> None:
        self.assertEqual(len(NFL_TEAMS), 32)

    def test_espn_team_ids_are_unique(self) -> None:
        ids = [row[0] for row in NFL_TEAMS]
        self.assertEqual(len(set(ids)), 32)

    def test_abbreviations_are_unique(self) -> None:
        abbrs = [row[1] for row in NFL_TEAMS]
        self.assertEqual(len(set(abbrs)), 32)

    def test_every_row_is_a_complete_triple(self) -> None:
        for row in NFL_TEAMS:
            self.assertEqual(len(row), 3, row)
            espn_team_id, abbreviation, display_name = row
            self.assertIsInstance(espn_team_id, int)
            self.assertIsInstance(abbreviation, str)
            self.assertIsInstance(display_name, str)

    def test_abbreviations_non_empty_within_limit(self) -> None:
        for _id, abbreviation, _name in NFL_TEAMS:
            self.assertTrue(abbreviation, "abbreviation must be non-empty")
            self.assertLessEqual(len(abbreviation), _MAX_ABBREVIATION, abbreviation)

    def test_display_names_non_empty_within_limit(self) -> None:
        for _id, _abbr, display_name in NFL_TEAMS:
            self.assertTrue(display_name, "display_name must be non-empty")
            self.assertLessEqual(len(display_name), _MAX_DISPLAY_NAME, display_name)


class SeedTeamsUpsertTests(unittest.TestCase):
    """Idempotent upsert behavior against an in-memory SQLite database."""

    def setUp(self) -> None:
        # Fresh in-memory db per test; no Postgres, no app.db import.
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _row_count(self, session: Session) -> int:
        return len(session.exec(select(Team)).all())

    def test_seeds_32_teams_on_empty_db(self) -> None:
        with Session(self.engine) as session:
            count = seed_teams(session)
            self.assertEqual(count, 32)
            self.assertEqual(self._row_count(session), 32)

    def test_second_run_is_idempotent_no_duplicates(self) -> None:
        with Session(self.engine) as session:
            seed_teams(session)
            seed_teams(session)
            self.assertEqual(self._row_count(session), 32)

    def test_corrects_drifted_values_to_canonical(self) -> None:
        with Session(self.engine) as session:
            seed_teams(session)

            # Pick a known canonical record and corrupt its row in the db.
            espn_id, canon_abbr, canon_name = NFL_TEAMS[0]
            stale = session.exec(select(Team).where(Team.espn_team_id == espn_id)).first()
            assert stale is not None
            stale.abbreviation = "ZZZ"
            stale.display_name = "Stale Team Name"
            session.add(stale)
            session.commit()

            seed_teams(session)

            fixed = session.exec(select(Team).where(Team.espn_team_id == espn_id)).first()
            assert fixed is not None
            self.assertEqual(fixed.abbreviation, canon_abbr)
            self.assertEqual(fixed.display_name, canon_name)
            # Correction must not create a duplicate row.
            self.assertEqual(self._row_count(session), 32)


if __name__ == "__main__":
    unittest.main()
