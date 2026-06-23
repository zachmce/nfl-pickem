"""Offline tests for the preordained bot-picks dataset.

These tests prove the static dataset in :mod:`app.seeds.data.bot_picks_2025` is
consistent with the real 2025 fixture and legal under the roster rules — fully
offline (in-memory SQLite, no Postgres, no network, no ``app.db`` import). Like
``tests/test_scoring.py`` they seed teams then import the real fixture to get
real ``Game`` rows.

They assert:

* **Real-game** — every pick's ``espn_event_id`` resolves to a seeded ``Game``
  AND that game is in the pick's stated week.
* **Roster legality** — for every bot/week, building ``Pick`` instances from the
  dataset and validating them via :func:`app.services.pick_validation.validate_roster`
  yields ``ok is True``.
* **Partial coverage** — at least one roster is partial (fewer than five picks),
  so downstream partial scoring is actually exercised.

Run from the ``backend/`` directory::

    cd backend && .venv/bin/python -m unittest tests.test_bot_picks_dataset -v
"""

from __future__ import annotations

import unittest

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Game, Pick
from app.seeds.data.bot_picks_2025 import BOT_PICKS
from app.seeds.fixture_2025 import import_fixture_2025
from app.seeds.teams import seed_teams
from app.services.pick_validation import validate_roster


class BotPicksDatasetTests(unittest.TestCase):
    """Real-game + roster-legality checks for every bot/week."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _seed(self, session: Session) -> dict[int, Game]:
        seed_teams(session)
        import_fixture_2025(session)
        # Index real seeded games by their stable espn_event_id.
        return {g.espn_event_id: g for g in session.exec(select(Game)).all()}

    def test_every_pick_resolves_to_real_game_in_its_week(self) -> None:
        with Session(self.engine) as session:
            games_by_event = self._seed(session)
            for display_name, weeks in BOT_PICKS.items():
                for week_number, picks in weeks.items():
                    for bp in picks:
                        game = games_by_event.get(bp.espn_event_id)
                        self.assertIsNotNone(
                            game,
                            f"{display_name} wk{week_number}: event "
                            f"{bp.espn_event_id} not in fixture",
                        )
                        self.assertEqual(
                            game.week,
                            week_number,
                            f"{display_name}: event {bp.espn_event_id} is in "
                            f"week {game.week}, dataset says week {week_number}",
                        )

    def test_every_roster_passes_validate_roster(self) -> None:
        with Session(self.engine) as session:
            games_by_event = self._seed(session)
            # validate_roster looks games up by pick.game_id (the PK).
            games_by_pk = {g.id: g for g in games_by_event.values()}
            for display_name, weeks in BOT_PICKS.items():
                for week_number, bot_picks in weeks.items():
                    picks = [
                        Pick(
                            user_id=1,
                            game_id=games_by_event[bp.espn_event_id].id,
                            week_id=games_by_event[bp.espn_event_id].week_id,
                            pick_type=bp.pick_type,
                            is_mortal_lock=bp.is_mortal_lock,
                        )
                        for bp in bot_picks
                    ]
                    result = validate_roster(picks, games_by_pk)
                    self.assertTrue(
                        result.ok,
                        f"{display_name} wk{week_number} invalid: "
                        f"{[v.code.value for v in result.violations]}",
                    )

    def test_at_least_one_partial_roster_exists(self) -> None:
        # A full roster is 5 picks; a partial roster has fewer.
        partials = [
            (name, wk, len(picks))
            for name, weeks in BOT_PICKS.items()
            for wk, picks in weeks.items()
            if len(picks) < 5
        ]
        self.assertTrue(partials, "no partial roster present in the dataset")

    def test_bot_keys_match_seed_accounts(self) -> None:
        from app.seeds.bots import BOT_ACCOUNTS

        self.assertEqual(
            set(BOT_PICKS.keys()),
            {name for name, _ in BOT_ACCOUNTS},
        )


if __name__ == "__main__":
    unittest.main()
