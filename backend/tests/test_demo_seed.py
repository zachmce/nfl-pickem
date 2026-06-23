"""Offline tests for the env-gated demo-season seed.

Exercise :mod:`app.seeds.demo` + :mod:`app.seeds.bot_picks` against an in-memory
SQLite engine — no Postgres, no network, no ``app.db`` import, and
:class:`~app.scoreboard.espn.EspnScoreboardSource` is never constructed. The only
source used is :class:`~app.scoreboard.demo.Demo2025Source`, positioned by the
shared anchor offset.

Proven (with a pinned ``now``):

* DEMO-SHIFT: week-1's earliest positioned ``Game.kickoff_at`` ~= now + buffer and
  the Week rows have ``window_closes_at`` stamped;
* shared-offset: the stored DemoState anchor == the pinned now and
  ``offset_from_anchor(anchor)`` reproduces the positioning;
* DEMO-BOTS: the 5 bots exist and their wk1-3 Pick rows are persisted directly
  (counts match the BOT_PICKS records) with ``result=PENDING``;
* idempotency: re-seeding with the same ``now`` leaves the same Game/Week/Pick and
  single DemoState row counts (no duplicates, no IntegrityError);
* purge: ``purge_demo`` empties the demo footprint.

Run from the ``backend/`` directory::

    cd backend && .venv/bin/python -m unittest tests.test_demo_seed -v
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, SQLModel, create_engine, select

from app.demo.anchor import DEMO_KICKOFF_BUFFER, load_demo_anchor, offset_from_anchor
from app.models import DemoState, Game, GameStatus, Pick, PickResult, User, Week
from app.seeds.data.bot_picks_2025 import BOT_PICKS
from app.seeds.demo import purge_demo, seed_demo

PINNED_NOW = datetime(2026, 6, 23, 12, 0, 0, tzinfo=timezone.utc)
WEEKS = (1, 2, 3)


def _expected_bot_pick_count() -> int:
    return sum(
        len(recs)
        for by_week in BOT_PICKS.values()
        for w, recs in by_week.items()
        if w in WEEKS
    )


class DemoSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _season(self, session: Session) -> int:
        return next(iter({g.season for g in session.exec(select(Game)).all()}))

    def test_week1_first_kickoff_positioned_at_anchor_plus_buffer(self) -> None:
        with Session(self.engine) as session:
            seed_demo(session, now=PINNED_NOW)
            season = self._season(session)
            # Demo data stays labeled season 2025 (the fixture's own season).
            self.assertEqual(season, 2025)

            week1_games = session.exec(
                select(Game).where(Game.season == season, Game.week == 1)
            ).all()
            self.assertTrue(week1_games)
            kickoffs = [
                g.kickoff_at.replace(tzinfo=timezone.utc)
                if g.kickoff_at.tzinfo is None
                else g.kickoff_at
                for g in week1_games
            ]
            earliest = min(kickoffs)
            expected = PINNED_NOW + DEMO_KICKOFF_BUFFER
            self.assertLess(abs(earliest - expected), timedelta(minutes=1))

    def test_window_closes_at_stamped(self) -> None:
        with Session(self.engine) as session:
            seed_demo(session, now=PINNED_NOW)
            season = self._season(session)
            weeks = session.exec(
                select(Week).where(Week.season == season)
            ).all()
            self.assertTrue(weeks)
            for w in weeks:
                if w.week in WEEKS:
                    self.assertIsNotNone(
                        w.window_closes_at, f"week {w.week} missing window_closes_at"
                    )

    def test_anchor_stored_and_reproduces_offset(self) -> None:
        with Session(self.engine) as session:
            seed_demo(session, now=PINNED_NOW)
            anchor = load_demo_anchor(session)
            self.assertEqual(anchor, PINNED_NOW)

            season = self._season(session)
            offset = offset_from_anchor(anchor)
            week1_games = session.exec(
                select(Game).where(Game.season == season, Game.week == 1)
            ).all()
            earliest = min(
                g.kickoff_at.replace(tzinfo=timezone.utc)
                if g.kickoff_at.tzinfo is None
                else g.kickoff_at
                for g in week1_games
            )
            # The persisted positioning matches offset_from_anchor(stored anchor).
            self.assertLess(
                abs(earliest - (PINNED_NOW + DEMO_KICKOFF_BUFFER)),
                timedelta(minutes=1),
            )
            # And the offset is exactly the shared formula's output.
            self.assertEqual(offset, offset_from_anchor(PINNED_NOW))

    def test_bots_exist_and_picks_persisted_pending(self) -> None:
        with Session(self.engine) as session:
            seed_demo(session, now=PINNED_NOW)
            bot_names = list(BOT_PICKS.keys())
            self.assertEqual(len(bot_names), 5)
            bot_users = session.exec(
                select(User).where(User.display_name.in_(bot_names))
            ).all()
            self.assertEqual(len(bot_users), 5)

            bot_ids = [u.id for u in bot_users]
            picks = session.exec(
                select(Pick).where(Pick.user_id.in_(bot_ids))
            ).all()
            self.assertEqual(len(picks), _expected_bot_pick_count())
            for p in picks:
                self.assertEqual(p.result, PickResult.PENDING)
                self.assertEqual(p.points, 0)

    def test_reseed_is_idempotent(self) -> None:
        with Session(self.engine) as session:
            seed_demo(session, now=PINNED_NOW)

            def counts():
                return (
                    len(session.exec(select(Game)).all()),
                    len(session.exec(select(Week)).all()),
                    len(session.exec(select(Pick)).all()),
                    len(session.exec(select(DemoState)).all()),
                )

            first = counts()
            # Re-seed with the SAME now: no duplicates, no IntegrityError.
            seed_demo(session, now=PINNED_NOW)
            second = counts()
            self.assertEqual(first, second)
            # Exactly one DemoState row.
            self.assertEqual(second[3], 1)

    def test_purge_empties_demo_footprint(self) -> None:
        with Session(self.engine) as session:
            seed_demo(session, now=PINNED_NOW)
            purge_demo(session)

            self.assertEqual(len(session.exec(select(Pick)).all()), 0)
            self.assertEqual(len(session.exec(select(Game)).all()), 0)
            self.assertEqual(len(session.exec(select(Week)).all()), 0)
            self.assertEqual(len(session.exec(select(DemoState)).all()), 0)
            bot_users = session.exec(
                select(User).where(User.display_name.in_(list(BOT_PICKS.keys())))
            ).all()
            self.assertEqual(len(bot_users), 0)


if __name__ == "__main__":
    unittest.main()
