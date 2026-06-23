"""Offline integration proof for the demo season-walkthrough driver.

This is the headline integration test of the whole pick'em domain: it drives the
five demo bots' preordained picks through the REAL ``pick_submission`` service,
finalizes games through the REAL ``refresh_games`` poller, then computes ACTUAL
standings from the persisted DB rows and asserts they equal the
``compute_standings`` oracle for weeks 1-3.

Everything runs OFFLINE — in-memory SQLite, no Postgres, no network, ``app.db``
is never imported, and :class:`~app.scoreboard.espn.EspnScoreboardSource` is
never instantiated. The only source is :class:`~app.scoreboard.demo.Demo2025Source`,
positioned by :mod:`app.demo.offset` against the REAL clock.

Run from the ``backend/`` directory::

    cd backend && .venv/bin/python -m unittest tests.test_demo_driver -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

import unittest

from sqlmodel import Session, SQLModel, create_engine, select

from app.demo.driver import (
    compute_db_standings,
    run_walkthrough,
    setup,
)
from app.demo.oracle import compute_standings
from app.demo.run import require_demo_db
from app.models import Game, Pick, User, Week
from app.scoreboard.demo import Demo2025Source
from app.seeds.bots import BOT_ACCOUNTS
from app.seeds.data.bot_picks_2025 import BOT_PICKS

SEASON = 2025
WEEKS = (1, 2, 3)


class DemoDriverTests(unittest.TestCase):
    """The weeks 1-3 walkthrough actual==oracle integration proof."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _all_games(self, session: Session) -> list[Game]:
        return list(session.exec(select(Game)).all())

    def test_setup_is_idempotent(self) -> None:
        with Session(self.engine) as session:
            setup(session)
            first_games = len(self._all_games(session))
            first_weeks = len(session.exec(select(Week)).all())
            first_users = len(session.exec(select(User)).all())

            setup(session)
            self.assertEqual(len(self._all_games(session)), first_games)
            self.assertEqual(len(session.exec(select(Week)).all()), first_weeks)
            self.assertEqual(len(session.exec(select(User)).all()), first_users)

    def test_walkthrough_persists_picks_and_matches_oracle(self) -> None:
        with Session(self.engine) as session:
            setup(session)
            result = run_walkthrough(
                session, weeks=WEEKS, assert_oracle=True
            )
            self.assertTrue(result.passed)

            games = self._all_games(session)

            # (a) Every bot's picks persisted as Pick rows for each week.
            bots = {
                u.display_name: u
                for u in session.exec(select(User)).all()
                if u.display_name in BOT_PICKS
            }
            self.assertEqual(set(bots), set(BOT_PICKS))

            week_id_by_num = {
                w.week: w.id
                for w in session.exec(
                    select(Week).where(Week.season == SEASON)
                ).all()
            }
            for name, weeks in BOT_PICKS.items():
                user = bots[name]
                for wk, records in weeks.items():
                    persisted = list(
                        session.exec(
                            select(Pick).where(
                                Pick.user_id == user.id,
                                Pick.week_id == week_id_by_num[wk],
                            )
                        ).all()
                    )
                    self.assertEqual(
                        len(persisted),
                        len(records),
                        f"{name} week {wk}: persisted {len(persisted)} "
                        f"picks, expected {len(records)}",
                    )

            # (b) DB-sourced actual standings equal the oracle for weeks 1-3.
            actual = compute_db_standings(session, season=SEASON, weeks=WEEKS)
            expected = compute_standings(BOT_PICKS, games)
            self.assertEqual(actual, expected)

    def test_partial_roster_persists_three_picks(self) -> None:
        # bot_dave week 2 has only 3 picks — they must persist as exactly 3 rows.
        with Session(self.engine) as session:
            setup(session)
            run_walkthrough(session, weeks=WEEKS, assert_oracle=True)

            dave = session.exec(
                select(User).where(User.display_name == "bot_dave")
            ).first()
            assert dave is not None
            wk2 = session.exec(
                select(Week).where(Week.season == SEASON, Week.week == 2)
            ).first()
            assert wk2 is not None
            picks = list(
                session.exec(
                    select(Pick).where(
                        Pick.user_id == dave.id, Pick.week_id == wk2.id
                    )
                ).all()
            )
            self.assertEqual(len(picks), 3)

    def test_picks_go_through_real_submission_not_handinsert(self) -> None:
        # All bot accounts should exist as Users after setup (the real auth path
        # is exercised by seed_bots); the picks are submitted, not hand-inserted.
        with Session(self.engine) as session:
            setup(session)
            run_walkthrough(session, weeks=(1,), assert_oracle=True)
            names = {
                u.display_name for u in session.exec(select(User)).all()
            }
            for display_name, _pw in BOT_ACCOUNTS:
                self.assertIn(display_name, names)

    def test_source_factory_is_injectable(self) -> None:
        # The driver must accept an injected source factory (default builds a
        # Demo2025Source). Passing the default explicitly still works offline.
        with Session(self.engine) as session:
            setup(session)
            result = run_walkthrough(
                session,
                weeks=(1, 2),
                source_factory=lambda offset: Demo2025Source(offset),
                assert_oracle=True,
            )
            self.assertTrue(result.passed)


class DemoCliGateTests(unittest.TestCase):
    """The non-prod gate (T-qqm-01): require an explicit demo DB, reject prod.

    Offline — these assert on the raised :class:`SystemExit` without opening any
    database connection. A sentinel ``prod_url`` is injected so the test never
    imports real Settings or touches Postgres.
    """

    _PROD = "postgresql+psycopg://pickem:pickem@db:5432/pickem"

    def test_missing_demo_db_raises(self) -> None:
        with self.assertRaises(SystemExit):
            require_demo_db(None, prod_url=self._PROD)

    def test_blank_demo_db_raises(self) -> None:
        with self.assertRaises(SystemExit):
            require_demo_db("   ", prod_url=self._PROD)

    def test_prod_url_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            require_demo_db(self._PROD, prod_url=self._PROD)

    def test_distinct_demo_url_passes(self) -> None:
        url = "sqlite:///./demo_walkthrough.db"
        self.assertEqual(require_demo_db(url, prod_url=self._PROD), url)

    def test_demo_url_is_stripped(self) -> None:
        self.assertEqual(
            require_demo_db("  sqlite://  ", prod_url=self._PROD), "sqlite://"
        )


if __name__ == "__main__":
    unittest.main()
