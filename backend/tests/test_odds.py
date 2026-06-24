"""Offline tests for the source-agnostic odds reconciler + freeze predicate.

These exercise :mod:`app.services.odds` — the odds-poll core (a sibling to
:mod:`app.services.refresh`): a freeze predicate computed parallel to the pick
window (``freeze_at = min(noon-ET-Wednesday, pick_lock)``), an odds reconciler
that writes a game's frozen-shape line from a source's ``ScoreboardOdds`` block
and REFUSES to write a frozen game (write-path enforcement), an odds-active
needy predicate, and a week-level reconcile.

Everything runs OFFLINE, mirroring :mod:`tests.test_refresh_games`:

* an in-memory SQLite engine is constructed inside the test (no Postgres),
* :mod:`app.db` is deliberately NOT imported,
* the line source is a SYNTHETIC in-test ``ScoreboardSource`` whose odds block
  can MOVE between calls (the demo source serves static fixture odds, so it
  cannot prove line movement) — no network of any kind.

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && .venv/bin/python -m unittest tests.test_odds -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Game, GameStatus, Team, Week
from app.scoreboard.types import (
    ScoreboardGame,
    ScoreboardOdds,
    ScoreboardTeam,
)
from app.seeds.fixture_2025 import FIXTURE_PATH, import_fixture_2025
from app.seeds.teams import seed_teams
from app.services import odds as odds_service
from app.services.odds import (
    OddsResult,
    freeze_at,
    is_odds_frozen,
    odds_needy_weeks,
    reconcile_odds,
    reconcile_odds_games,
    reconcile_week_odds,
)

_ET = ZoneInfo("America/New_York")


def _load_fixture() -> dict:
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _team_map(session: Session) -> dict[int, int]:
    """espn_team_id (int) -> team.id, mirroring fixture_2025._build_team_map."""
    return {t.espn_team_id: t.id for t in session.exec(select(Team)).all()}


def _game(
    *,
    event_id: int,
    season: int,
    week: int,
    kickoff: datetime,
    status: GameStatus = GameStatus.SCHEDULED,
) -> Game:
    """A bare in-memory Game (not persisted) for freeze-math unit tests."""
    return Game(
        espn_event_id=event_id,
        week_id=1,
        season=season,
        week=week,
        home_team_id=1,
        away_team_id=2,
        kickoff_at=kickoff,
        status=status,
    )


class _MovingLineSource:
    """A synthetic ScoreboardSource whose odds block can MOVE between calls.

    It returns one configurable ScoreboardGame per fetched week (only the week it
    is told about), carrying a settable ScoreboardOdds block. ``set_line`` mutates
    the line so a second reconcile sees a different spread/favorite — proving line
    movement, which the static-fixture demo source cannot.
    """

    def __init__(
        self,
        *,
        season: int,
        week: int,
        event_id: int,
        odds: ScoreboardOdds | None,
        kickoff: datetime,
    ) -> None:
        self.season = season
        self.week = week
        self.event_id = event_id
        self.odds = odds
        self.kickoff = kickoff
        self.fetched: list[tuple[int, int]] = []

    def set_line(self, odds: ScoreboardOdds | None) -> None:
        self.odds = odds

    def fetch_week(self, season: int, week: int) -> list[ScoreboardGame]:
        self.fetched.append((season, week))
        if (season, week) != (self.season, self.week):
            return []
        return [
            ScoreboardGame(
                espn_event_id=str(self.event_id),
                season=season,
                week=week,
                kickoff_at=self.kickoff,
                status=GameStatus.SCHEDULED,
                home=ScoreboardTeam(None, None, score=None),
                away=ScoreboardTeam(None, None, score=None),
                odds=self.odds,
            )
        ]


class FreezeMathTests(unittest.TestCase):
    """freeze_at = min(noon-ET-Wednesday, pick_lock) + fail-loud guard."""

    def test_freeze_at_is_noon_et_wednesday_on_or_before_kickoff_edt(self) -> None:
        # A Sunday Sep kickoff (EDT). The Wednesday on/before is Sep 3, 2025;
        # noon ET that day is 16:00 UTC (EDT = UTC-4).
        kickoff = datetime(2025, 9, 7, 17, 0, tzinfo=timezone.utc)
        games = [_game(event_id=1, season=2025, week=1, kickoff=kickoff)]
        expected = datetime(2025, 9, 3, 12, 0, tzinfo=_ET).astimezone(timezone.utc)
        self.assertEqual(freeze_at(games), expected)
        # The Wednesday-noon must be on or before the kickoff.
        self.assertLessEqual(freeze_at(games), kickoff)

    def test_freeze_at_handles_est_offset(self) -> None:
        # A January kickoff (EST = UTC-5). Wednesday on/before Jan 11 2026 (Sun)
        # is Jan 7; noon ET that day is 17:00 UTC.
        kickoff = datetime(2026, 1, 11, 18, 0, tzinfo=timezone.utc)
        games = [_game(event_id=1, season=2025, week=18, kickoff=kickoff)]
        expected = datetime(2026, 1, 7, 12, 0, tzinfo=_ET).astimezone(timezone.utc)
        self.assertEqual(freeze_at(games), expected)
        # EST is UTC-5 so noon ET == 17:00 UTC (distinct from the EDT case).
        self.assertEqual(expected, datetime(2026, 1, 7, 17, 0, tzinfo=timezone.utc))

    def test_freeze_at_collapses_to_pick_lock_when_lock_before_noon_wed(self) -> None:
        # A game that kicks off Wednesday MORNING (before noon ET): noon-ET-Wed of
        # that same Wednesday would be AFTER kickoff, so min() must collapse
        # freeze_at to the pick_lock (the earliest kickoff = the close boundary).
        kickoff = datetime(2025, 9, 3, 14, 0, tzinfo=timezone.utc)  # 10:00 EDT Wed
        games = [_game(event_id=1, season=2025, week=1, kickoff=kickoff)]
        pick_lock = kickoff  # only one game -> earliest kickoff is the close
        self.assertEqual(freeze_at(games), pick_lock)
        self.assertLessEqual(freeze_at(games), pick_lock)

    def test_freeze_at_never_exceeds_pick_lock(self) -> None:
        # General invariant across a normal week.
        kickoff = datetime(2025, 9, 7, 17, 0, tzinfo=timezone.utc)
        games = [_game(event_id=1, season=2025, week=1, kickoff=kickoff)]
        from app.services.pick_window import compute_window

        pick_lock = compute_window(games).close_at
        self.assertLessEqual(freeze_at(games), pick_lock)

    def test_fail_loud_guard_raises_when_freeze_would_exceed_pick_lock(self) -> None:
        # Directly trip the internal guard: ask for a freeze_at computed against a
        # pick_lock that is BEFORE the noon-ET-Wed, with min() bypassed via the
        # guard-exposing helper. We assert the public path can never produce a
        # freeze_at > pick_lock, and the guard raises a labeled ValueError when
        # the precondition is violated.
        with self.assertRaises(ValueError):
            odds_service._guard_freeze_at_le_pick_lock(
                freeze=datetime(2025, 9, 4, 0, 0, tzinfo=timezone.utc),
                pick_lock=datetime(2025, 9, 3, 0, 0, tzinfo=timezone.utc),
            )


class IsOddsFrozenTests(unittest.TestCase):
    """is_odds_frozen: now vs freeze_at, plus the lines_frozen override."""

    def setUp(self) -> None:
        self.kickoff = datetime(2025, 9, 7, 17, 0, tzinfo=timezone.utc)
        self.games = [_game(event_id=1, season=2025, week=1, kickoff=self.kickoff)]
        self.freeze = freeze_at(self.games)

    def test_now_before_freeze_is_not_frozen(self) -> None:
        week = Week(season=2025, week=1, lines_frozen=False)
        before = self.freeze - timedelta(hours=1)
        self.assertFalse(is_odds_frozen(week, self.games, now=before))

    def test_now_at_or_after_freeze_is_frozen(self) -> None:
        week = Week(season=2025, week=1, lines_frozen=False)
        self.assertTrue(is_odds_frozen(week, self.games, now=self.freeze))
        after = self.freeze + timedelta(seconds=1)
        self.assertTrue(is_odds_frozen(week, self.games, now=after))

    def test_lines_frozen_override_is_frozen_regardless_of_now(self) -> None:
        week = Week(season=2025, week=1, lines_frozen=True)
        before = self.freeze - timedelta(days=30)
        self.assertTrue(is_odds_frozen(week, self.games, now=before))


class ReconcileOddsWriteTests(unittest.TestCase):
    """The write-path core: fixture-shape normalization + freeze refusal."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _load_fixture()
        cls.season = int(cls.fixture["metadata"]["season"])

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)
        with Session(self.engine) as session:
            seed_teams(session)
            import_fixture_2025(session)
            # Reset week 1 odds so we can prove the reconciler WRITES them.
            for g in session.exec(
                select(Game).where(
                    Game.season == self.season, Game.week == 1
                )
            ).all():
                g.status = GameStatus.SCHEDULED
                g.home_score = None
                g.away_score = None
                g.spread = None
                g.total = None
                g.favorite_team_id = None
                g.underdog_team_id = None
                g.odds_provider = None
                g.odds_frozen = False
                g.odds_captured_at = None
                session.add(g)
            session.commit()
        self.now = datetime(2025, 9, 1, 12, 0, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _two_team_ids(self, session: Session) -> tuple[int, int]:
        """Two real espn_team_ids (as strings) for fav/dog."""
        teams = session.exec(select(Team)).all()
        return teams[0].espn_team_id, teams[1].espn_team_id

    def test_reconcile_writes_fixture_shaped_line(self) -> None:
        with Session(self.engine) as session:
            tmap = _team_map(session)
            fav_espn, dog_espn = self._two_team_ids(session)
            row = session.exec(
                select(Game).where(
                    Game.season == self.season, Game.week == 1
                )
            ).first()
            src = ScoreboardOdds(
                provider="DraftKings",
                spread=-3.5,  # signed home-relative; stored positive magnitude
                total=44.5,
                favorite_team_id=str(fav_espn),
                underdog_team_id=str(dog_espn),
            )
            wrote = reconcile_odds(row, src, tmap, frozen=False, now=self.now)
            self.assertTrue(wrote)
            # Fixture-shape: positive Decimal magnitude + int FK ids.
            self.assertEqual(row.spread, Decimal("3.5"))
            self.assertEqual(row.total, Decimal("44.5"))
            self.assertEqual(row.favorite_team_id, tmap[int(fav_espn)])
            self.assertEqual(row.underdog_team_id, tmap[int(dog_espn)])
            self.assertIsInstance(row.favorite_team_id, int)
            self.assertEqual(row.odds_provider, "DraftKings")
            self.assertEqual(row.odds_captured_at, self.now)

    def test_moving_line_rewrites_changed_fields(self) -> None:
        with Session(self.engine) as session:
            tmap = _team_map(session)
            fav_espn, dog_espn = self._two_team_ids(session)
            row = session.exec(
                select(Game).where(
                    Game.season == self.season, Game.week == 1
                )
            ).first()
            reconcile_odds(
                row,
                ScoreboardOdds(
                    provider="DraftKings",
                    spread=-3.5,
                    total=44.5,
                    favorite_team_id=str(fav_espn),
                    underdog_team_id=str(dog_espn),
                ),
                tmap,
                frozen=False,
                now=self.now,
            )
            # The line MOVES: favorite flips, spread changes.
            later = self.now + timedelta(hours=2)
            wrote = reconcile_odds(
                row,
                ScoreboardOdds(
                    provider="DraftKings",
                    spread=2.5,  # now positive (away/other favored)
                    total=45.0,
                    favorite_team_id=str(dog_espn),
                    underdog_team_id=str(fav_espn),
                ),
                tmap,
                frozen=False,
                now=later,
            )
            self.assertTrue(wrote)
            self.assertEqual(row.spread, Decimal("2.5"))
            self.assertEqual(row.total, Decimal("45.0"))
            self.assertEqual(row.favorite_team_id, tmap[int(dog_espn)])
            self.assertEqual(row.underdog_team_id, tmap[int(fav_espn)])
            self.assertEqual(row.odds_captured_at, later)

    def test_frozen_refuses_to_write(self) -> None:
        with Session(self.engine) as session:
            tmap = _team_map(session)
            fav_espn, dog_espn = self._two_team_ids(session)
            row = session.exec(
                select(Game).where(
                    Game.season == self.season, Game.week == 1
                )
            ).first()
            wrote = reconcile_odds(
                row,
                ScoreboardOdds(
                    provider="DraftKings",
                    spread=-3.5,
                    total=44.5,
                    favorite_team_id=str(fav_espn),
                    underdog_team_id=str(dog_espn),
                ),
                tmap,
                frozen=True,  # write-path refusal
                now=self.now,
            )
            self.assertFalse(wrote)
            self.assertIsNone(row.spread)
            self.assertIsNone(row.favorite_team_id)
            self.assertIsNone(row.odds_provider)

    def test_unchanged_rerun_is_a_noop(self) -> None:
        with Session(self.engine) as session:
            tmap = _team_map(session)
            fav_espn, dog_espn = self._two_team_ids(session)
            row = session.exec(
                select(Game).where(
                    Game.season == self.season, Game.week == 1
                )
            ).first()
            src = ScoreboardOdds(
                provider="DraftKings",
                spread=-3.5,
                total=44.5,
                favorite_team_id=str(fav_espn),
                underdog_team_id=str(dog_espn),
            )
            reconcile_odds(row, src, tmap, frozen=False, now=self.now)
            # Compare Decimals by value: an identical second apply writes nothing.
            wrote = reconcile_odds(row, src, tmap, frozen=False, now=self.now)
            self.assertFalse(wrote)

    def test_none_odds_never_nulls_present_line(self) -> None:
        with Session(self.engine) as session:
            tmap = _team_map(session)
            fav_espn, dog_espn = self._two_team_ids(session)
            row = session.exec(
                select(Game).where(
                    Game.season == self.season, Game.week == 1
                )
            ).first()
            reconcile_odds(
                row,
                ScoreboardOdds(
                    provider="DraftKings",
                    spread=-3.5,
                    total=44.5,
                    favorite_team_id=str(fav_espn),
                    underdog_team_id=str(dog_espn),
                ),
                tmap,
                frozen=False,
                now=self.now,
            )
            # A None odds block must never null the present line.
            wrote = reconcile_odds(row, None, tmap, frozen=False, now=self.now)
            self.assertFalse(wrote)
            self.assertEqual(row.spread, Decimal("3.5"))
            self.assertEqual(row.favorite_team_id, tmap[int(fav_espn)])

    def test_none_fields_never_null_present_values(self) -> None:
        with Session(self.engine) as session:
            tmap = _team_map(session)
            fav_espn, dog_espn = self._two_team_ids(session)
            row = session.exec(
                select(Game).where(
                    Game.season == self.season, Game.week == 1
                )
            ).first()
            reconcile_odds(
                row,
                ScoreboardOdds(
                    provider="DraftKings",
                    spread=-3.5,
                    total=44.5,
                    favorite_team_id=str(fav_espn),
                    underdog_team_id=str(dog_espn),
                ),
                tmap,
                frozen=False,
                now=self.now,
            )
            # A line that omits spread/total/fav (None) keeps the present values.
            wrote = reconcile_odds(
                row,
                ScoreboardOdds(provider="DraftKings"),
                tmap,
                frozen=False,
                now=self.now,
            )
            self.assertFalse(wrote)
            self.assertEqual(row.spread, Decimal("3.5"))
            self.assertEqual(row.total, Decimal("44.5"))

    def test_unresolvable_team_id_is_skipped_not_crashed(self) -> None:
        with Session(self.engine) as session:
            tmap = _team_map(session)
            row = session.exec(
                select(Game).where(
                    Game.season == self.season, Game.week == 1
                )
            ).first()
            # A fav/dog id not in the team map: the fav/dog write is skipped, not
            # a crash. spread/total still apply.
            wrote = reconcile_odds(
                row,
                ScoreboardOdds(
                    provider="DraftKings",
                    spread=-3.5,
                    total=44.5,
                    favorite_team_id="999999",
                    underdog_team_id="888888",
                ),
                tmap,
                frozen=False,
                now=self.now,
            )
            self.assertTrue(wrote)
            self.assertEqual(row.spread, Decimal("3.5"))
            self.assertIsNone(row.favorite_team_id)


class ReconcileWeekAndNeedyTests(unittest.TestCase):
    """Week-level reconcile + the odds-active needy predicate + freeze-stop."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _load_fixture()
        cls.season = int(cls.fixture["metadata"]["season"])

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)
        with Session(self.engine) as session:
            seed_teams(session)
            import_fixture_2025(session)
            for g in session.exec(select(Game)).all():
                g.status = GameStatus.SCHEDULED
                g.home_score = None
                g.away_score = None
                session.add(g)
            session.commit()
        # A `now` well before week 1's freeze so week 1 is odds-active.
        self.now_open = datetime(2025, 8, 1, 12, 0, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _week1_event_and_kickoff(self, session: Session) -> tuple[int, datetime]:
        row = session.exec(
            select(Game).where(Game.season == self.season, Game.week == 1)
        ).first()
        ko = row.kickoff_at
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        return row.espn_event_id, ko

    def test_needy_selects_unfrozen_week_excludes_frozen(self) -> None:
        with Session(self.engine) as session:
            all_games = list(session.exec(select(Game)).all())
            week_rows = {
                w.week: w
                for w in session.exec(
                    select(Week).where(Week.season == self.season)
                ).all()
            }
            from app.services.refresh import group_games_by_week

            by_week = group_games_by_week(all_games)
            needy = odds_needy_weeks(
                by_week, week_rows, now=self.now_open
            )
            self.assertIn((self.season, 1), set(needy))

            # Freeze week 1 explicitly -> excluded from needy.
            week_rows[1].lines_frozen = True
            needy_after = odds_needy_weeks(
                by_week, week_rows, now=self.now_open
            )
            self.assertNotIn((self.season, 1), set(needy_after))

    def test_reconcile_week_writes_then_freeze_stops_updates(self) -> None:
        with Session(self.engine) as session:
            tmap = _team_map(session)
            event_id, kickoff = self._week1_event_and_kickoff(session)
            teams = session.exec(select(Team)).all()
            fav_espn, dog_espn = teams[0].espn_team_id, teams[1].espn_team_id

            week_games = list(
                session.exec(
                    select(Game).where(
                        Game.season == self.season, Game.week == 1
                    )
                ).all()
            )
            week_row = session.exec(
                select(Week).where(
                    Week.season == self.season, Week.week == 1
                )
            ).first()

            src = _MovingLineSource(
                season=self.season,
                week=1,
                event_id=event_id,
                odds=ScoreboardOdds(
                    provider="DraftKings",
                    spread=-3.5,
                    total=44.5,
                    favorite_team_id=str(fav_espn),
                    underdog_team_id=str(dog_espn),
                ),
                kickoff=kickoff,
            )
            fetched = src.fetch_week(self.season, 1)

            result = reconcile_week_odds(
                session,
                fetched,
                week_games,
                week_row=week_row,
                now=self.now_open,
                team_map=tmap,
            )
            self.assertIsInstance(result, OddsResult)
            self.assertEqual(result.games_updated, 1)
            session.commit()

            target = session.exec(
                select(Game).where(Game.espn_event_id == event_id)
            ).first()
            self.assertEqual(target.spread, Decimal("3.5"))

            # Now FREEZE the week and MOVE the line -> the row must NOT change.
            src.set_line(
                ScoreboardOdds(
                    provider="DraftKings",
                    spread=7.5,
                    total=50.0,
                    favorite_team_id=str(dog_espn),
                    underdog_team_id=str(fav_espn),
                )
            )
            fetched2 = src.fetch_week(self.season, 1)
            frozen_now = freeze_at(week_games) + timedelta(hours=1)
            result2 = reconcile_week_odds(
                session,
                fetched2,
                week_games,
                week_row=week_row,
                now=frozen_now,
                team_map=tmap,
            )
            self.assertEqual(result2.games_updated, 0)
            self.assertIn(1, [w for _s, w in result2.frozen_weeks])
            session.commit()
            target2 = session.exec(
                select(Game).where(Game.espn_event_id == event_id)
            ).first()
            # Frozen: keeps its earlier value (3.5), not the moved 7.5.
            self.assertEqual(target2.spread, Decimal("3.5"))

    def test_reconcile_games_entry_writes_and_is_idempotent(self) -> None:
        with Session(self.engine) as session:
            tmap = _team_map(session)
            event_id, kickoff = self._week1_event_and_kickoff(session)
            teams = session.exec(select(Team)).all()
            fav_espn, dog_espn = teams[0].espn_team_id, teams[1].espn_team_id
            src = _MovingLineSource(
                season=self.season,
                week=1,
                event_id=event_id,
                odds=ScoreboardOdds(
                    provider="DraftKings",
                    spread=-3.5,
                    total=44.5,
                    favorite_team_id=str(fav_espn),
                    underdog_team_id=str(dog_espn),
                ),
                kickoff=kickoff,
            )
            result = reconcile_odds_games(
                session, src, now=self.now_open, team_map=tmap
            )
            self.assertIsInstance(result, OddsResult)
            self.assertGreaterEqual(result.games_updated, 1)
            session.commit()

            target = session.exec(
                select(Game).where(Game.espn_event_id == event_id)
            ).first()
            self.assertEqual(target.spread, Decimal("3.5"))

            # Idempotent: a second identical run dirties nothing.
            second = reconcile_odds_games(
                session, src, now=self.now_open, team_map=tmap
            )
            self.assertEqual(second.games_updated, 0)
            self.assertFalse(session.dirty, session.dirty)
            self.assertFalse(session.new, session.new)

    def test_reconcile_skips_unmatched_event_id(self) -> None:
        with Session(self.engine) as session:
            tmap = _team_map(session)
            _event_id, kickoff = self._week1_event_and_kickoff(session)
            teams = session.exec(select(Team)).all()
            fav_espn, dog_espn = teams[0].espn_team_id, teams[1].espn_team_id
            # A source whose event id matches no row in week 1.
            src = _MovingLineSource(
                season=self.season,
                week=1,
                event_id=123456789,
                odds=ScoreboardOdds(
                    provider="DraftKings",
                    spread=-3.5,
                    total=44.5,
                    favorite_team_id=str(fav_espn),
                    underdog_team_id=str(dog_espn),
                ),
                kickoff=kickoff,
            )
            result = reconcile_odds_games(
                session, src, now=self.now_open, team_map=tmap
            )
            self.assertEqual(result.games_updated, 0)


class OddsServiceSourceAgnosticGuardTests(unittest.TestCase):
    """odds.py must not couple to config / ESPN / the demo gate."""

    def test_odds_module_is_source_agnostic(self) -> None:
        from pathlib import Path

        source = Path(odds_service.__file__).read_text(encoding="utf-8")
        self.assertNotIn("app.config", source)
        self.assertNotIn("scoreboard.espn", source)
        self.assertNotIn("IS_DEMO_DATA", source)


if __name__ == "__main__":
    unittest.main()
