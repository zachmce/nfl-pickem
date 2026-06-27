"""Offline tests for the manual line-freeze service (:mod:`app.services.freeze`).

These exercise :func:`app.services.freeze.freeze_week` end-to-end against an
in-memory SQLite db, driven by a SYNTHETIC ``ScoreboardSource`` that yields
hand-built :class:`~app.scoreboard.types.ScoreboardGame` / ``ScoreboardOdds``
value objects — there is NO live ESPN and NO network access of any kind. The
synthetic source hands the service ALREADY-NORMALIZED odds (the drift-proof
DraftKings SELECTION itself is the adapter's job, covered in
``test_scoreboard_espn``); here we assert the freeze service snapshots the
handed provider name+id and locks the week.

Everything runs OFFLINE (mirrors ``test_ingest_season`` setUp):

* an in-memory SQLite engine is constructed inside the test (no Postgres),
* :mod:`app.db` is deliberately NOT imported (it builds a Postgres engine at
  import time),
* teams are seeded first (``Game`` rows FK to ``team.id``).

The SQLite-only suite does NOT run migrations — the schema is built from the
SQLModel metadata; no new column is added by this plan.

Run from ``backend/``::

    cd backend && .venv/bin/python -m unittest tests.test_freeze_week -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Game, GameStatus, Team, Week
from app.scoreboard.port import ScoreboardFetchError
from app.scoreboard.types import ScoreboardGame, ScoreboardOdds, ScoreboardTeam
from app.services.freeze import FreezeResult, freeze_week
from app.services.odds import is_odds_frozen
from app.seeds.teams import seed_teams

# A fixed injected ``now`` so odds_captured_at is deterministic. Picked to be
# EARLIER than the seeded kickoffs (so the computed freeze_at clock has NOT
# fired) — the freeze must therefore win ONLY because lines_frozen was flipped.
FIXED_NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)

# Seeded espn_team_ids (see app.seeds.teams): 1=ATL, 2=BUF, 4=CIN, 5=CLE, 6=DAL.


def _team(espn_id: str, *, score: int | None = None) -> ScoreboardTeam:
    return ScoreboardTeam(espn_team_id=espn_id, abbreviation=None, score=score)


def _game(
    *,
    event_id: str,
    season: int,
    week: int,
    home: str,
    away: str,
    status: GameStatus = GameStatus.SCHEDULED,
    home_score: int | None = None,
    away_score: int | None = None,
    kickoff: datetime | None = None,
    odds: ScoreboardOdds | None = None,
) -> ScoreboardGame:
    return ScoreboardGame(
        espn_event_id=event_id,
        season=season,
        week=week,
        kickoff_at=kickoff,
        status=status,
        home=_team(home, score=home_score),
        away=_team(away, score=away_score),
        odds=odds,
    )


class _FakeSource:
    """Synthetic ScoreboardSource: serves a pre-built per-week games map.

    ``games_by_week`` maps week -> list[ScoreboardGame]; a week absent from the
    map yields an empty list. ``fail_weeks`` raises
    :class:`ScoreboardFetchError` for the given weeks (the per-week guard test).
    Records each fetched ``(season, week)`` for assertions.
    """

    def __init__(
        self,
        games_by_week: dict[int, list[ScoreboardGame]],
        *,
        fail_weeks: set[int] | None = None,
    ) -> None:
        self._games_by_week = games_by_week
        self._fail_weeks = fail_weeks or set()
        self.fetched: list[tuple[int, int]] = []

    def fetch_week(self, season: int, week: int) -> list[ScoreboardGame]:
        self.fetched.append((season, week))
        if week in self._fail_weeks:
            raise ScoreboardFetchError(f"synthetic failure for week {week}")
        return list(self._games_by_week.get(week, []))


# The two seeded games' kickoffs are LATER than FIXED_NOW so the computed
# freeze_at clock has not fired at FIXED_NOW (freeze must come from the flag).
_KICKOFF_1 = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
_KICKOFF_2 = datetime(2026, 9, 13, 20, 25, tzinfo=timezone.utc)


def _odds(
    *,
    provider: str = "DraftKings",
    provider_id: str = "100",
    spread: float = -3.5,
    total: float = 44.5,
    favorite: str = "1",
    underdog: str = "2",
) -> ScoreboardOdds:
    return ScoreboardOdds(
        provider=provider,
        provider_id=provider_id,
        spread=spread,
        total=total,
        favorite_team_id=favorite,
        underdog_team_id=underdog,
    )


class FreezeWeekTests(unittest.TestCase):
    SEASON = 2026
    WEEK = 1

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _team_id(self, session: Session, espn_id: int) -> int:
        team = session.exec(
            select(Team).where(Team.espn_team_id == espn_id)
        ).first()
        assert team is not None
        return team.id

    def _seed_week_and_games(
        self, session: Session, *, with_odds: bool = False
    ) -> None:
        """Seed one Week (lines_frozen=False) + two odds-less Game rows."""
        week = Week(season=self.SEASON, week=self.WEEK)
        session.add(week)
        session.commit()
        session.refresh(week)

        for event_id, home, away, kickoff in (
            (800001, 1, 2, _KICKOFF_1),
            (800002, 4, 5, _KICKOFF_2),
        ):
            g = Game(
                espn_event_id=event_id,
                week_id=week.id,
                season=self.SEASON,
                week=self.WEEK,
                home_team_id=self._team_id(session, home),
                away_team_id=self._team_id(session, away),
                kickoff_at=kickoff,
                status=GameStatus.SCHEDULED,
            )
            if with_odds:
                g.spread = Decimal("2.0")
                g.total = Decimal("40.0")
                g.odds_provider = "OldBook"
                g.odds_provider_id = "1"
            session.add(g)
        session.commit()

    def _source(self, *, fail: bool = False) -> _FakeSource:
        games = {
            self.WEEK: [
                _game(
                    event_id="800001",
                    season=self.SEASON,
                    week=self.WEEK,
                    home="1",
                    away="2",
                    kickoff=_KICKOFF_1,
                    odds=_odds(
                        spread=-3.5, total=44.5, favorite="1", underdog="2"
                    ),
                ),
                _game(
                    event_id="800002",
                    season=self.SEASON,
                    week=self.WEEK,
                    home="4",
                    away="5",
                    kickoff=_KICKOFF_2,
                    odds=_odds(
                        provider="DraftKings",
                        provider_id="100",
                        spread=-6.5,
                        total=41.0,
                        favorite="4",
                        underdog="5",
                    ),
                ),
            ]
        }
        return _FakeSource(games, fail_weeks={self.WEEK} if fail else None)

    def _week_row(self, session: Session) -> Week:
        return session.exec(
            select(Week).where(
                Week.season == self.SEASON, Week.week == self.WEEK
            )
        ).first()

    def _week_games(self, session: Session) -> list[Game]:
        return list(
            session.exec(
                select(Game).where(
                    Game.season == self.SEASON, Game.week == self.WEEK
                )
            ).all()
        )

    # -- (1) snapshot odds (name+id) AND lock the week --------------------

    def test_freeze_snapshots_odds_and_locks_week(self) -> None:
        with Session(self.engine) as session:
            seed_teams(session)
            self._seed_week_and_games(session)
            source = self._source()

            result = freeze_week(
                session, source, self.SEASON, self.WEEK, now=FIXED_NOW
            )
            self.assertIsInstance(result, FreezeResult)
            self.assertFalse(result.failed)

            game1 = session.exec(
                select(Game).where(Game.espn_event_id == 800001)
            ).first()
            self.assertEqual(game1.odds_provider, "DraftKings")
            self.assertEqual(game1.odds_provider_id, "100")
            self.assertEqual(game1.spread, Decimal("3.5"))
            self.assertGreater(game1.spread, 0)
            self.assertEqual(game1.total, Decimal("44.5"))
            self.assertEqual(
                game1.favorite_team_id, self._team_id(session, 1)
            )
            self.assertEqual(
                game1.underdog_team_id, self._team_id(session, 2)
            )
            captured = game1.odds_captured_at
            if captured.tzinfo is None:
                captured = captured.replace(tzinfo=timezone.utc)
            self.assertEqual(captured, FIXED_NOW)

            week_row = self._week_row(session)
            self.assertTrue(week_row.lines_frozen)

    # -- (2) is_odds_frozen True BEFORE the computed clock ----------------

    def test_freeze_makes_is_odds_frozen_true_before_clock(self) -> None:
        with Session(self.engine) as session:
            seed_teams(session)
            self._seed_week_and_games(session)

            # Sanity: before the freeze, at FIXED_NOW (earlier than the kickoffs
            # / computed freeze_at), the week is NOT frozen.
            week_row = self._week_row(session)
            week_games = self._week_games(session)
            self.assertFalse(
                is_odds_frozen(week_row, week_games, now=FIXED_NOW)
            )

            freeze_week(
                session, self._source(), self.SEASON, self.WEEK, now=FIXED_NOW
            )

            week_row = self._week_row(session)
            week_games = self._week_games(session)
            # Pick a now EARLIER than the computed freeze_at — still frozen,
            # ONLY because lines_frozen is set.
            earlier = FIXED_NOW - timedelta(days=1)
            self.assertTrue(
                is_odds_frozen(week_row, week_games, now=earlier)
            )

    # -- (3) idempotent: second run, no dup rows, stays locked ------------

    def test_freeze_is_idempotent(self) -> None:
        with Session(self.engine) as session:
            seed_teams(session)
            self._seed_week_and_games(session)

            freeze_week(
                session, self._source(), self.SEASON, self.WEEK, now=FIXED_NOW
            )
            r2 = freeze_week(
                session, self._source(), self.SEASON, self.WEEK, now=FIXED_NOW
            )

            self.assertEqual(len(self._week_games(session)), 2)
            self.assertEqual(
                len(session.exec(select(Week)).all()), 1
            )
            week_row = self._week_row(session)
            self.assertTrue(week_row.lines_frozen)
            self.assertTrue(r2.already_frozen)
            self.assertEqual(r2.games_updated, 0)

    # -- (4) fetch failure recorded WITHOUT locking -----------------------

    def test_freeze_records_fetch_failure_without_locking(self) -> None:
        with Session(self.engine) as session:
            seed_teams(session)
            self._seed_week_and_games(session)

            result = freeze_week(
                session,
                self._source(fail=True),
                self.SEASON,
                self.WEEK,
                now=FIXED_NOW,
            )
            self.assertTrue(result.failed)
            week_row = self._week_row(session)
            self.assertFalse(week_row.lines_frozen)

    # -- (5) missing Week raises a clear, labeled error -------------------

    def test_freeze_missing_week_raises(self) -> None:
        with Session(self.engine) as session:
            seed_teams(session)
            # No Week row seeded for (2026, 1).
            with self.assertRaises(ValueError) as ctx:
                freeze_week(
                    session, self._source(), self.SEASON, self.WEEK, now=FIXED_NOW
                )
            self.assertIn("week_not_found", str(ctx.exception))

    # -- (6) source-agnostic: no demo branch, no app.config import --------

    def test_service_is_source_agnostic(self) -> None:
        import app.services.freeze as freeze_mod

        with open(freeze_mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("IS_DEMO_DATA", src)
        self.assertNotIn("import app.config", src)
        self.assertNotIn("from app.config", src)


if __name__ == "__main__":
    unittest.main()
