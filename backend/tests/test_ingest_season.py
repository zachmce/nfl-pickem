"""Offline tests for the season-bootstrap ingest (:mod:`app.services.ingest`).

These exercise :func:`app.services.ingest.ingest_season` end-to-end against an
in-memory SQLite db, driven by a SYNTHETIC ``ScoreboardSource`` that yields
hand-built :class:`~app.scoreboard.types.ScoreboardGame` / ``ScoreboardOdds``
value objects — there is NO live ESPN and NO network access of any kind. The
synthetic source hands the service ALREADY-NORMALIZED odds (the drift-proof
DraftKings SELECTION itself is the adapter's ``normalize_odds`` /
``select_odds_item`` job, covered in ``test_scoreboard_espn``); here we assert the
service persists the handed provider name+id faithfully.

Everything runs OFFLINE (mirrors ``test_import_fixture_2025`` setUp):

* an in-memory SQLite engine is constructed inside the test (no Postgres),
* :mod:`app.db` is deliberately NOT imported (it builds a Postgres engine at
  import time),
* teams are seeded first (``Game`` rows FK to ``team.id``).

The SQLite-only suite does NOT run migrations — the schema is built from the
SQLModel metadata, so the new ``Game.odds_provider_id`` column is exercised here
via the model field (migration 0009 covers the live Postgres path).

Run from ``backend/``::

    cd backend && .venv/bin/python -m unittest tests.test_ingest_season -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from decimal import Decimal

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Game, GameStatus, Team, Week
from app.scoreboard.port import ScoreboardFetchError
from app.scoreboard.types import ScoreboardGame, ScoreboardOdds, ScoreboardTeam
from app.seeds.fixture_2025 import TeamsNotSeededError
from app.seeds.teams import seed_teams
from app.services.ingest import IngestResult, ingest_season

# A fixed injected ``now`` so odds_captured_at is deterministic.
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
    map yields an empty list (a week the source has no games for). ``fail_weeks``
    raises :class:`ScoreboardFetchError` for the given weeks (per-week guard
    test). Records each fetched ``(season, week)`` for assertions.
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


class IngestSeasonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _team_id(self, session: Session, espn_id: int) -> int:
        team = session.exec(select(Team).where(Team.espn_team_id == espn_id)).first()
        assert team is not None
        return team.id

    # -- (1) create path: Week + Game skeleton with resolved FKs ----------

    def test_creates_week_and_game_skeleton_on_empty_db(self) -> None:
        kickoff = datetime(2026, 9, 11, 0, 20, tzinfo=timezone.utc)
        source = _FakeSource(
            {
                1: [
                    _game(
                        event_id="900001",
                        season=2026,
                        week=1,
                        home="1",  # ATL
                        away="2",  # BUF
                        status=GameStatus.FINAL,
                        home_score=24,
                        away_score=20,
                        kickoff=kickoff,
                    )
                ]
            }
        )
        with Session(self.engine) as session:
            seed_teams(session)
            result = ingest_season(session, source, 2026, weeks=range(1, 3), now=FIXED_NOW)

            self.assertIsInstance(result, IngestResult)
            self.assertEqual(result.weeks_created, 1)
            self.assertEqual(result.games_created, 1)
            self.assertEqual(result.weeks_present, 1)
            self.assertEqual(result.games_present, 1)

            week = session.exec(select(Week).where(Week.season == 2026, Week.week == 1)).first()
            self.assertIsNotNone(week)

            game = session.exec(select(Game).where(Game.espn_event_id == 900001)).first()
            self.assertIsNotNone(game)
            self.assertEqual(game.week_id, week.id)
            self.assertEqual(game.season, 2026)
            self.assertEqual(game.week, 1)
            self.assertEqual(game.home_team_id, self._team_id(session, 1))
            self.assertEqual(game.away_team_id, self._team_id(session, 2))
            self.assertEqual(game.status, GameStatus.FINAL)
            self.assertEqual(game.home_score, 24)
            self.assertEqual(game.away_score, 20)
            self.assertIsNotNone(game.kickoff_at)

            # All requested weeks are fetched (one fetch per week); the empty
            # week 2 creates no rows but does not abort.
            self.assertEqual(source.fetched, [(2026, 1), (2026, 2)])

    # -- (2) DraftKings preferred: persists name AND id, positive spread --

    def test_persists_chosen_provider_name_and_id_with_snapshot(self) -> None:
        odds = ScoreboardOdds(
            provider="DraftKings",
            provider_id="100",
            spread=-3.5,  # signed home-relative; stored as positive magnitude
            total=44.5,
            favorite_team_id="1",  # ATL
            underdog_team_id="2",  # BUF
        )
        source = _FakeSource(
            {
                1: [
                    _game(
                        event_id="900010",
                        season=2026,
                        week=1,
                        home="1",
                        away="2",
                        odds=odds,
                    )
                ]
            }
        )
        with Session(self.engine) as session:
            seed_teams(session)
            ingest_season(session, source, 2026, weeks=range(1, 2), now=FIXED_NOW)

            game = session.exec(select(Game).where(Game.espn_event_id == 900010)).first()
            self.assertIsNotNone(game)
            self.assertEqual(game.odds_provider, "DraftKings")
            self.assertEqual(game.odds_provider_id, "100")
            # Positive magnitude regardless of the source's signed spread.
            self.assertEqual(game.spread, Decimal("3.5"))
            self.assertGreater(game.spread, 0)
            self.assertEqual(game.total, Decimal("44.5"))
            self.assertEqual(game.favorite_team_id, self._team_id(session, 1))
            self.assertEqual(game.underdog_team_id, self._team_id(session, 2))
            # ``DateTime(timezone=True)`` round-trips NAIVE on SQLite (Postgres
            # preserves tz), so compare tz-normalized — the stored instant is
            # ``now`` re-attached to UTC.
            captured = game.odds_captured_at
            if captured.tzinfo is None:
                captured = captured.replace(tzinfo=timezone.utc)
            self.assertEqual(captured, FIXED_NOW)

    # -- (3) fallback provider: persisted as-handed (not hardcoded) -------

    def test_persists_fallback_provider_as_handed(self) -> None:
        odds = ScoreboardOdds(
            provider="SomeOtherBook",
            provider_id="999",
            spread=2.0,
            total=40.0,
            favorite_team_id="2",
            underdog_team_id="1",
        )
        source = _FakeSource(
            {1: [_game(event_id="900020", season=2026, week=1, home="1", away="2", odds=odds)]}
        )
        with Session(self.engine) as session:
            seed_teams(session)
            ingest_season(session, source, 2026, weeks=range(1, 2), now=FIXED_NOW)
            game = session.exec(select(Game).where(Game.espn_event_id == 900020)).first()
            # The service persists whatever name+id it was handed — it never
            # re-selects DraftKings or hardcodes an id.
            self.assertEqual(game.odds_provider, "SomeOtherBook")
            self.assertEqual(game.odds_provider_id, "999")

    def test_game_without_odds_keeps_all_odds_fields_none(self) -> None:
        source = _FakeSource(
            {1: [_game(event_id="900030", season=2026, week=1, home="1", away="2", odds=None)]}
        )
        with Session(self.engine) as session:
            seed_teams(session)
            ingest_season(session, source, 2026, weeks=range(1, 2), now=FIXED_NOW)
            game = session.exec(select(Game).where(Game.espn_event_id == 900030)).first()
            self.assertIsNone(game.spread)
            self.assertIsNone(game.total)
            self.assertIsNone(game.favorite_team_id)
            self.assertIsNone(game.underdog_team_id)
            self.assertIsNone(game.odds_provider)
            self.assertIsNone(game.odds_provider_id)
            self.assertIsNone(game.odds_captured_at)

    # -- (4) idempotency: no dup rows, None never nulls a present value ---

    def test_reingest_is_idempotent_and_none_never_nulls(self) -> None:
        kickoff = datetime(2026, 9, 11, 0, 20, tzinfo=timezone.utc)
        odds = ScoreboardOdds(
            provider="DraftKings",
            provider_id="100",
            spread=-3.5,
            total=44.5,
            favorite_team_id="1",
            underdog_team_id="2",
        )
        first = _FakeSource(
            {
                1: [
                    _game(
                        event_id="900040",
                        season=2026,
                        week=1,
                        home="1",
                        away="2",
                        status=GameStatus.FINAL,
                        home_score=27,
                        away_score=13,
                        kickoff=kickoff,
                        odds=odds,
                    )
                ]
            }
        )
        # Second run: same event, but scores withheld (None) — must NOT null the
        # already-present scores; and no duplicate rows.
        second = _FakeSource(
            {
                1: [
                    _game(
                        event_id="900040",
                        season=2026,
                        week=1,
                        home="1",
                        away="2",
                        status=GameStatus.FINAL,
                        home_score=None,
                        away_score=None,
                        kickoff=kickoff,
                        odds=odds,
                    )
                ]
            }
        )
        with Session(self.engine) as session:
            seed_teams(session)
            r1 = ingest_season(session, first, 2026, weeks=range(1, 2), now=FIXED_NOW)
            self.assertEqual(r1.games_created, 1)

            r2 = ingest_season(session, second, 2026, weeks=range(1, 2), now=FIXED_NOW)
            # No new rows on the re-run.
            self.assertEqual(r2.weeks_created, 0)
            self.assertEqual(r2.games_created, 0)
            self.assertEqual(r2.weeks_present, 1)
            self.assertEqual(r2.games_present, 1)

            self.assertEqual(len(session.exec(select(Week)).all()), 1)
            self.assertEqual(len(session.exec(select(Game)).all()), 1)

            game = session.exec(select(Game).where(Game.espn_event_id == 900040)).first()
            # None scores on the second run did NOT null the present values.
            self.assertEqual(game.home_score, 27)
            self.assertEqual(game.away_score, 13)

    # -- (5) per-week fetch error isolated in failed_weeks ----------------

    def test_per_week_fetch_error_isolated(self) -> None:
        source = _FakeSource(
            {
                1: [_game(event_id="900050", season=2026, week=1, home="1", away="2")],
                3: [_game(event_id="900052", season=2026, week=3, home="4", away="5")],
            },
            fail_weeks={2},
        )
        with Session(self.engine) as session:
            seed_teams(session)
            result = ingest_season(session, source, 2026, weeks=range(1, 4), now=FIXED_NOW)
            # Week 2 raised -> recorded; weeks 1 and 3 still ingested.
            self.assertEqual(result.failed_weeks, ((2026, 2),))
            self.assertEqual(result.weeks_created, 2)
            self.assertEqual(result.games_created, 2)
            self.assertIsNotNone(
                session.exec(select(Game).where(Game.espn_event_id == 900050)).first()
            )
            self.assertIsNotNone(
                session.exec(select(Game).where(Game.espn_event_id == 900052)).first()
            )

    # -- unseeded team raises (no orphan FK) ------------------------------

    def test_unseeded_team_raises(self) -> None:
        # espn_team_id "99" is not a seeded team.
        source = _FakeSource(
            {1: [_game(event_id="900060", season=2026, week=1, home="99", away="1")]}
        )
        with Session(self.engine) as session:
            seed_teams(session)
            with self.assertRaises(TeamsNotSeededError):
                ingest_season(session, source, 2026, weeks=range(1, 2), now=FIXED_NOW)


class IngestSeasonWrapperSummaryTests(unittest.TestCase):
    """The task wrapper's return summary must be JSON-serializable.

    Asserts the wrapper's contract at the SERVICE + summary level (offline) —
    WITHOUT importing app.db / app.config or constructing the ESPN adapter, and
    without executing the Celery task against a real broker. We build the same
    summary dict shape ``ingest_season_task`` returns from a real
    :class:`IngestResult` and prove ``json.dumps`` accepts it (Celery's json
    result serializer would).
    """

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_summary_is_json_serializable(self) -> None:
        # Drive the SAME service entry the task uses, including a failed week so
        # failed_weeks is non-empty (the tuple-of-tuples that must flatten).
        source = _FakeSource(
            {1: [_game(event_id="900070", season=2026, week=1, home="1", away="2")]},
            fail_weeks={2},
        )
        with Session(self.engine) as session:
            seed_teams(session)
            result = ingest_season(session, source, 2026, weeks=range(1, 3), now=FIXED_NOW)

        # Mirror ingest_season_task's summary construction exactly.
        summary = {
            "weeks_present": result.weeks_present,
            "games_present": result.games_present,
            "weeks_created": result.weeks_created,
            "games_created": result.games_created,
            "failed_weeks": [list(w) for w in result.failed_weeks],
        }
        # Must not raise — JSON-serializable for Celery's json serializer.
        encoded = json.dumps(summary)
        roundtrip = json.loads(encoded)
        self.assertEqual(roundtrip["failed_weeks"], [[2026, 2]])
        self.assertEqual(roundtrip["games_created"], 1)


if __name__ == "__main__":
    unittest.main()
