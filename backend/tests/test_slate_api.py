"""Offline tests for the pickable-slate read endpoint.

Covers the thin authenticated HTTP router (:mod:`app.api.slate`) via an in-memory
``TestClient`` — mirroring the conventions of :mod:`tests.test_current_week_api`:

* a single shared in-memory SQLite connection (``StaticPool``) so every
  ``Session`` — including the one ``get_current_user`` opens — sees the SAME db,
* ``app.dependency_overrides[get_session]`` routed at that engine so importing
  :mod:`app.main` never opens a real Postgres connection,
* no network of any kind,
* bearer auth for reads (CSRF-exempt) via ``_bearer_headers``; ``_clear_auth``
  for the unauthenticated 401 case; ``_assert_envelope`` for the error shape.

Line/lock cases are driven by seeding ``Game`` line fields + ``kickoff_at``
relative to the real clock (future = now + days, past = now - days), so the
results are deterministic and not clock-flaky. No picks are seeded for the HTTP
cases — this endpoint does not read picks.

Case 9 is a pure-service regression (no HTTP): it calls ``validate_roster`` /
``check_new_pick`` directly to assert an OVER/UNDER pick on a totals-less game is
NEVER routed into the spread-eligibility branch (no ``PICKEM_SPREAD_INELIGIBLE``).

> Note: on this machine there is no bare ``python`` on ``PATH``; run with
> ``backend/.venv/bin/python -m unittest``.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.db import get_session
from app.main import app
from app.models import Game, GameStatus, Pick, PickType, Team, User, Week
from app.services.auth import create_session_cookie, hash_password
from app.services.pick_validation import (
    ViolationCode,
    check_new_pick,
    validate_roster,
)

SEASON = 2025


def _aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from SQLite."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class SlateTests(unittest.TestCase):
    """HTTP coverage for the slate read surface (+ one pure-service regression)."""

    user_id: int

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        with Session(self.engine) as session:
            # Teams (FK targets for the games each test seeds).
            teams = [
                Team(espn_team_id=i, abbreviation=f"T{i}", display_name=f"Team {i}")
                for i in range(1, 5)
            ]
            session.add_all(teams)
            session.commit()
            for t in teams:
                session.refresh(t)
            self.tid = [t.id for t in teams]

            # One active user to authenticate the shared read.
            user = User(
                display_name="alice",
                password_hash=hash_password("correct horse battery staple"),
                is_active=True,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            assert user.id is not None
            self.user_id = user.id

        def _override_get_session():
            with Session(self.engine) as session:
                yield session

        app.dependency_overrides[get_session] = _override_get_session
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.pop(get_session, None)
        self.client.close()
        self.engine.dispose()

    # -- helpers -----------------------------------------------------------

    def _session(self) -> Session:
        return Session(self.engine)

    def _seed_week_row(self, week: int, *, lines_frozen: bool = False) -> int:
        with self._session() as session:
            week_row = Week(season=SEASON, week=week, lines_frozen=lines_frozen)
            session.add(week_row)
            session.commit()
            session.refresh(week_row)
            assert week_row.id is not None
            return week_row.id

    def _add_game(
        self,
        *,
        week_id: int,
        week: int,
        espn_event_id: int,
        kickoff_at: datetime | None,
        home_team_id: int,
        away_team_id: int,
        spread: Decimal | None = None,
        total: Decimal | None = None,
        favorite_team_id: int | None = None,
        underdog_team_id: int | None = None,
        status: GameStatus = GameStatus.SCHEDULED,
    ) -> int:
        """Add a single game with explicit line + kickoff; returns its id."""
        with self._session() as session:
            g = Game(
                espn_event_id=espn_event_id,
                week_id=week_id,
                season=SEASON,
                week=week,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                kickoff_at=kickoff_at,
                status=status,
                spread=spread,
                total=total,
                favorite_team_id=favorite_team_id,
                underdog_team_id=underdog_team_id,
            )
            session.add(g)
            session.commit()
            session.refresh(g)
            assert g.id is not None
            return g.id

    def _bearer_headers(self, user_id: int) -> dict[str, str]:
        """Bearer auth for reads (CSRF-exempt)."""
        return {"Authorization": f"Bearer {create_session_cookie(user_id)}"}

    def _clear_auth(self) -> None:
        self.client.cookies.clear()

    @staticmethod
    def _assert_envelope(body: dict) -> dict:
        assert "error" in body, f"expected an error envelope, got: {body}"
        err = body["error"]
        assert "code" in err, f"envelope missing 'code': {err}"
        return err

    def _get(self, week: int) -> object:
        return self.client.get(
            f"/api/slate?season={SEASON}&week={week}",
            headers=self._bearer_headers(self.user_id),
        )

    # -- case 1: line fields + team identity + ordering --------------------

    def test_games_returned_with_line_fields_and_team_identity(self) -> None:
        """A normal game round-trips its line + home/away identity; games sorted
        by kickoff."""
        now = datetime.now(timezone.utc)
        wk_id = self._seed_week_row(1)
        # Seed two games out of kickoff order to prove the sort.
        later = now + timedelta(days=2)
        earlier = now + timedelta(days=1)
        self._add_game(
            week_id=wk_id, week=1, espn_event_id=1002, kickoff_at=later,
            home_team_id=self.tid[2], away_team_id=self.tid[3],
            spread=Decimal("3.5"), total=Decimal("44.5"),
            favorite_team_id=self.tid[2], underdog_team_id=self.tid[3],
        )
        self._add_game(
            week_id=wk_id, week=1, espn_event_id=1001, kickoff_at=earlier,
            home_team_id=self.tid[0], away_team_id=self.tid[1],
            spread=Decimal("7.0"), total=Decimal("48.0"),
            favorite_team_id=self.tid[0], underdog_team_id=self.tid[1],
        )

        resp = self._get(1)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["season"], SEASON)
        self.assertEqual(body["week"], 1)
        self.assertEqual(len(body["games"]), 2)

        # Sorted by kickoff: the earlier game (espn 1001) comes first.
        first = body["games"][0]
        self.assertEqual(_aware(datetime.fromisoformat(first["kickoff_at"])), earlier)
        self.assertEqual(Decimal(str(first["spread"])), Decimal("7.0"))
        self.assertEqual(Decimal(str(first["total"])), Decimal("48.0"))
        self.assertEqual(first["favorite_team_id"], self.tid[0])
        self.assertEqual(first["underdog_team_id"], self.tid[1])
        self.assertEqual(first["home_team"]["team_id"], self.tid[0])
        self.assertEqual(first["home_team"]["abbreviation"], "T1")
        self.assertEqual(first["home_team"]["display_name"], "Team 1")
        self.assertEqual(first["away_team"]["team_id"], self.tid[1])

        second = body["games"][1]
        self.assertEqual(_aware(datetime.fromisoformat(second["kickoff_at"])), later)

    # -- case 2: eligibility, normal game ---------------------------------

    def test_eligibility_normal_game_all_four_true(self) -> None:
        """A spread+sides+total game -> all four PickType values eligible."""
        now = datetime.now(timezone.utc)
        wk_id = self._seed_week_row(1)
        self._add_game(
            week_id=wk_id, week=1, espn_event_id=1001,
            kickoff_at=now + timedelta(days=1),
            home_team_id=self.tid[0], away_team_id=self.tid[1],
            spread=Decimal("3.5"), total=Decimal("44.5"),
            favorite_team_id=self.tid[0], underdog_team_id=self.tid[1],
        )

        resp = self._get(1)
        self.assertEqual(resp.status_code, 200, resp.text)
        elig = resp.json()["games"][0]["eligibility"]
        for t in PickType:
            self.assertTrue(elig[t.value], f"{t.value} should be eligible")

    # -- case 3: no total ---------------------------------------------------

    def test_eligibility_no_total_over_under_false(self) -> None:
        """total=None -> OVER & UNDER false; spread types still true."""
        now = datetime.now(timezone.utc)
        wk_id = self._seed_week_row(1)
        self._add_game(
            week_id=wk_id, week=1, espn_event_id=1001,
            kickoff_at=now + timedelta(days=1),
            home_team_id=self.tid[0], away_team_id=self.tid[1],
            spread=Decimal("3.5"), total=None,
            favorite_team_id=self.tid[0], underdog_team_id=self.tid[1],
        )

        resp = self._get(1)
        self.assertEqual(resp.status_code, 200, resp.text)
        elig = resp.json()["games"][0]["eligibility"]
        self.assertFalse(elig[PickType.OVER.value])
        self.assertFalse(elig[PickType.UNDER.value])
        self.assertTrue(elig[PickType.FAVORITE_COVER.value])
        self.assertTrue(elig[PickType.UNDERDOG_COVER.value])

    # -- case 4: true pick'em ----------------------------------------------

    def test_eligibility_true_pickem_spread_types_false(self) -> None:
        """A true pick'em (spread None) but WITH a total -> spread types false,
        OVER/UNDER true."""
        now = datetime.now(timezone.utc)
        wk_id = self._seed_week_row(1)
        self._add_game(
            week_id=wk_id, week=1, espn_event_id=1001,
            kickoff_at=now + timedelta(days=1),
            home_team_id=self.tid[0], away_team_id=self.tid[1],
            spread=None, total=Decimal("44.5"),
            favorite_team_id=None, underdog_team_id=None,
        )

        resp = self._get(1)
        self.assertEqual(resp.status_code, 200, resp.text)
        elig = resp.json()["games"][0]["eligibility"]
        self.assertFalse(elig[PickType.FAVORITE_COVER.value])
        self.assertFalse(elig[PickType.UNDERDOG_COVER.value])
        self.assertTrue(elig[PickType.OVER.value])
        self.assertTrue(elig[PickType.UNDER.value])

    # -- case 5: per-game locked split -------------------------------------

    def test_locked_past_true_future_false(self) -> None:
        """A PAST-kickoff game -> locked True; a FUTURE-kickoff game in the same
        week -> locked False."""
        now = datetime.now(timezone.utc)
        wk_id = self._seed_week_row(1)
        past = now - timedelta(hours=2)
        future = now + timedelta(days=1)
        self._add_game(
            week_id=wk_id, week=1, espn_event_id=1001, kickoff_at=past,
            home_team_id=self.tid[0], away_team_id=self.tid[1],
            status=GameStatus.IN_PROGRESS,
        )
        self._add_game(
            week_id=wk_id, week=1, espn_event_id=1002, kickoff_at=future,
            home_team_id=self.tid[2], away_team_id=self.tid[3],
        )

        resp = self._get(1)
        self.assertEqual(resp.status_code, 200, resp.text)
        games = resp.json()["games"]
        locked_by_event = {
            _aware(datetime.fromisoformat(g["kickoff_at"])): g["locked"]
            for g in games
        }
        self.assertTrue(locked_by_event[past])
        self.assertFalse(locked_by_event[future])

    # -- case 6: demo-shift -> everything future, all unlocked -------------

    def test_demo_like_shift_all_unlocked(self) -> None:
        """All kickoffs shifted into the FUTURE -> every game locked False,
        computed against real now (no IS_DEMO_DATA branch)."""
        now = datetime.now(timezone.utc)
        wk_id = self._seed_week_row(1)
        self._add_game(
            week_id=wk_id, week=1, espn_event_id=1001,
            kickoff_at=now + timedelta(days=1),
            home_team_id=self.tid[0], away_team_id=self.tid[1],
        )
        self._add_game(
            week_id=wk_id, week=1, espn_event_id=1002,
            kickoff_at=now + timedelta(days=2),
            home_team_id=self.tid[2], away_team_id=self.tid[3],
        )

        resp = self._get(1)
        self.assertEqual(resp.status_code, 200, resp.text)
        games = resp.json()["games"]
        self.assertEqual(len(games), 2)
        for g in games:
            self.assertFalse(g["locked"])

    # -- case 7: unknown/empty week ----------------------------------------

    def test_unknown_week_returns_empty_games(self) -> None:
        """An unseeded week -> 200 with games == [] (a pure read, never 404)."""
        resp = self._get(99)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["week"], 99)
        self.assertEqual(body["games"], [])

    # -- case 8: unauthenticated -------------------------------------------

    def test_unauthenticated_rejected_401(self) -> None:
        """GET /api/slate with no auth -> 401 + error envelope."""
        self._clear_auth()
        resp = self.client.get(f"/api/slate?season={SEASON}&week=1")
        self.assertEqual(resp.status_code, 401, resp.text)
        self._assert_envelope(resp.json())

    # -- case 9: odds_frozen false (future kickoff, not frozen) ------------

    def test_slate_reports_odds_frozen_false_before_freeze(self) -> None:
        """A week whose earliest kickoff is FAR in the future -> freeze_at is in
        the future relative to real now -> top-level odds_frozen is False."""
        now = datetime.now(timezone.utc)
        wk_id = self._seed_week_row(1)  # lines_frozen defaults False
        # Kickoff far in the future so freeze_at (min noon-ET-Wed, earliest
        # kickoff) is still ahead of real now -> not yet frozen.
        self._add_game(
            week_id=wk_id, week=1, espn_event_id=1001,
            kickoff_at=now + timedelta(days=14),
            home_team_id=self.tid[0], away_team_id=self.tid[1],
            spread=Decimal("3.5"), total=Decimal("44.5"),
            favorite_team_id=self.tid[0], underdog_team_id=self.tid[1],
        )

        resp = self._get(1)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertFalse(resp.json()["odds_frozen"])

    # -- case 10: odds_frozen true (lines_frozen override) -----------------

    def test_slate_reports_odds_frozen_true_after_freeze(self) -> None:
        """A week with lines_frozen=True hits the explicit override branch of
        is_odds_frozen (clock-independent) -> top-level odds_frozen is True.

        Uses a FUTURE kickoff so the game itself is not locked — proving the
        week-level freeze flag is independent of per-game lock.
        """
        now = datetime.now(timezone.utc)
        wk_id = self._seed_week_row(1, lines_frozen=True)
        self._add_game(
            week_id=wk_id, week=1, espn_event_id=1001,
            kickoff_at=now + timedelta(days=14),
            home_team_id=self.tid[0], away_team_id=self.tid[1],
            spread=Decimal("3.5"), total=Decimal("44.5"),
            favorite_team_id=self.tid[0], underdog_team_id=self.tid[1],
        )

        resp = self._get(1)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(resp.json()["odds_frozen"])

    # -- case 11: pure-service refactor regression -------------------------

    def test_totals_pick_on_totalless_game_not_spread_ineligible(self) -> None:
        """REGRESSION: an OVER/UNDER pick on a game with NO total must NOT raise
        PICKEM_SPREAD_INELIGIBLE from validate_roster / check_new_pick.

        Proves the refactor that routes the spread guard through
        is_pick_type_eligible keeps the outer _SPREAD_PICK_TYPES filter — a totals
        pick never reaches the spread-eligibility branch even though
        is_pick_type_eligible(game, OVER) is False when total is None.
        """
        # A totals-less game (also a true pick'em, the harshest case).
        game = Game(
            id=1,
            espn_event_id=9001,
            week_id=1,
            season=SEASON,
            week=1,
            home_team_id=self.tid[0],
            away_team_id=self.tid[1],
            spread=None,
            total=None,
            favorite_team_id=None,
            underdog_team_id=None,
        )
        games_by_id = {1: game}

        for pt in (PickType.OVER, PickType.UNDER):
            with self.subTest(pick_type=pt):
                pick = Pick(
                    user_id=self.user_id, game_id=1, week_id=1, pick_type=pt
                )
                roster_result = validate_roster([pick], games_by_id)
                self.assertNotIn(
                    ViolationCode.PICKEM_SPREAD_INELIGIBLE,
                    {v.code for v in roster_result.violations},
                    f"validate_roster wrongly flagged {pt.value} as spread-ineligible",
                )

                new_result = check_new_pick(pick, [], games_by_id)
                self.assertNotIn(
                    ViolationCode.PICKEM_SPREAD_INELIGIBLE,
                    {v.code for v in new_result.violations},
                    f"check_new_pick wrongly flagged {pt.value} as spread-ineligible",
                )


if __name__ == "__main__":
    unittest.main()
