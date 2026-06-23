"""Offline HTTP tests for the authenticated /api/picks router.

This is the FIRST HTTP test in the repo, so it establishes the conventions every
later API test can follow. It is fully OFFLINE:

* an in-memory SQLite engine is built inside ``setUp`` (no Postgres),
* the app's DB dependency (:func:`app.db.get_session`) is replaced via
  ``app.dependency_overrides`` with a session bound to that in-memory engine, so
  importing :mod:`app.main` (which constructs a Postgres engine object lazily)
  never actually opens a Postgres connection,
* no network of any kind is touched — the picks router calls the in-process
  submission service against the in-memory db, never ESPN.

Auth is exercised on BOTH supported paths so the CSRF surface is covered:

* mutating ``POST`` uses the signed session COOKIE plus the double-submit CSRF
  pair (``csrftoken`` cookie + matching ``X-CSRF-Token`` header), exactly as the
  real SPA does and as :mod:`app.csrf` enforces;
* reads use a ``Authorization: Bearer <token>`` header, which is CSRF-exempt.

Datetime handling mirrors the service: ``DateTime(timezone=True)`` round-trips
NAIVE on SQLite, so kickoffs are re-attached to UTC where the window/lock math
needs tz-aware values. Real ``datetime.now(timezone.utc)`` is used (no virtual
clock per PROJECT.md): the happy-path week's earliest kickoff is positioned in
the FUTURE (window open); the lock-test game's kickoff is in the PAST.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); run with ``backend/.venv/bin/python -m unittest``.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.csrf import CSRF_COOKIE_NAME, CSRF_HEADER_NAME
from app.db import get_session
from app.main import app
from app.models import (
    Game,
    GameStatus,
    Pick,
    PickResult,
    PickType,
    Team,
    User,
    Week,
)
from app.services.auth import create_session_cookie, hash_password

SEASON = 2025
WEEK = 1

# Kickoffs relative to the real clock: the open-window week starts well in the
# FUTURE; the lock-test game has already kicked off (PAST).
_FUTURE = timedelta(days=2)
_PAST = timedelta(hours=2)


def _aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from SQLite."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class PicksApiTests(unittest.TestCase):
    """Offline TestClient coverage for submit + read on /api/picks."""

    # Game id handles populated in setUp (stable across a test via the fixture).
    game_spread_id: int
    game_total_id: int
    game_pickem_id: int
    game_locked_id: int
    user_a_id: int
    user_b_id: int
    week_id: int

    def setUp(self) -> None:
        # A single shared in-memory connection (StaticPool) so every Session —
        # including the one get_current_user opens via its own Depends — sees the
        # SAME database. The default sqlite:// pool hands each connection a fresh,
        # EMPTY in-memory db; StaticPool pins one connection for the whole test.
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        now = datetime.now(timezone.utc)
        with Session(self.engine) as session:
            # --- Teams (FK targets for game home/away/favorite/underdog) ------
            teams = [
                Team(espn_team_id=i, abbreviation=f"T{i}", display_name=f"Team {i}")
                for i in range(1, 9)
            ]
            session.add_all(teams)
            session.commit()
            for t in teams:
                session.refresh(t)
            tid = [t.id for t in teams]

            # --- Week (window stamping is irrelevant: the service computes the
            #     window from kickoffs; we leave window cols None) -------------
            week = Week(season=SEASON, week=WEEK)
            session.add(week)
            session.commit()
            session.refresh(week)
            assert week.id is not None
            self.week_id = week.id

            # --- Games --------------------------------------------------------
            # All future-kickoff games share the SAME earliest kickoff so the
            # week-level window stays OPEN for the happy path; the locked game is
            # in the past but does NOT pull the window close earlier than the
            # happy-path games (it shares week 1 — so to keep the window open we
            # put the locked game in its OWN week so its past kickoff can't close
            # week 1). Simpler: keep the lock case in a SEPARATE week.
            #
            # Week 1 (open window): a spread game, a totals game, and a true
            # pick'em game — all kicking off in the FUTURE.
            game_spread = Game(
                espn_event_id=1001,
                week_id=week.id,
                season=SEASON,
                week=WEEK,
                home_team_id=tid[0],
                away_team_id=tid[1],
                kickoff_at=now + _FUTURE,
                status=GameStatus.SCHEDULED,
                spread=Decimal("3.5"),
                total=Decimal("44.5"),
                favorite_team_id=tid[0],
                underdog_team_id=tid[1],
            )
            game_total = Game(
                espn_event_id=1002,
                week_id=week.id,
                season=SEASON,
                week=WEEK,
                home_team_id=tid[2],
                away_team_id=tid[3],
                kickoff_at=now + _FUTURE + timedelta(hours=3),
                status=GameStatus.SCHEDULED,
                spread=Decimal("6.5"),
                total=Decimal("41.0"),
                favorite_team_id=tid[2],
                underdog_team_id=tid[3],
            )
            game_pickem = Game(
                espn_event_id=1003,
                week_id=week.id,
                season=SEASON,
                week=WEEK,
                home_team_id=tid[4],
                away_team_id=tid[5],
                kickoff_at=now + _FUTURE + timedelta(hours=6),
                status=GameStatus.SCHEDULED,
                # True pick'em: spread 0 / no favorite-underdog side.
                spread=Decimal("0.0"),
                total=Decimal("48.0"),
                favorite_team_id=None,
                underdog_team_id=None,
            )
            session.add_all([game_spread, game_total, game_pickem])

            # The locked-game case lives in its OWN week so its PAST kickoff
            # closes only that week's window — week 1 (above) stays open.
            locked_week = Week(season=SEASON, week=WEEK + 1)
            session.add(locked_week)
            session.commit()
            session.refresh(locked_week)
            assert locked_week.id is not None
            self.locked_week_id = locked_week.id
            game_locked = Game(
                espn_event_id=2001,
                week_id=locked_week.id,
                season=SEASON,
                week=WEEK + 1,
                home_team_id=tid[6],
                away_team_id=tid[7],
                kickoff_at=now - _PAST,  # already kicked off
                status=GameStatus.IN_PROGRESS,
                spread=Decimal("2.5"),
                total=Decimal("40.0"),
                favorite_team_id=tid[6],
                underdog_team_id=tid[7],
            )
            session.add(game_locked)
            session.commit()
            for g in (game_spread, game_total, game_pickem, game_locked):
                session.refresh(g)
            self.game_spread_id = game_spread.id
            self.game_total_id = game_total.id
            self.game_pickem_id = game_pickem.id
            self.game_locked_id = game_locked.id

            # --- Users --------------------------------------------------------
            pw = hash_password("correct horse battery staple")
            user_a = User(
                display_name="userA",
                password_hash=pw,
                is_active=True,
            )
            user_b = User(
                display_name="userB",
                password_hash=pw,
                is_active=True,
            )
            session.add_all([user_a, user_b])
            session.commit()
            session.refresh(user_a)
            session.refresh(user_b)
            assert user_a.id is not None and user_b.id is not None
            self.user_a_id = user_a.id
            self.user_b_id = user_b.id

        # Route the app's DB dependency at the in-memory engine. The same
        # callable object is the override key used by both the router and
        # get_current_user (both do ``from app.db import get_session``).
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

    def _picks_for(self, user_id: int, week_id: int) -> list[Pick]:
        with self._session() as session:
            return list(
                session.exec(
                    select(Pick).where(
                        Pick.user_id == user_id, Pick.week_id == week_id
                    )
                ).all()
            )

    def _seed_pick(
        self,
        *,
        user_id: int,
        game_id: int,
        week_id: int,
        pick_type: PickType,
        is_mortal_lock: bool = False,
    ) -> int:
        with self._session() as session:
            pick = Pick(
                user_id=user_id,
                game_id=game_id,
                week_id=week_id,
                pick_type=pick_type,
                is_mortal_lock=is_mortal_lock,
            )
            session.add(pick)
            session.commit()
            session.refresh(pick)
            assert pick.id is not None
            return pick.id

    def _cookie_auth_headers(self, user_id: int) -> dict[str, str]:
        """Set the signed session + CSRF cookies and return the CSRF header.

        Mirrors the SPA double-submit contract: a ``csrftoken`` cookie whose
        value is echoed in ``X-CSRF-Token``. The session cookie authenticates;
        the CSRF pair satisfies :mod:`app.csrf` for the mutating POST.
        """
        csrf_value = "test-csrf-token-value"
        self.client.cookies.set("session", create_session_cookie(user_id))
        self.client.cookies.set(CSRF_COOKIE_NAME, csrf_value)
        return {CSRF_HEADER_NAME: csrf_value}

    def _bearer_headers(self, user_id: int) -> dict[str, str]:
        """Bearer auth for reads (CSRF-exempt)."""
        return {"Authorization": f"Bearer {create_session_cookie(user_id)}"}

    def _clear_auth(self) -> None:
        self.client.cookies.clear()

    @staticmethod
    def _assert_envelope(body: dict) -> dict:
        """Assert the body is the ``{"error": {"code", ...}}`` envelope."""
        assert "error" in body, f"expected an error envelope, got: {body}"
        err = body["error"]
        assert "code" in err, f"envelope missing 'code': {err}"
        return err

    # -- tests -------------------------------------------------------------

    def test_happy_path_submit_in_open_window(self) -> None:
        """POST valid picks (incl. a mortal lock) in an OPEN window -> 200."""
        headers = self._cookie_auth_headers(self.user_a_id)
        resp = self.client.post(
            "/api/picks",
            json={
                "season": SEASON,
                "week": WEEK,
                "picks": [
                    {
                        "game_id": self.game_spread_id,
                        "pick_type": "FAVORITE_COVER",
                        "is_mortal_lock": True,
                    },
                    {"game_id": self.game_total_id, "pick_type": "OVER"},
                ],
            },
            headers=headers,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        out = resp.json()
        self.assertEqual(len(out), 2)
        # Persisted as PENDING / 0 points, owned by userA.
        for item in out:
            self.assertEqual(item["result"], PickResult.PENDING.value)
            self.assertEqual(item["points"], 0)

        rows = self._picks_for(self.user_a_id, self.week_id)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r.user_id == self.user_a_id for r in rows))
        self.assertTrue(any(r.is_mortal_lock for r in rows))
        # No leakage into userB.
        self.assertEqual(self._picks_for(self.user_b_id, self.week_id), [])

    def test_window_closed_is_rejected_4xx_no_writes(self) -> None:
        """A closed-window submit -> 409 envelope; nothing persisted."""
        # The locked week's only game kicked off in the past -> window closed.
        headers = self._cookie_auth_headers(self.user_a_id)
        resp = self.client.post(
            "/api/picks",
            json={
                "season": SEASON,
                "week": WEEK + 1,
                "picks": [
                    {"game_id": self.game_locked_id, "pick_type": "FAVORITE_COVER"}
                ],
            },
            headers=headers,
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("reason"), "window_closed")
        self.assertEqual(self._picks_for(self.user_a_id, self.locked_week_id), [])

    def test_per_game_locked_is_rejected_4xx_no_writes(self) -> None:
        """A pick on a game that has kicked off is rejected 4xx; no writes.

        The week-level window closes at the week's EARLIEST kickoff, so once any
        game in a week has kicked off the window is already closed and a later
        game's per-game lock is the deeper guard behind it. Both surface as a
        409 conflict envelope (``window_closed`` or ``game_locked``) and never a
        500. We seed a week whose only games are already past kickoff and assert
        the pick on the locked game is rejected with no row written.
        """
        now = datetime.now(timezone.utc)
        with self._session() as session:
            wk = Week(season=SEASON, week=99)
            session.add(wk)
            session.commit()
            session.refresh(wk)
            assert wk.id is not None
            wk_id = wk.id
            # Earliest kickoff is in the future (window OPEN), but the picked
            # game itself has already kicked off, so the PER-GAME lock is the
            # guard that fires.
            open_anchor = Game(
                espn_event_id=9901,
                week_id=wk.id,
                season=SEASON,
                week=99,
                home_team_id=1,
                away_team_id=2,
                kickoff_at=now + _FUTURE,
                status=GameStatus.SCHEDULED,
                spread=Decimal("3.0"),
                total=Decimal("40.0"),
                favorite_team_id=1,
                underdog_team_id=2,
            )
            locked_game = Game(
                espn_event_id=9902,
                week_id=wk.id,
                season=SEASON,
                week=99,
                home_team_id=3,
                away_team_id=4,
                kickoff_at=now - _PAST,
                status=GameStatus.IN_PROGRESS,
                spread=Decimal("3.0"),
                total=Decimal("40.0"),
                favorite_team_id=3,
                underdog_team_id=4,
            )
            session.add_all([open_anchor, locked_game])
            session.commit()
            session.refresh(locked_game)
            locked_id = locked_game.id

        # is_pick_open uses min(kickoffs): the locked game's PAST kickoff is the
        # earliest, so the window is closed -> service raises window_closed
        # BEFORE reaching the per-game lock. Either way it is a 409 conflict
        # envelope with no write. (The per-game lock itself is unit-tested
        # directly in test_pick_window; here we prove the API rejects a pick on
        # an already-started game via the structured 4xx envelope.)
        headers = self._cookie_auth_headers(self.user_a_id)
        resp = self.client.post(
            "/api/picks",
            json={
                "season": SEASON,
                "week": 99,
                "picks": [{"game_id": locked_id, "pick_type": "FAVORITE_COVER"}],
            },
            headers=headers,
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertIn(err.get("reason"), {"window_closed", "game_locked"})
        self.assertEqual(self._picks_for(self.user_a_id, wk_id), [])

    def test_conflict_first_pick_precedence(self) -> None:
        """Existing FAVORITE_COVER wins; incoming UNDERDOG_COVER -> 409."""
        existing_id = self._seed_pick(
            user_id=self.user_a_id,
            game_id=self.game_spread_id,
            week_id=self.week_id,
            pick_type=PickType.FAVORITE_COVER,
        )
        headers = self._cookie_auth_headers(self.user_a_id)
        resp = self.client.post(
            "/api/picks",
            json={
                "season": SEASON,
                "week": WEEK,
                "picks": [
                    {
                        "game_id": self.game_spread_id,
                        "pick_type": "UNDERDOG_COVER",
                    }
                ],
            },
            headers=headers,
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("reason"), "CONTRADICTORY_PICK")
        # Existing pick unchanged: still exactly one, still FAVORITE_COVER.
        rows = self._picks_for(self.user_a_id, self.week_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].id, existing_id)
        self.assertEqual(rows[0].pick_type, PickType.FAVORITE_COVER)

    def test_roster_second_mortal_lock_rejected(self) -> None:
        """A second mortal lock when one already exists -> 409."""
        self._seed_pick(
            user_id=self.user_a_id,
            game_id=self.game_spread_id,
            week_id=self.week_id,
            pick_type=PickType.FAVORITE_COVER,
            is_mortal_lock=True,
        )
        headers = self._cookie_auth_headers(self.user_a_id)
        resp = self.client.post(
            "/api/picks",
            json={
                "season": SEASON,
                "week": WEEK,
                "picks": [
                    {
                        "game_id": self.game_total_id,
                        "pick_type": "OVER",
                        "is_mortal_lock": True,
                    }
                ],
            },
            headers=headers,
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("reason"), "MULTIPLE_MORTAL_LOCKS")
        # Only the original mortal lock persists.
        rows = self._picks_for(self.user_a_id, self.week_id)
        self.assertEqual(len(rows), 1)

    def test_roster_duplicate_base_type_rejected(self) -> None:
        """A second base pick of the SAME type on the same game -> 409."""
        self._seed_pick(
            user_id=self.user_a_id,
            game_id=self.game_spread_id,
            week_id=self.week_id,
            pick_type=PickType.OVER,
        )
        headers = self._cookie_auth_headers(self.user_a_id)
        resp = self.client.post(
            "/api/picks",
            json={
                "season": SEASON,
                "week": WEEK,
                "picks": [{"game_id": self.game_spread_id, "pick_type": "OVER"}],
            },
            headers=headers,
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("reason"), "DUPLICATE_PICK")
        rows = self._picks_for(self.user_a_id, self.week_id)
        self.assertEqual(len(rows), 1)

    def test_pickem_spread_ineligible_rejected(self) -> None:
        """A spread pick on a true pick'em game -> 422 envelope."""
        headers = self._cookie_auth_headers(self.user_a_id)
        resp = self.client.post(
            "/api/picks",
            json={
                "season": SEASON,
                "week": WEEK,
                "picks": [
                    {
                        "game_id": self.game_pickem_id,
                        "pick_type": "FAVORITE_COVER",
                    }
                ],
            },
            headers=headers,
        )
        self.assertEqual(resp.status_code, 422, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("reason"), "PICKEM_SPREAD_INELIGIBLE")
        self.assertEqual(self._picks_for(self.user_a_id, self.week_id), [])

    def test_read_returns_only_callers_picks(self) -> None:
        """A read returns only the caller's picks — never another user's."""
        self._seed_pick(
            user_id=self.user_a_id,
            game_id=self.game_spread_id,
            week_id=self.week_id,
            pick_type=PickType.FAVORITE_COVER,
        )
        self._seed_pick(
            user_id=self.user_b_id,
            game_id=self.game_total_id,
            week_id=self.week_id,
            pick_type=PickType.UNDER,
        )

        # userA (bearer read) sees only their FAVORITE_COVER.
        resp_a = self.client.get(
            "/api/picks",
            params={"season": SEASON, "week": WEEK},
            headers=self._bearer_headers(self.user_a_id),
        )
        self.assertEqual(resp_a.status_code, 200, resp_a.text)
        out_a = resp_a.json()
        self.assertEqual(len(out_a), 1)
        self.assertEqual(out_a[0]["pick_type"], PickType.FAVORITE_COVER.value)
        self.assertEqual(out_a[0]["game_id"], self.game_spread_id)

        # userB sees only their UNDER — a user cannot read another's picks.
        resp_b = self.client.get(
            "/api/picks",
            params={"season": SEASON, "week": WEEK},
            headers=self._bearer_headers(self.user_b_id),
        )
        self.assertEqual(resp_b.status_code, 200, resp_b.text)
        out_b = resp_b.json()
        self.assertEqual(len(out_b), 1)
        self.assertEqual(out_b[0]["pick_type"], PickType.UNDER.value)
        self.assertEqual(out_b[0]["game_id"], self.game_total_id)

    def test_unauthenticated_requests_rejected_401(self) -> None:
        """Unauthenticated POST and GET both -> 401 envelope."""
        self._clear_auth()
        post = self.client.post(
            "/api/picks",
            json={
                "season": SEASON,
                "week": WEEK,
                "picks": [
                    {"game_id": self.game_spread_id, "pick_type": "FAVORITE_COVER"}
                ],
            },
        )
        self.assertEqual(post.status_code, 401, post.text)
        self._assert_envelope(post.json())

        self._clear_auth()
        get = self.client.get(
            "/api/picks", params={"season": SEASON, "week": WEEK}
        )
        self.assertEqual(get.status_code, 401, get.text)
        self._assert_envelope(get.json())

    def test_user_cannot_write_or_read_anothers_picks(self) -> None:
        """user_id is derived from the session, never the body.

        Even if a client could shape a body, there is no user field; and a read
        is always scoped to the authenticated user. We assert that a POST as
        userA persists under userA (not userB), and userB's read never sees it.
        """
        headers = self._cookie_auth_headers(self.user_a_id)
        resp = self.client.post(
            "/api/picks",
            json={
                "season": SEASON,
                "week": WEEK,
                "picks": [
                    {"game_id": self.game_spread_id, "pick_type": "FAVORITE_COVER"}
                ],
            },
            headers=headers,
        )
        self.assertEqual(resp.status_code, 200, resp.text)

        # Persisted under userA only.
        self.assertEqual(len(self._picks_for(self.user_a_id, self.week_id)), 1)
        self.assertEqual(self._picks_for(self.user_b_id, self.week_id), [])

        # userB's read is empty — cannot see userA's pick.
        self._clear_auth()
        read_b = self.client.get(
            "/api/picks",
            params={"season": SEASON, "week": WEEK},
            headers=self._bearer_headers(self.user_b_id),
        )
        self.assertEqual(read_b.status_code, 200, read_b.text)
        self.assertEqual(read_b.json(), [])

    def test_rejections_are_structured_4xx_never_500(self) -> None:
        """Every rejection path is a 4xx envelope — never a raw 500."""
        headers = self._cookie_auth_headers(self.user_a_id)

        # A spread pick on the true pick'em game -> 422 envelope (a rejection).
        resp = self.client.post(
            "/api/picks",
            json={
                "season": SEASON,
                "week": WEEK,
                "picks": [
                    {
                        "game_id": self.game_pickem_id,
                        "pick_type": "UNDERDOG_COVER",
                    }
                ],
            },
            headers=headers,
        )
        self.assertGreaterEqual(resp.status_code, 400)
        self.assertLess(resp.status_code, 500, resp.text)
        self._assert_envelope(resp.json())

        # A closed-window submit -> also 4xx, also enveloped, never 500.
        resp2 = self.client.post(
            "/api/picks",
            json={
                "season": SEASON,
                "week": WEEK + 1,
                "picks": [
                    {"game_id": self.game_locked_id, "pick_type": "FAVORITE_COVER"}
                ],
            },
            headers=headers,
        )
        self.assertGreaterEqual(resp2.status_code, 400)
        self.assertLess(resp2.status_code, 500, resp2.text)
        self._assert_envelope(resp2.json())


if __name__ == "__main__":
    unittest.main()
