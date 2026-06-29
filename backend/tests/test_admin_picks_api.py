"""Offline tests for the admin pick-override surface (QT-1).

Fully OFFLINE (in-memory SQLite ``StaticPool``, no Postgres, no network), mirroring
:mod:`tests.test_picks_api` (fixture shape) and :mod:`tests.test_admin_api` (auth
helpers + the ``PRAGMA foreign_keys=ON`` connect listener registered BEFORE
``create_all``).

What these tests pin (the QT-1 spec):

* add-when-absent, change-existing (upsert, not duplicate), clear, set/clear of the
  mortal-lock slot;
* roster integrity is STILL enforced on the admin path (409 duplicate/contradiction
  /2nd-mortal-lock; 422 spread-on-pick'em) — only window/lock is bypassed;
* WINDOW/LOCK BYPASS PROVEN — a PUT succeeds (200) on a CLOSED-window / past-kickoff
  game AND on a FINAL game, where the SAME input is a 409 on the user-facing
  /api/picks path (asserted by contrast);
* every override writes exactly one PickEditAudit row with the right who/whom,
  before/after, and ``game_was_final``;
* 403 (non-admin) / 401 (unauthenticated) on all three verbs.

> Run from backend/ with ``.venv/bin/python -m unittest`` (unittest, NOT pytest).
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import mock

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.csrf import CSRF_COOKIE_NAME, CSRF_HEADER_NAME
from app.db import get_session
from app.main import app
from app.models import (
    Game,
    GameStatus,
    Pick,
    PickEditAudit,
    PickResult,
    PickType,
    Team,
    User,
    Week,
)
from app.services.auth import create_session_cookie, hash_password

SEASON = 2025
WEEK = 1

_FUTURE = timedelta(days=2)
_PAST = timedelta(hours=2)


def _enable_sqlite_fks(dbapi_connection, _connection_record):  # noqa: ANN001
    """Connect listener: turn SQLite FK (and cascade) enforcement ON."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class AdminPicksApiTests(unittest.TestCase):
    """Offline TestClient coverage for /api/admin/users/{user_id}/picks."""

    admin_id: int
    member_id: int
    target_id: int
    week_id: int
    game_open_id: int      # future kickoff, spread+total (open window)
    game_total_id: int     # future kickoff, totals (open window)
    game_pickem_id: int    # future kickoff, true pick'em (spread ineligible)
    game_locked_id: int    # PAST kickoff, IN_PROGRESS (closed window / locked)
    game_final_id: int     # PAST kickoff, FINAL

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        event.listen(self.engine, "connect", _enable_sqlite_fks)
        SQLModel.metadata.create_all(self.engine)

        now = datetime.now(timezone.utc)
        pw = hash_password("correct horse battery staple")
        with Session(self.engine) as session:
            # Distinct discord_ids: the one-null-discord_id invariant (260629-n59)
            # caps NULL discord_ids at one.
            admin = User(display_name="admin", password_hash=pw, is_admin=True, is_active=True, discord_id=1)
            member = User(display_name="member", password_hash=pw, is_admin=False, is_active=True, discord_id=2)
            target = User(display_name="target", password_hash=pw, is_admin=False, is_active=True, discord_id=3)
            session.add_all([admin, member, target])
            session.commit()
            for u in (admin, member, target):
                session.refresh(u)
            self.admin_id = admin.id
            self.member_id = member.id
            self.target_id = target.id

            teams = [
                Team(espn_team_id=i, abbreviation=f"T{i}", display_name=f"Team {i}")
                for i in range(1, 11)
            ]
            session.add_all(teams)
            session.commit()
            for t in teams:
                session.refresh(t)
            tid = [t.id for t in teams]

            # Open-window week: its EARLIEST kickoff is in the future. The
            # closed/locked and FINAL games live in their OWN weeks so their PAST
            # kickoffs do not pull WEEK 1's window closed.
            week = Week(season=SEASON, week=WEEK)
            session.add(week)
            session.commit()
            session.refresh(week)
            self.week_id = week.id

            game_open = Game(
                espn_event_id=1001, week_id=week.id, season=SEASON, week=WEEK,
                home_team_id=tid[0], away_team_id=tid[1],
                kickoff_at=now + _FUTURE, status=GameStatus.SCHEDULED,
                spread=Decimal("3.5"), total=Decimal("44.5"),
                favorite_team_id=tid[0], underdog_team_id=tid[1],
            )
            game_total = Game(
                espn_event_id=1002, week_id=week.id, season=SEASON, week=WEEK,
                home_team_id=tid[2], away_team_id=tid[3],
                kickoff_at=now + _FUTURE + timedelta(hours=3),
                status=GameStatus.SCHEDULED,
                spread=Decimal("6.5"), total=Decimal("41.0"),
                favorite_team_id=tid[2], underdog_team_id=tid[3],
            )
            game_pickem = Game(
                espn_event_id=1003, week_id=week.id, season=SEASON, week=WEEK,
                home_team_id=tid[4], away_team_id=tid[5],
                kickoff_at=now + _FUTURE + timedelta(hours=6),
                status=GameStatus.SCHEDULED,
                spread=Decimal("0.0"), total=Decimal("48.0"),
                favorite_team_id=None, underdog_team_id=None,
            )
            session.add_all([game_open, game_total, game_pickem])

            # Closed-window / locked week (only game already kicked off, past).
            locked_week = Week(season=SEASON, week=WEEK + 1)
            session.add(locked_week)
            session.commit()
            session.refresh(locked_week)
            self.locked_week_id = locked_week.id
            game_locked = Game(
                espn_event_id=2001, week_id=locked_week.id, season=SEASON, week=WEEK + 1,
                home_team_id=tid[6], away_team_id=tid[7],
                kickoff_at=now - _PAST, status=GameStatus.IN_PROGRESS,
                spread=Decimal("2.5"), total=Decimal("40.0"),
                favorite_team_id=tid[6], underdog_team_id=tid[7],
            )
            session.add(game_locked)

            # FINAL week (past kickoff, status FINAL).
            final_week = Week(season=SEASON, week=WEEK + 2)
            session.add(final_week)
            session.commit()
            session.refresh(final_week)
            self.final_week_id = final_week.id
            game_final = Game(
                espn_event_id=3001, week_id=final_week.id, season=SEASON, week=WEEK + 2,
                home_team_id=tid[8], away_team_id=tid[9],
                kickoff_at=now - _PAST, status=GameStatus.FINAL,
                spread=Decimal("4.0"), total=Decimal("45.0"),
                favorite_team_id=tid[8], underdog_team_id=tid[9],
            )
            session.add(game_final)
            session.commit()

            for g in (game_open, game_total, game_pickem, game_locked, game_final):
                session.refresh(g)
            self.game_open_id = game_open.id
            self.game_total_id = game_total.id
            self.game_pickem_id = game_pickem.id
            self.game_locked_id = game_locked.id
            self.game_final_id = game_final.id

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

    def _cookie_auth_headers(self, user_id: int) -> dict[str, str]:
        csrf_value = "test-csrf-token-value"
        self.client.cookies.set("session", create_session_cookie(user_id))
        self.client.cookies.set(CSRF_COOKIE_NAME, csrf_value)
        return {CSRF_HEADER_NAME: csrf_value}

    def _bearer_headers(self, user_id: int) -> dict[str, str]:
        return {"Authorization": f"Bearer {create_session_cookie(user_id)}"}

    def _clear_auth(self) -> None:
        self.client.cookies.clear()

    def _picks_for(self, user_id: int, week_id: int) -> list[Pick]:
        with self._session() as session:
            return list(
                session.exec(
                    select(Pick).where(
                        Pick.user_id == user_id, Pick.week_id == week_id
                    )
                ).all()
            )

    def _audits(self) -> list[PickEditAudit]:
        with self._session() as session:
            return list(session.exec(select(PickEditAudit)).all())

    def _seed_pick(
        self, *, user_id: int, game_id: int, week_id: int,
        pick_type: PickType, is_mortal_lock: bool = False,
    ) -> int:
        with self._session() as session:
            pick = Pick(
                user_id=user_id, game_id=game_id, week_id=week_id,
                pick_type=pick_type, is_mortal_lock=is_mortal_lock,
            )
            session.add(pick)
            session.commit()
            session.refresh(pick)
            return pick.id

    @staticmethod
    def _assert_envelope(body: dict) -> dict:
        assert "error" in body, f"expected an error envelope, got: {body}"
        err = body["error"]
        assert "code" in err, f"envelope missing 'code': {err}"
        return err

    def _put(self, user_id: int, *, season: int, week: int, body: dict, as_user: int):
        return self.client.put(
            f"/api/admin/users/{user_id}/picks",
            params={"season": season, "week": week},
            json=body,
            headers=self._cookie_auth_headers(as_user),
        )

    def _delete(self, user_id: int, *, season: int, week: int, pick_type: str,
                is_mortal_lock: bool, as_user: int):
        return self.client.delete(
            f"/api/admin/users/{user_id}/picks",
            params={
                "season": season, "week": week,
                "pick_type": pick_type, "is_mortal_lock": is_mortal_lock,
            },
            headers=self._cookie_auth_headers(as_user),
        )

    def _grade(self, user_id: int, *, season: int, week: int, body: dict,
               as_user: int):
        return self.client.put(
            f"/api/admin/users/{user_id}/picks/misc-grade",
            params={"season": season, "week": week},
            json=body,
            headers=self._cookie_auth_headers(as_user),
        )

    # -- add / change / clear ---------------------------------------------

    def test_add_pick_when_absent(self) -> None:
        """PUT a pick the target does not have -> 200, row created."""
        resp = self._put(
            self.target_id, season=SEASON, week=WEEK,
            body={"game_id": self.game_open_id, "pick_type": "FAVORITE_COVER"},
            as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["game_id"], self.game_open_id)
        self.assertEqual(body["pick_type"], "FAVORITE_COVER")
        rows = self._picks_for(self.target_id, self.week_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].game_id, self.game_open_id)

    def test_change_existing_pick_upserts_not_duplicates(self) -> None:
        """PUT the same slot a different game -> 200; one row, game_id updated."""
        self._seed_pick(
            user_id=self.target_id, game_id=self.game_open_id,
            week_id=self.week_id, pick_type=PickType.OVER,
        )
        resp = self._put(
            self.target_id, season=SEASON, week=WEEK,
            body={"game_id": self.game_total_id, "pick_type": "OVER"},
            as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        rows = self._picks_for(self.target_id, self.week_id)
        self.assertEqual(len(rows), 1, "should upsert the slot, not duplicate")
        self.assertEqual(rows[0].game_id, self.game_total_id)

    def test_clear_existing_pick(self) -> None:
        """DELETE an existing slot -> 204; row gone."""
        self._seed_pick(
            user_id=self.target_id, game_id=self.game_open_id,
            week_id=self.week_id, pick_type=PickType.FAVORITE_COVER,
        )
        resp = self._delete(
            self.target_id, season=SEASON, week=WEEK,
            pick_type="FAVORITE_COVER", is_mortal_lock=False, as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 204, resp.text)
        self.assertEqual(self._picks_for(self.target_id, self.week_id), [])

    def test_set_and_clear_mortal_lock(self) -> None:
        """PUT is_mortal_lock=true -> 200; DELETE is_mortal_lock=true -> 204."""
        put = self._put(
            self.target_id, season=SEASON, week=WEEK,
            body={"game_id": self.game_open_id, "pick_type": "FAVORITE_COVER",
                  "is_mortal_lock": True},
            as_user=self.admin_id,
        )
        self.assertEqual(put.status_code, 200, put.text)
        self.assertTrue(put.json()["is_mortal_lock"])
        rows = self._picks_for(self.target_id, self.week_id)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].is_mortal_lock)

        delete = self._delete(
            self.target_id, season=SEASON, week=WEEK,
            pick_type="FAVORITE_COVER", is_mortal_lock=True, as_user=self.admin_id,
        )
        self.assertEqual(delete.status_code, 204, delete.text)
        self.assertEqual(self._picks_for(self.target_id, self.week_id), [])

    # -- roster integrity STILL enforced ----------------------------------

    def test_roster_contradiction_still_rejected_409(self) -> None:
        """A same-game contradiction is STILL 409 on the admin path; no write."""
        self._seed_pick(
            user_id=self.target_id, game_id=self.game_open_id,
            week_id=self.week_id, pick_type=PickType.FAVORITE_COVER,
        )
        resp = self._put(
            self.target_id, season=SEASON, week=WEEK,
            body={"game_id": self.game_open_id, "pick_type": "UNDERDOG_COVER"},
            as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("reason"), "CONTRADICTORY_PICK")
        rows = self._picks_for(self.target_id, self.week_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].pick_type, PickType.FAVORITE_COVER)

    def test_roster_second_mortal_lock_still_rejected_409(self) -> None:
        """A 2nd mortal lock is STILL 409 on the admin path; no write."""
        self._seed_pick(
            user_id=self.target_id, game_id=self.game_open_id,
            week_id=self.week_id, pick_type=PickType.FAVORITE_COVER,
            is_mortal_lock=True,
        )
        resp = self._put(
            self.target_id, season=SEASON, week=WEEK,
            body={"game_id": self.game_total_id, "pick_type": "OVER",
                  "is_mortal_lock": True},
            as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("reason"), "MULTIPLE_MORTAL_LOCKS")
        rows = self._picks_for(self.target_id, self.week_id)
        self.assertEqual(len(rows), 1)

    def test_pickem_spread_ineligible_still_rejected_422(self) -> None:
        """A spread pick on a true pick'em is STILL 422 on the admin path."""
        resp = self._put(
            self.target_id, season=SEASON, week=WEEK,
            body={"game_id": self.game_pickem_id, "pick_type": "FAVORITE_COVER"},
            as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 422, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("reason"), "PICKEM_SPREAD_INELIGIBLE")
        self.assertEqual(self._picks_for(self.target_id, self.week_id), [])

    # -- WINDOW/LOCK BYPASS PROVEN ----------------------------------------

    def test_admin_set_bypasses_closed_window_and_lock(self) -> None:
        """PUT succeeds (200) on a CLOSED-window / past-kickoff locked game.

        The SAME input is a 409 on the user-facing /api/picks path (asserted by
        contrast) — proving the admin path bypasses window/lock.
        """
        resp = self._put(
            self.target_id, season=SEASON, week=WEEK + 1,
            body={"game_id": self.game_locked_id, "pick_type": "FAVORITE_COVER"},
            as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        rows = self._picks_for(self.target_id, self.locked_week_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].game_id, self.game_locked_id)

        # Contrast: the user-facing path rejects the SAME pick (window closed).
        user_resp = self.client.post(
            "/api/picks",
            json={"season": SEASON, "week": WEEK + 1,
                  "picks": [{"game_id": self.game_locked_id,
                             "pick_type": "UNDER"}]},
            headers=self._cookie_auth_headers(self.target_id),
        )
        self.assertEqual(user_resp.status_code, 409, user_resp.text)
        self.assertIn(
            self._assert_envelope(user_resp.json()).get("reason"),
            {"window_closed", "game_locked"},
        )

    def test_admin_set_succeeds_on_final_game(self) -> None:
        """PUT succeeds (200) on a FINAL game (past kickoff, status FINAL)."""
        resp = self._put(
            self.target_id, season=SEASON, week=WEEK + 2,
            body={"game_id": self.game_final_id, "pick_type": "FAVORITE_COVER"},
            as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        rows = self._picks_for(self.target_id, self.final_week_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].game_id, self.game_final_id)

    # -- audit rows --------------------------------------------------------

    def test_audit_row_written_on_set_final_game(self) -> None:
        """A successful set on the FINAL game writes one audit row, final=True."""
        resp = self._put(
            self.target_id, season=SEASON, week=WEEK + 2,
            body={"game_id": self.game_final_id, "pick_type": "OVER",
                  "is_mortal_lock": False},
            as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        audits = self._audits()
        self.assertEqual(len(audits), 1)
        a = audits[0]
        self.assertEqual(a.admin_user_id, self.admin_id)
        self.assertEqual(a.target_user_id, self.target_id)
        self.assertEqual(a.action, "set")
        self.assertFalse(a.before_existed)
        self.assertIsNone(a.before_pick_type)
        self.assertEqual(a.after_pick_type, PickType.OVER)
        self.assertFalse(a.after_is_mortal_lock)
        self.assertTrue(a.game_was_final)
        self.assertEqual(a.game_id, self.game_final_id)

    def test_audit_row_written_on_clear(self) -> None:
        """A clear writes one action='clear' row, before_existed True, after None."""
        self._seed_pick(
            user_id=self.target_id, game_id=self.game_open_id,
            week_id=self.week_id, pick_type=PickType.FAVORITE_COVER,
        )
        resp = self._delete(
            self.target_id, season=SEASON, week=WEEK,
            pick_type="FAVORITE_COVER", is_mortal_lock=False, as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 204, resp.text)
        audits = self._audits()
        self.assertEqual(len(audits), 1)
        a = audits[0]
        self.assertEqual(a.action, "clear")
        self.assertTrue(a.before_existed)
        self.assertEqual(a.before_pick_type, PickType.FAVORITE_COVER)
        self.assertIsNone(a.after_pick_type)
        self.assertIsNone(a.after_is_mortal_lock)
        self.assertFalse(a.game_was_final)

    def test_audit_not_cascaded_on_user_delete(self) -> None:
        """Audit rows are NOT cascaded away when a user is deleted.

        The two user FKs on ``pick_edit_audit`` carry NO ``ondelete`` (NO ACTION,
        the OPPOSITE of ``pick.user_id``'s CASCADE), so the audit is a permanent
        record. With SQLite FK enforcement ON (as in production Postgres), the DB
        therefore REFUSES to delete a user still referenced by an audit row rather
        than silently cascading the audit away — the audit always survives.

        Contrast with the target's OWN pick, which DOES cascade: we delete that
        pick directly to prove the cascade path leaves the audit untouched.
        """
        resp = self._put(
            self.target_id, season=SEASON, week=WEEK,
            body={"game_id": self.game_open_id, "pick_type": "FAVORITE_COVER"},
            as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(len(self._audits()), 1)

        # Deleting a user referenced by an audit row is BLOCKED (NO ACTION FK) —
        # the audit is never cascaded away.
        from sqlalchemy.exc import IntegrityError

        with self._session() as session:
            user = session.get(User, self.target_id)
            session.delete(user)
            with self.assertRaises(IntegrityError):
                session.commit()
            session.rollback()

        # The audit row is intact, and the user still exists (delete refused).
        self.assertEqual(len(self._audits()), 1)
        with self._session() as session:
            self.assertIsNotNone(session.get(User, self.target_id))

        # The target's OWN pick cascades cleanly without touching the audit.
        with self._session() as session:
            for p in session.exec(
                select(Pick).where(Pick.user_id == self.target_id)
            ).all():
                session.delete(p)
            session.commit()
        self.assertEqual(self._picks_for(self.target_id, self.week_id), [])
        self.assertEqual(len(self._audits()), 1)

    # -- missing target user -> clean 404 (not an FK 500) ------------------

    def test_set_missing_user_404(self) -> None:
        """PUT a pick for a non-existent target_user_id -> 404, no FK 500.

        A typo'd / missing target user must be a clean ``user_not_found`` 404
        BEFORE any pick/audit add, not an FK IntegrityError at commit. No audit
        row is written.
        """
        resp = self._put(
            999999, season=SEASON, week=WEEK,
            body={"game_id": self.game_open_id, "pick_type": "FAVORITE_COVER"},
            as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("reason"), "user_not_found")
        self.assertEqual(self._audits(), [], "no audit row for a missing user")

    def test_clear_missing_user_404(self) -> None:
        """DELETE a slot for a non-existent target_user_id -> 404 (user_not_found)."""
        resp = self._delete(
            999999, season=SEASON, week=WEEK,
            pick_type="FAVORITE_COVER", is_mortal_lock=False,
            as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("reason"), "user_not_found")

    def test_grade_missing_user_404(self) -> None:
        """PUT misc-grade for a non-existent target_user_id -> 404 (user_not_found)."""
        resp = self._grade(
            999999, season=SEASON, week=WEEK,
            body={"result": "WIN", "points": 3}, as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("reason"), "user_not_found")

    # -- read --------------------------------------------------------------

    def test_get_returns_target_user_picks(self) -> None:
        """GET returns the PATH user's roster for the week."""
        self._seed_pick(
            user_id=self.target_id, game_id=self.game_open_id,
            week_id=self.week_id, pick_type=PickType.FAVORITE_COVER,
        )
        resp = self.client.get(
            f"/api/admin/users/{self.target_id}/picks",
            params={"season": SEASON, "week": WEEK},
            headers=self._bearer_headers(self.admin_id),
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        out = resp.json()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["pick_type"], "FAVORITE_COVER")

    # -- 403 / 401 ---------------------------------------------------------

    def test_non_admin_forbidden_403(self) -> None:
        """GET/PUT/DELETE as a non-admin member -> 403."""
        get = self.client.get(
            f"/api/admin/users/{self.target_id}/picks",
            params={"season": SEASON, "week": WEEK},
            headers=self._bearer_headers(self.member_id),
        )
        self.assertEqual(get.status_code, 403, get.text)
        self._assert_envelope(get.json())

        put = self._put(
            self.target_id, season=SEASON, week=WEEK,
            body={"game_id": self.game_open_id, "pick_type": "FAVORITE_COVER"},
            as_user=self.member_id,
        )
        self.assertEqual(put.status_code, 403, put.text)
        self._assert_envelope(put.json())

        delete = self._delete(
            self.target_id, season=SEASON, week=WEEK,
            pick_type="FAVORITE_COVER", is_mortal_lock=False, as_user=self.member_id,
        )
        self.assertEqual(delete.status_code, 403, delete.text)
        self._assert_envelope(delete.json())

    def test_unauthenticated_401(self) -> None:
        """GET/PUT/DELETE with no auth -> 401."""
        self._clear_auth()
        get = self.client.get(
            f"/api/admin/users/{self.target_id}/picks",
            params={"season": SEASON, "week": WEEK},
        )
        self.assertEqual(get.status_code, 401, get.text)
        self._assert_envelope(get.json())

        self._clear_auth()
        put = self.client.put(
            f"/api/admin/users/{self.target_id}/picks",
            params={"season": SEASON, "week": WEEK},
            json={"game_id": self.game_open_id, "pick_type": "FAVORITE_COVER"},
        )
        self.assertEqual(put.status_code, 401, put.text)
        self._assert_envelope(put.json())

        self._clear_auth()
        delete = self.client.delete(
            f"/api/admin/users/{self.target_id}/picks",
            params={"season": SEASON, "week": WEEK,
                    "pick_type": "FAVORITE_COVER", "is_mortal_lock": False},
        )
        self.assertEqual(delete.status_code, 401, delete.text)
        self._assert_envelope(delete.json())


    # -- MISC retroactive create + grade -----------------------------------

    def test_admin_retroactive_create_misc_writes_audit(self) -> None:
        """Admin sets a MISC pick (with text) for a user via PUT .../picks.

        200, pick persisted with misc_text, exactly one action='set' audit row
        with the right admin/target and game_was_final.
        """
        resp = self._put(
            self.target_id, season=SEASON, week=WEEK + 2,
            body={"game_id": self.game_final_id, "pick_type": "MISC",
                  "misc_text": "Mahomes throws for 400 yards"},
            as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["pick_type"], "MISC")
        self.assertEqual(body["misc_text"], "Mahomes throws for 400 yards")

        rows = self._picks_for(self.target_id, self.final_week_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].misc_text, "Mahomes throws for 400 yards")

        audits = self._audits()
        self.assertEqual(len(audits), 1)
        a = audits[0]
        self.assertEqual(a.admin_user_id, self.admin_id)
        self.assertEqual(a.target_user_id, self.target_id)
        self.assertEqual(a.action, "set")
        self.assertEqual(a.after_pick_type, PickType.MISC)
        self.assertTrue(a.game_was_final)

    def test_admin_grade_misc_sets_result_points_and_audits(self) -> None:
        """Grade a MISC pick via PUT .../picks/misc-grade -> 200, result/points set.

        A SECOND audit row is written (the create wrote the first).
        """
        # Retroactive create first (writes audit #1).
        self._put(
            self.target_id, season=SEASON, week=WEEK + 2,
            body={"game_id": self.game_final_id, "pick_type": "MISC",
                  "misc_text": "a prediction"},
            as_user=self.admin_id,
        )
        self.assertEqual(len(self._audits()), 1)

        resp = self._grade(
            self.target_id, season=SEASON, week=WEEK + 2,
            body={"result": "WIN", "points": 3}, as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["result"], "WIN")
        self.assertEqual(body["points"], 3)

        rows = self._picks_for(self.target_id, self.final_week_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].result, PickResult.WIN)
        self.assertEqual(rows[0].points, 3)

        # The grade wrote a SECOND audit row (who graded whose pick).
        audits = self._audits()
        self.assertEqual(len(audits), 2)
        self.assertTrue(all(a.admin_user_id == self.admin_id for a in audits))

    def test_grade_misc_on_closed_window_publishes_chat_event(self) -> None:
        """Grading a MISC pick on a CLOSED-window week publishes one misc.graded.

        Week WEEK+2's only game has a PAST kickoff (window CLOSED), so the leak
        guard lets the chat publish through. Exactly one event with the display
        fields is captured.
        """
        # Retroactive create on the FINAL (closed-window) week, then grade WIN.
        self._put(
            self.target_id, season=SEASON, week=WEEK + 2,
            body={"game_id": self.game_final_id, "pick_type": "MISC",
                  "misc_text": "Mahomes throws 4 TDs"},
            as_user=self.admin_id,
        )
        captured: list[dict] = []
        with mock.patch(
            "app.api.admin.publish_event", side_effect=captured.append
        ):
            resp = self._grade(
                self.target_id, season=SEASON, week=WEEK + 2,
                body={"result": "WIN", "points": 3}, as_user=self.admin_id,
            )
        self.assertEqual(resp.status_code, 200, resp.text)

        misc_events = [e for e in captured if e.get("type") == "misc.graded"]
        self.assertEqual(len(misc_events), 1, captured)
        event = misc_events[0]
        self.assertEqual(event["targets"], ["chat"])
        self.assertEqual(event["actor"], "target")
        self.assertEqual(event["week"], WEEK + 2)
        self.assertEqual(event["prediction"], "Mahomes throws 4 TDs")
        self.assertEqual(event["verdict"], "correct")
        self.assertEqual(event["points"], 3)

    def test_grade_misc_on_open_window_publishes_nothing(self) -> None:
        """Grading a MISC pick on an OPEN-window week publishes NO chat event.

        Week WEEK's games have FUTURE kickoffs (window OPEN), so the leak guard
        suppresses the misc.graded publish (the prediction is hidden-until-lock).
        The grade still returns 200 and persists result/points.
        """
        # Retroactive create on the OPEN-window base week, then grade LOSS.
        self._put(
            self.target_id, season=SEASON, week=WEEK,
            body={"game_id": self.game_open_id, "pick_type": "MISC",
                  "misc_text": "a bold prediction"},
            as_user=self.admin_id,
        )
        captured: list[dict] = []
        with mock.patch(
            "app.api.admin.publish_event", side_effect=captured.append
        ):
            resp = self._grade(
                self.target_id, season=SEASON, week=WEEK,
                body={"result": "LOSS", "points": -1}, as_user=self.admin_id,
            )
        # The grade itself still succeeds and persists.
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["result"], "LOSS")
        self.assertEqual(body["points"], -1)
        rows = self._picks_for(self.target_id, self.week_id)
        misc_rows = [r for r in rows if r.pick_type == PickType.MISC]
        self.assertEqual(len(misc_rows), 1)
        self.assertEqual(misc_rows[0].result, PickResult.LOSS)
        self.assertEqual(misc_rows[0].points, -1)
        # The leak guard suppressed the chat publish.
        misc_events = [e for e in captured if e.get("type") == "misc.graded"]
        self.assertEqual(misc_events, [], captured)

    def test_graded_misc_survives_recompute(self) -> None:
        """A live week_results recompute over a graded MISC keeps the admin points.

        Proves the auto-grade path does NOT overwrite the stored MISC grade.
        """
        self._put(
            self.target_id, season=SEASON, week=WEEK + 2,
            body={"game_id": self.game_final_id, "pick_type": "MISC",
                  "misc_text": "a prediction"},
            as_user=self.admin_id,
        )
        self._grade(
            self.target_id, season=SEASON, week=WEEK + 2,
            body={"result": "WIN", "points": 3}, as_user=self.admin_id,
        )

        # Recompute the week server-side (the recompute-on-read path).
        from app.services.standings import week_results

        with self._session() as session:
            results = week_results(session, season=SEASON, week=WEEK + 2)
        target = next(r for r in results if r.display_name == "target")
        self.assertEqual(target.weekly_score, 3)
        misc_entries = [p for p in target.picks if p.pick_type == PickType.MISC]
        self.assertEqual(len(misc_entries), 1)
        self.assertEqual(misc_entries[0].points, 3)
        self.assertEqual(misc_entries[0].outcome, "WIN")
        # The MISC game is FINAL (locked), so the revealed entry carries the text.
        self.assertEqual(misc_entries[0].misc_text, "a prediction")

    def test_grade_pending_rejected_422(self) -> None:
        """Grading with result=PENDING -> 422 (misc_grade_must_decide)."""
        self._put(
            self.target_id, season=SEASON, week=WEEK + 2,
            body={"game_id": self.game_final_id, "pick_type": "MISC",
                  "misc_text": "a prediction"},
            as_user=self.admin_id,
        )
        resp = self._grade(
            self.target_id, season=SEASON, week=WEEK + 2,
            body={"result": "PENDING", "points": 0}, as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 422, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("reason"), "misc_grade_must_decide")

    def test_grade_missing_pick_404(self) -> None:
        """Grading when the user has no MISC pick -> 404 (pick_not_found)."""
        resp = self._grade(
            self.target_id, season=SEASON, week=WEEK + 2,
            body={"result": "WIN", "points": 3}, as_user=self.admin_id,
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("reason"), "pick_not_found")

    def test_grade_non_admin_forbidden_403(self) -> None:
        """The grade endpoint rejects a non-admin member -> 403."""
        resp = self._grade(
            self.target_id, season=SEASON, week=WEEK + 2,
            body={"result": "WIN", "points": 3}, as_user=self.member_id,
        )
        self.assertEqual(resp.status_code, 403, resp.text)
        self._assert_envelope(resp.json())

    def test_grade_unauthenticated_401(self) -> None:
        """The grade endpoint rejects an unauthenticated request -> 401."""
        self._clear_auth()
        resp = self.client.put(
            f"/api/admin/users/{self.target_id}/picks/misc-grade",
            params={"season": SEASON, "week": WEEK + 2},
            json={"result": "WIN", "points": 3},
        )
        self.assertEqual(resp.status_code, 401, resp.text)
        self._assert_envelope(resp.json())


if __name__ == "__main__":
    unittest.main()
