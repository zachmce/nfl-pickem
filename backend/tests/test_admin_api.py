"""Offline tests for the web admin user-management surface (QT-C).

Two layers, both fully OFFLINE (in-memory SQLite, no Postgres, no network):

* :class:`AdminApiTests` — HTTP tests over the /api/admin router mirroring
  :mod:`tests.test_picks_api` (``StaticPool`` engine, ``dependency_overrides``,
  ``TestClient``, cookie+CSRF for mutations / bearer for the GET). Covers every
  action happy path (the pinned 200-AdminUserRead / 204 contract), the three
  self-guards, 403 (non-admin), 401 (unauthenticated), and the delete cascade.

* :class:`AdminLastAdminGuardTests` — SERVICE-layer tests for the three
  last-admin guards. The guards are only genuinely reachable when the caller is
  DISTINCT from the sole-admin target; over HTTP the caller is always a
  session-resolved admin, so a distinct caller implies a second admin exists and
  the count is never 1 — meaning at the HTTP layer the last-admin branch is only
  ever reached as a self-action, which the self-guard pre-empts (see
  ``test_*_last_admin_pre_empted_by_self_guard`` in the HTTP class). To exercise
  the last-admin BRANCH itself (count==1, distinct caller), these tests call the
  service directly with a synthetic ``caller_id`` — the service treats caller_id
  as a pure parameter. This also pins the DISTINCT counting rules: revoke/delete
  count ALL ``is_admin=True``; deactivate counts ACTIVE admins only (proved with a
  fixture that has one ACTIVE admin and a separate INACTIVE admin).

CRITICAL (cascade): plain ``test_picks_api`` does NOT enable SQLite FK
enforcement, so a copy of it would let SQLite SILENTLY IGNORE the QT-A
``ON DELETE CASCADE`` and the cascade assertion would falsely pass even if the
cascade were broken. The ``@event.listens_for(engine, "connect")`` handler issues
``PRAGMA foreign_keys=ON`` on EVERY DBAPI connection (registered BEFORE
``create_all`` so the schema-building connection is covered) — that is what makes
``test_delete_cascades_user_picks`` a genuine cascade test.

> Run from backend/ with ``.venv/bin/python -m unittest`` (unittest, NOT pytest).
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.csrf import CSRF_COOKIE_NAME, CSRF_HEADER_NAME
from app.db import get_session
from app.main import app
from app.models import Game, GameStatus, Pick, PickType, Team, User, Week
from app.services import admin as admin_service
from app.services.auth import create_session_cookie, hash_password

SEASON = 2025
WEEK = 1


def _enable_sqlite_fks(dbapi_connection, _connection_record):  # noqa: ANN001
    """Connect listener: turn SQLite FK (and cascade) enforcement ON."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class AdminApiTests(unittest.TestCase):
    """Offline TestClient coverage for the /api/admin user-management router."""

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        # Register the FK-enforcement listener BEFORE create_all so the cascade is
        # really enforced (and the schema-building connection is covered too).
        event.listen(self.engine, "connect", _enable_sqlite_fks)
        SQLModel.metadata.create_all(self.engine)

        pw = hash_password("correct horse battery staple")
        with Session(self.engine) as session:
            # Two admins (so revoke/delete happy paths do NOT trip last-admin) and
            # two plain members. Each carries a distinct discord_id: the
            # one-null-discord_id invariant (260629-n59) caps NULL discord_ids at
            # one, so these fixtures must NOT all leave discord_id null.
            admin = User(
                display_name="admin", password_hash=pw, is_admin=True,
                is_active=True, discord_id=1001,
            )
            admin2 = User(
                display_name="admin2", password_hash=pw, is_admin=True,
                is_active=True, discord_id=1002,
            )
            # member carries a Discord identity (avatar hash present) so the
            # list response exposes a non-null discord_avatar_hash.
            member = User(
                display_name="member",
                password_hash=pw,
                is_admin=False,
                is_active=True,
                discord_id=7777,
                discord_avatar_hash="memberavatarhash",
            )
            member2 = User(
                display_name="member2", password_hash=pw, is_admin=False,
                is_active=True, discord_id=1003,
            )
            session.add_all([admin, admin2, member, member2])
            session.commit()
            for u in (admin, admin2, member, member2):
                session.refresh(u)
            self.admin_id = admin.id
            self.admin2_id = admin2.id
            self.member_id = member.id
            self.member2_id = member2.id

            # Teams + week + game so picks (pick_count + cascade) can be seeded.
            team_home = Team(espn_team_id=1, abbreviation="HOM", display_name="Home")
            team_away = Team(espn_team_id=2, abbreviation="AWY", display_name="Away")
            session.add_all([team_home, team_away])
            session.commit()
            session.refresh(team_home)
            session.refresh(team_away)

            week = Week(season=SEASON, week=WEEK)
            session.add(week)
            session.commit()
            session.refresh(week)
            self.week_id = week.id

            game = Game(
                espn_event_id=1001,
                week_id=week.id,
                season=SEASON,
                week=WEEK,
                home_team_id=team_home.id,
                away_team_id=team_away.id,
                kickoff_at=datetime.now(timezone.utc),
                status=GameStatus.SCHEDULED,
            )
            session.add(game)
            session.commit()
            session.refresh(game)
            self.game_id = game.id

            # member -> 2 picks (pick_count==2); member2 -> 0 (pick_count==0).
            session.add_all(
                [
                    Pick(user_id=member.id, game_id=game.id, week_id=week.id,
                         pick_type=PickType.UNDERDOG_COVER),
                    Pick(user_id=member.id, game_id=game.id, week_id=week.id,
                         pick_type=PickType.OVER),
                ]
            )
            session.commit()

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

    def _get_user(self, user_id: int) -> User | None:
        with self._session() as session:
            return session.get(User, user_id)

    @staticmethod
    def _assert_envelope(body: dict) -> dict:
        assert "error" in body, f"expected an error envelope, got: {body}"
        err = body["error"]
        assert "code" in err, f"envelope missing 'code': {err}"
        return err

    def _assert_admin_user_body(self, body: dict, *, user_id: int) -> None:
        self.assertEqual(body["id"], user_id, body)
        self.assertIn("pick_count", body)
        self.assertNotIn("password_hash", body)

    # -- GET /users --------------------------------------------------------

    def test_list_users_returns_all_with_pick_count(self) -> None:
        """GET /users as admin -> 200, every user, correct pick_count, no hash."""
        resp = self.client.get(
            "/api/admin/users", headers=self._bearer_headers(self.admin_id)
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        users = resp.json()["users"]
        by_id = {u["id"]: u for u in users}
        self.assertEqual(
            set(by_id),
            {self.admin_id, self.admin2_id, self.member_id, self.member2_id},
        )
        self.assertEqual(by_id[self.member_id]["pick_count"], 2)
        self.assertEqual(by_id[self.member2_id]["pick_count"], 0)
        for u in users:
            self.assertNotIn("password_hash", u)
        for field in (
            "id",
            "display_name",
            "discord_id",
            "discord_avatar_hash",
            "is_admin",
            "is_active",
            "created_at",
        ):
            self.assertIn(field, by_id[self.admin_id])
        # member has a seeded avatar hash; the others (no Discord identity) null.
        self.assertEqual(
            by_id[self.member_id]["discord_avatar_hash"], "memberavatarhash"
        )
        self.assertIsNone(by_id[self.admin_id]["discord_avatar_hash"])
        self.assertIsNone(by_id[self.member2_id]["discord_avatar_hash"])

    # -- action happy paths (pinned 200 AdminUserRead / 204 contract) ------

    def test_deactivate_happy_path(self) -> None:
        resp = self.client.post(
            f"/api/admin/users/{self.member_id}/deactivate",
            headers=self._cookie_auth_headers(self.admin_id),
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self._assert_admin_user_body(body, user_id=self.member_id)
        self.assertFalse(body["is_active"])
        self.assertFalse(self._get_user(self.member_id).is_active)

    def test_reactivate_happy_path(self) -> None:
        with self._session() as session:
            u = session.get(User, self.member_id)
            u.is_active = False
            session.add(u)
            session.commit()
        resp = self.client.post(
            f"/api/admin/users/{self.member_id}/reactivate",
            headers=self._cookie_auth_headers(self.admin_id),
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self._assert_admin_user_body(body, user_id=self.member_id)
        self.assertTrue(body["is_active"])
        self.assertTrue(self._get_user(self.member_id).is_active)

    def test_grant_admin_happy_path(self) -> None:
        resp = self.client.post(
            f"/api/admin/users/{self.member_id}/grant-admin",
            headers=self._cookie_auth_headers(self.admin_id),
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self._assert_admin_user_body(body, user_id=self.member_id)
        self.assertTrue(body["is_admin"])
        self.assertTrue(self._get_user(self.member_id).is_admin)

    def test_revoke_admin_happy_path(self) -> None:
        # admin2 is a second admin so the last-admin guard does NOT fire.
        resp = self.client.post(
            f"/api/admin/users/{self.admin2_id}/revoke-admin",
            headers=self._cookie_auth_headers(self.admin_id),
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self._assert_admin_user_body(body, user_id=self.admin2_id)
        self.assertFalse(body["is_admin"])
        self.assertFalse(self._get_user(self.admin2_id).is_admin)

    def test_delete_happy_path(self) -> None:
        resp = self.client.delete(
            f"/api/admin/users/{self.member2_id}",
            headers=self._cookie_auth_headers(self.admin_id),
        )
        self.assertEqual(resp.status_code, 204, resp.text)
        self.assertIn(resp.content, (b"", b"null"))
        self.assertIsNone(self._get_user(self.member2_id))

    # -- self-guards -------------------------------------------------------

    def test_self_deactivate_rejected(self) -> None:
        resp = self.client.post(
            f"/api/admin/users/{self.admin_id}/deactivate",
            headers=self._cookie_auth_headers(self.admin_id),
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(self._assert_envelope(resp.json()).get("reason"), "cannot_act_on_self")
        self.assertTrue(self._get_user(self.admin_id).is_active)

    def test_self_revoke_admin_rejected(self) -> None:
        resp = self.client.post(
            f"/api/admin/users/{self.admin_id}/revoke-admin",
            headers=self._cookie_auth_headers(self.admin_id),
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(self._assert_envelope(resp.json()).get("reason"), "cannot_act_on_self")
        self.assertTrue(self._get_user(self.admin_id).is_admin)

    def test_self_delete_rejected(self) -> None:
        resp = self.client.delete(
            f"/api/admin/users/{self.admin_id}",
            headers=self._cookie_auth_headers(self.admin_id),
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(self._assert_envelope(resp.json()).get("reason"), "cannot_act_on_self")
        self.assertIsNotNone(self._get_user(self.admin_id))

    # -- missing user 404 --------------------------------------------------

    def test_action_on_missing_user_404(self) -> None:
        missing = 999999
        resp = self.client.post(
            f"/api/admin/users/{missing}/reactivate",
            headers=self._cookie_auth_headers(self.admin_id),
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(self._assert_envelope(resp.json()).get("reason"), "user_not_found")

    # -- 403 / 401 ---------------------------------------------------------

    def test_non_admin_forbidden_403(self) -> None:
        get = self.client.get(
            "/api/admin/users", headers=self._bearer_headers(self.member_id)
        )
        self.assertEqual(get.status_code, 403, get.text)
        self._assert_envelope(get.json())

        action = self.client.post(
            f"/api/admin/users/{self.member2_id}/deactivate",
            headers=self._cookie_auth_headers(self.member_id),
        )
        self.assertEqual(action.status_code, 403, action.text)
        self._assert_envelope(action.json())

    def test_unauthenticated_401(self) -> None:
        self._clear_auth()
        get = self.client.get("/api/admin/users")
        self.assertEqual(get.status_code, 401, get.text)
        self._assert_envelope(get.json())

        self._clear_auth()
        action = self.client.delete(f"/api/admin/users/{self.member_id}")
        self.assertEqual(action.status_code, 401, action.text)
        self._assert_envelope(action.json())

    # -- delete cascade (ties QT-A) ---------------------------------------

    def test_delete_cascades_user_picks(self) -> None:
        """Deleting member (who has 2 picks) -> 204 and the picks are gone."""
        with self._session() as session:
            before = session.exec(
                select(Pick).where(Pick.user_id == self.member_id)
            ).all()
            self.assertEqual(len(before), 2, "fixture should give member 2 picks")

        resp = self.client.delete(
            f"/api/admin/users/{self.member_id}",
            headers=self._cookie_auth_headers(self.admin_id),
        )
        self.assertEqual(resp.status_code, 204, resp.text)

        with self._session() as session:
            self.assertIsNone(session.get(User, self.member_id))
            orphans = session.exec(
                select(Pick).where(Pick.user_id == self.member_id)
            ).all()
            self.assertEqual(orphans, [], "member's picks should be cascade-deleted")


class AdminLastAdminGuardTests(unittest.TestCase):
    """Service-layer coverage for the three last-admin guards + their counting rules.

    The last-admin branch is only reachable with a DISTINCT caller against a
    count==1 (revoke/delete) or active-count==1 (deactivate) target — a situation
    the HTTP layer cannot produce because a distinct admin caller raises the count
    above 1 (and a self caller is pre-empted by the self-guard; see the HTTP class).
    Calling the service directly with a synthetic ``caller_id`` exercises the guard
    itself and pins the DISTINCT counting rules.
    """

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)
        self.pw = hash_password("correct horse battery staple")
        # Monotonic source of distinct non-null discord_ids: the one-null
        # invariant (260629-n59) caps NULL discord_ids at one, so every fixture
        # user gets a distinct id unless the caller passes one explicitly.
        self._next_discord_id = 9000

    def tearDown(self) -> None:
        self.engine.dispose()

    def _add(self, **kwargs) -> int:
        if "discord_id" not in kwargs:
            self._next_discord_id += 1
            kwargs["discord_id"] = self._next_discord_id
        with Session(self.engine) as session:
            u = User(password_hash=self.pw, **kwargs)
            session.add(u)
            session.commit()
            session.refresh(u)
            return u.id

    # Use a synthetic caller_id that is guaranteed distinct from any target so the
    # self-guard never fires and the last-admin branch is genuinely exercised.
    CALLER = 10_000_000

    def test_revoke_last_admin_guard(self) -> None:
        """revoke-admin on the sole is_admin user -> ValueError(last_admin)."""
        sole_admin = self._add(display_name="sole", is_admin=True, is_active=True)
        # A non-admin also exists — it must NOT count toward the admin total.
        self._add(display_name="plain", is_admin=False, is_active=True)
        with Session(self.engine) as session:
            with self.assertRaises(ValueError) as ctx:
                admin_service.revoke_admin(session, caller_id=self.CALLER, user_id=sole_admin)
            self.assertTrue(str(ctx.exception).startswith("last_admin"))
            self.assertTrue(session.get(User, sole_admin).is_admin)

    def test_delete_last_admin_guard(self) -> None:
        """delete on the sole is_admin user -> ValueError(last_admin); row intact."""
        sole_admin = self._add(display_name="sole", is_admin=True, is_active=True)
        self._add(display_name="plain", is_admin=False, is_active=True)
        with Session(self.engine) as session:
            with self.assertRaises(ValueError) as ctx:
                admin_service.delete_user(session, caller_id=self.CALLER, user_id=sole_admin)
            self.assertTrue(str(ctx.exception).startswith("last_admin"))
            self.assertIsNotNone(session.get(User, sole_admin))

    def test_deactivate_last_active_admin_guard(self) -> None:
        """deactivate the sole ACTIVE admin -> ValueError(last_admin).

        DISTINCT counting rule: an INACTIVE admin also exists, but deactivate
        counts ACTIVE admins only, so the lone active admin is the last one.
        """
        active_admin = self._add(display_name="active", is_admin=True, is_active=True)
        # An INACTIVE admin: present, is_admin=True, but is_active=False — must NOT
        # count toward the active-admin total (proves the active-only rule).
        self._add(display_name="inactive", is_admin=True, is_active=False)
        with Session(self.engine) as session:
            with self.assertRaises(ValueError) as ctx:
                admin_service.deactivate_user(session, caller_id=self.CALLER, user_id=active_admin)
            self.assertTrue(str(ctx.exception).startswith("last_admin"))
            self.assertTrue(session.get(User, active_admin).is_active)

    def test_deactivate_admin_allowed_when_another_active_admin_exists(self) -> None:
        """Deactivating an active admin is OK while a SECOND active admin remains.

        Confirms the active-admin guard fires on count==1, not always — the
        deactivate succeeds here because two active admins exist.
        """
        a1 = self._add(display_name="a1", is_admin=True, is_active=True)
        self._add(display_name="a2", is_admin=True, is_active=True)
        with Session(self.engine) as session:
            row = admin_service.deactivate_user(session, caller_id=self.CALLER, user_id=a1)
            self.assertFalse(row.is_active)
            self.assertFalse(session.get(User, a1).is_active)


if __name__ == "__main__":
    unittest.main()
