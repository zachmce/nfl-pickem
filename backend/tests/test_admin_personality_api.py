"""Offline HTTP tests for the admin bot-personality routes (260627-xbb).

Fully OFFLINE (in-memory SQLite ``StaticPool`` + ``dependency_overrides`` +
``TestClient``), mirroring :mod:`tests.test_admin_api`. They pin:

* GET /api/admin/bot-personality -> 200 with the active id (the sarcastic default
  when unset) + the registry's available ids;
* POST sets the active id and the change is reflected on the next GET;
* POST with an unknown id -> 409 (reason ``unknown_personality``);
* anon -> 401, non-admin -> 403 on both verbs (require_admin gating, T-xbb-01).

Run from backend/ with ``.venv/bin/python -m unittest`` (unittest, NOT pytest).
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.bot.personality import DEFAULT_PERSONALITY_ID, available_personality_ids
from app.csrf import CSRF_COOKIE_NAME, CSRF_HEADER_NAME
from app.db import get_session
from app.main import app
from app.models import User
from app.services.auth import create_session_cookie, hash_password


class AdminPersonalityApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        pw = hash_password("correct horse battery staple")
        with Session(self.engine) as session:
            admin = User(
                display_name="admin", password_hash=pw, is_admin=True, is_active=True
            )
            member = User(
                display_name="member", password_hash=pw, is_admin=False, is_active=True
            )
            session.add_all([admin, member])
            session.commit()
            session.refresh(admin)
            session.refresh(member)
            self.admin_id = admin.id
            self.member_id = member.id

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

    def _cookie_auth_headers(self, user_id: int) -> dict[str, str]:
        csrf_value = "test-csrf-token-value"
        self.client.cookies.set("session", create_session_cookie(user_id))
        self.client.cookies.set(CSRF_COOKIE_NAME, csrf_value)
        return {CSRF_HEADER_NAME: csrf_value}

    def _bearer_headers(self, user_id: int) -> dict[str, str]:
        return {"Authorization": f"Bearer {create_session_cookie(user_id)}"}

    def _clear_auth(self) -> None:
        self.client.cookies.clear()

    @staticmethod
    def _assert_envelope(body: dict) -> dict:
        assert "error" in body, f"expected an error envelope, got: {body}"
        return body["error"]

    def _non_default_id(self) -> str:
        return next(
            pid for pid in available_personality_ids() if pid != DEFAULT_PERSONALITY_ID
        )

    # -- GET ---------------------------------------------------------------

    def test_get_returns_default_active_and_available(self) -> None:
        resp = self.client.get(
            "/api/admin/bot-personality", headers=self._bearer_headers(self.admin_id)
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["active_id"], DEFAULT_PERSONALITY_ID)
        self.assertEqual(body["available_ids"], available_personality_ids())

    # -- POST happy path ---------------------------------------------------

    def test_post_sets_active_and_is_reflected_on_get(self) -> None:
        target = self._non_default_id()
        resp = self.client.post(
            "/api/admin/bot-personality",
            headers=self._cookie_auth_headers(self.admin_id),
            json={"personality_id": target},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["active_id"], target)

        # The next GET reflects the new active id.
        got = self.client.get(
            "/api/admin/bot-personality", headers=self._bearer_headers(self.admin_id)
        )
        self.assertEqual(got.json()["active_id"], target)

    # -- POST unknown id -> 409 -------------------------------------------

    def test_post_unknown_id_409(self) -> None:
        resp = self.client.post(
            "/api/admin/bot-personality",
            headers=self._cookie_auth_headers(self.admin_id),
            json={"personality_id": "made_up_voice"},
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(
            self._assert_envelope(resp.json()).get("reason"), "unknown_personality"
        )

    # -- 401 / 403 ---------------------------------------------------------

    def test_anonymous_401(self) -> None:
        self._clear_auth()
        get = self.client.get("/api/admin/bot-personality")
        self.assertEqual(get.status_code, 401, get.text)
        self._assert_envelope(get.json())

        self._clear_auth()
        post = self.client.post(
            "/api/admin/bot-personality", json={"personality_id": "sarcastic"}
        )
        self.assertEqual(post.status_code, 401, post.text)
        self._assert_envelope(post.json())

    def test_non_admin_403(self) -> None:
        get = self.client.get(
            "/api/admin/bot-personality", headers=self._bearer_headers(self.member_id)
        )
        self.assertEqual(get.status_code, 403, get.text)
        self._assert_envelope(get.json())

        post = self.client.post(
            "/api/admin/bot-personality",
            headers=self._cookie_auth_headers(self.member_id),
            json={"personality_id": "sarcastic"},
        )
        self.assertEqual(post.status_code, 403, post.text)
        self._assert_envelope(post.json())


if __name__ == "__main__":
    unittest.main()
