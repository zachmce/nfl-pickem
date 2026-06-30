"""Offline tests for the admin ingest/freeze TRIGGER routes (QT-2).

Two mutating POST routes under the already-gated /api/admin router:

* ``POST /api/admin/ingest-season`` {season} -> dispatches ``ingest_season_task``
* ``POST /api/admin/freeze-week`` {season, week} -> dispatches ``freeze_week_task``

Both are require_admin-gated and DISPATCH a Celery task (return 202 with the
AsyncResult id) rather than running blocking ESPN fetches in the request thread.

Fully OFFLINE (mirrors :mod:`tests.test_admin_api` setUp): ``StaticPool``
in-memory SQLite, ``dependency_overrides[get_session]``, cookie+CSRF for the
mutating POSTs, a bearer helper, one admin + one non-admin member. There is NO
real broker — the dispatch is exercised by MONKEYPATCHING the task objects on the
``app.api.admin`` module with fakes whose ``.delay(...)`` records the call args and
returns a ``SimpleNamespace(id=...)``. Restored in tearDown.

> Run from backend/ with ``.venv/bin/python -m unittest`` (unittest, NOT pytest).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.api import admin as admin_module
from app.csrf import CSRF_COOKIE_NAME, CSRF_HEADER_NAME
from app.db import get_session
from app.main import app
from app.models import User
from app.services.auth import create_session_cookie, hash_password


class _FakeTask:
    """Records the args of each ``.delay(...)`` and returns a fake AsyncResult."""

    def __init__(self, task_id: str = "fake-task-id") -> None:
        self.task_id = task_id
        self.calls: list[tuple] = []

    def delay(self, *args):  # noqa: ANN002
        self.calls.append(args)
        return SimpleNamespace(id=self.task_id)


class AdminTriggersApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        pw = hash_password("correct horse battery staple")
        with Session(self.engine) as session:
            # Distinct discord_ids: the one-null-discord_id invariant (260629-n59)
            # caps NULL discord_ids at one.
            admin = User(
                display_name="admin",
                password_hash=pw,
                is_admin=True,
                is_active=True,
                discord_id=1,
            )
            member = User(
                display_name="member",
                password_hash=pw,
                is_admin=False,
                is_active=True,
                discord_id=2,
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

        # Monkeypatch the dispatched task objects on the admin module with fakes
        # (no real broker). Saved so tearDown can restore them.
        self._real_ingest = admin_module.ingest_season_task
        self._real_freeze = admin_module.freeze_week_task
        self.fake_ingest = _FakeTask("ingest-task-id")
        self.fake_freeze = _FakeTask("freeze-task-id")
        admin_module.ingest_season_task = self.fake_ingest
        admin_module.freeze_week_task = self.fake_freeze

    def tearDown(self) -> None:
        admin_module.ingest_season_task = self._real_ingest
        admin_module.freeze_week_task = self._real_freeze
        app.dependency_overrides.pop(get_session, None)
        self.client.close()
        self.engine.dispose()

    # -- helpers -----------------------------------------------------------

    def _cookie_auth_headers(self, user_id: int) -> dict[str, str]:
        csrf_value = "test-csrf-token-value"
        self.client.cookies.set("session", create_session_cookie(user_id))
        self.client.cookies.set(CSRF_COOKIE_NAME, csrf_value)
        return {CSRF_HEADER_NAME: csrf_value}

    def _clear_auth(self) -> None:
        self.client.cookies.clear()

    @staticmethod
    def _assert_envelope(body: dict) -> dict:
        assert "error" in body, f"expected an error envelope, got: {body}"
        err = body["error"]
        assert "code" in err, f"envelope missing 'code': {err}"
        return err

    # -- ingest-season: admin happy path -----------------------------------

    def test_ingest_season_admin_dispatches_and_returns_202(self) -> None:
        resp = self.client.post(
            "/api/admin/ingest-season",
            json={"season": 2026},
            headers=self._cookie_auth_headers(self.admin_id),
        )
        self.assertEqual(resp.status_code, 202, resp.text)
        body = resp.json()
        self.assertEqual(body["task_id"], "ingest-task-id")
        self.assertEqual(body["season"], 2026)
        self.assertEqual(self.fake_ingest.calls, [(2026,)])
        self.assertEqual(self.fake_freeze.calls, [])

    # -- freeze-week: admin happy path -------------------------------------

    def test_freeze_week_admin_dispatches_and_returns_202(self) -> None:
        resp = self.client.post(
            "/api/admin/freeze-week",
            json={"season": 2026, "week": 3},
            headers=self._cookie_auth_headers(self.admin_id),
        )
        self.assertEqual(resp.status_code, 202, resp.text)
        body = resp.json()
        self.assertEqual(body["task_id"], "freeze-task-id")
        self.assertEqual(body["season"], 2026)
        self.assertEqual(body["week"], 3)
        self.assertEqual(self.fake_freeze.calls, [(2026, 3)])
        self.assertEqual(self.fake_ingest.calls, [])

    # -- non-admin -> 403, no dispatch -------------------------------------

    def test_ingest_season_non_admin_forbidden(self) -> None:
        resp = self.client.post(
            "/api/admin/ingest-season",
            json={"season": 2026},
            headers=self._cookie_auth_headers(self.member_id),
        )
        self.assertEqual(resp.status_code, 403, resp.text)
        self._assert_envelope(resp.json())
        self.assertEqual(self.fake_ingest.calls, [])

    def test_freeze_week_non_admin_forbidden(self) -> None:
        resp = self.client.post(
            "/api/admin/freeze-week",
            json={"season": 2026, "week": 3},
            headers=self._cookie_auth_headers(self.member_id),
        )
        self.assertEqual(resp.status_code, 403, resp.text)
        self._assert_envelope(resp.json())
        self.assertEqual(self.fake_freeze.calls, [])

    # -- anonymous -> 401, no dispatch -------------------------------------

    def test_ingest_season_anonymous_unauthorized(self) -> None:
        self._clear_auth()
        resp = self.client.post("/api/admin/ingest-season", json={"season": 2026})
        self.assertEqual(resp.status_code, 401, resp.text)
        self._assert_envelope(resp.json())
        self.assertEqual(self.fake_ingest.calls, [])

    def test_freeze_week_anonymous_unauthorized(self) -> None:
        self._clear_auth()
        resp = self.client.post("/api/admin/freeze-week", json={"season": 2026, "week": 3})
        self.assertEqual(resp.status_code, 401, resp.text)
        self._assert_envelope(resp.json())
        self.assertEqual(self.fake_freeze.calls, [])


if __name__ == "__main__":
    unittest.main()
