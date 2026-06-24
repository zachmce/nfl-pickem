"""Offline tests for the unauthenticated client-config read surface.

Covers the thin GET-only HTTP router (:mod:`app.api.config`) via an in-memory
``TestClient`` — mirroring the conventions established by
:mod:`tests.test_results_api`:

* a single shared in-memory SQLite connection (``StaticPool``) so every
  ``Session`` sees the SAME db,
* ``app.dependency_overrides[get_session]`` routed at that engine so importing
  :mod:`app.main` never opens a real Postgres connection,
* no network of any kind.

Unlike /api/results and /api/current-week, /api/config is a deliberately
UNAUTHENTICATED read (the pre-auth signal the bare login page reads). The
load-bearing guarantee here is that an anonymous request returns 200 (NOT 401),
with a fixed {is_demo, season} shape.

> Note: on this machine there is no bare ``python`` on ``PATH``; run with
> ``backend/.venv/bin/python -m unittest``.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings
from app.db import get_session
from app.main import app
from app.models import Game, GameStatus, Team, Week

SEASON = 2025


class ConfigApiTests(unittest.TestCase):
    """HTTP coverage for the unauthenticated /api/config read surface."""

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

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

    def _seed_season(self, season: int) -> None:
        """Seed two teams + a week + one game so a single distinct season exists."""
        now = datetime.now(timezone.utc)
        with Session(self.engine) as session:
            teams = [
                Team(espn_team_id=i, abbreviation=f"T{i}", display_name=f"Team {i}")
                for i in range(1, 3)
            ]
            session.add_all(teams)
            session.commit()
            for t in teams:
                session.refresh(t)
            tid = [t.id for t in teams]

            week = Week(season=season, week=1)
            session.add(week)
            session.commit()
            session.refresh(week)
            assert week.id is not None

            game = Game(
                espn_event_id=1,
                week_id=week.id,
                season=season,
                week=1,
                home_team_id=tid[0],
                away_team_id=tid[1],
                kickoff_at=now,
                status=GameStatus.SCHEDULED,
            )
            session.add(game)
            session.commit()

    # -- tests -------------------------------------------------------------

    def test_no_games_returns_season_zero(self) -> None:
        """With no seeded games the season falls back to 0 (never a 404)."""
        with patch.object(settings, "is_demo_data", False):
            resp = self.client.get("/api/config")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json(), {"is_demo": False, "season": 0})

    def test_one_season_returns_that_season(self) -> None:
        """A single seeded season is reflected in the response."""
        self._seed_season(SEASON)
        with patch.object(settings, "is_demo_data", True):
            resp = self.client.get("/api/config")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json(), {"is_demo": True, "season": SEASON})

    def test_unauthenticated_request_is_200_not_401(self) -> None:
        """The load-bearing guarantee: no auth cookie/header still yields 200.

        /api/config is the pre-auth demo signal — it must NOT 401 like the other
        read endpoints do for anonymous callers.
        """
        self.client.cookies.clear()
        resp = self.client.get("/api/config")
        self.assertNotEqual(resp.status_code, 401, resp.text)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIn("is_demo", body)
        self.assertIn("season", body)
        # Exact shape (extra="forbid" on the schema means only these two keys).
        self.assertEqual(set(body.keys()), {"is_demo", "season"})


if __name__ == "__main__":
    unittest.main()
