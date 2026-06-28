"""Offline tests for the read-only calendar (date-range games) endpoint.

Covers the thin authenticated HTTP router (:mod:`app.api.calendar`) via an
in-memory ``TestClient`` — mirroring the conventions of
:mod:`tests.test_slate_api`:

* a single shared in-memory SQLite connection (``StaticPool``) so every
  ``Session`` — including the one ``get_current_user`` opens — sees the SAME db,
* ``app.dependency_overrides[get_session]`` routed at that engine so importing
  :mod:`app.main` never opens a real Postgres connection,
* no network of any kind,
* bearer auth for reads (CSRF-exempt) via ``_bearer_headers``; ``_clear_auth``
  for the unauthenticated 401 case; ``_assert_envelope`` for the error shape.

The date-range cases seed games with EXPLICIT fixed UTC instants (NOT ``now ±
delta``) so they are clock-independent — the range filter is absolute. The
chosen window is well away from any boundary except the deliberate inclusive-
upper-bound case.

> Note: on this machine there is no bare ``python`` on ``PATH``; run with
> ``backend/.venv/bin/python -m unittest tests.test_calendar_api``.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.db import get_session
from app.main import app
from app.models import Game, GameStatus, Team, User, Week
from app.services.auth import create_session_cookie, hash_password

SEASON = 2025


def _aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from SQLite."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class CalendarTests(unittest.TestCase):
    """HTTP coverage for the read-only calendar surface."""

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

    def _seed_week_row(self, week: int) -> int:
        with self._session() as session:
            week_row = Week(season=SEASON, week=week)
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
        status: GameStatus = GameStatus.SCHEDULED,
        home_score: int | None = None,
        away_score: int | None = None,
    ) -> int:
        """Add a single game with an explicit kickoff; returns its id."""
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
                home_score=home_score,
                away_score=away_score,
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

    def _get(self, from_date: str, to_date: str) -> object:
        return self.client.get(
            f"/api/calendar?from={from_date}&to={to_date}",
            headers=self._bearer_headers(self.user_id),
        )

    # -- case 1: range filter + display fields + ordering ------------------

    def test_range_filter_display_fields_and_ordering(self) -> None:
        """Only in-window games return, carrying matchup abbrs + raw kickoff +
        status, sorted by kickoff."""
        wk = self._seed_week_row(1)
        # One BEFORE the window.
        self._add_game(
            week_id=wk, week=1, espn_event_id=1,
            kickoff_at=datetime(2026, 8, 31, 18, 0, tzinfo=timezone.utc),
            home_team_id=self.tid[0], away_team_id=self.tid[1],
        )
        # Two INSIDE the window, seeded OUT of kickoff order to prove the sort.
        self._add_game(
            week_id=wk, week=1, espn_event_id=2,
            kickoff_at=datetime(2026, 9, 12, 20, 0, tzinfo=timezone.utc),
            home_team_id=self.tid[2], away_team_id=self.tid[3],
        )
        self._add_game(
            week_id=wk, week=1, espn_event_id=3,
            kickoff_at=datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc),
            home_team_id=self.tid[0], away_team_id=self.tid[2],
        )
        # One AFTER the window.
        self._add_game(
            week_id=wk, week=1, espn_event_id=4,
            kickoff_at=datetime(2026, 10, 1, 18, 0, tzinfo=timezone.utc),
            home_team_id=self.tid[1], away_team_id=self.tid[3],
        )

        resp = self._get("2026-09-01", "2026-09-30")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["from_date"], "2026-09-01")
        self.assertEqual(body["to_date"], "2026-09-30")
        games = body["games"]
        self.assertEqual(len(games), 2)

        # Sorted by kickoff: the Sep-10 game (espn 3) comes first.
        first, second = games
        self.assertEqual(
            _aware(datetime.fromisoformat(first["kickoff_at"])),
            datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(first["home_team"]["abbreviation"], "T1")  # tid[0]
        self.assertEqual(first["away_team"]["abbreviation"], "T3")  # tid[2]
        self.assertEqual(first["status"], "SCHEDULED")
        self.assertEqual(
            _aware(datetime.fromisoformat(second["kickoff_at"])),
            datetime(2026, 9, 12, 20, 0, tzinfo=timezone.utc),
        )

    # -- case 2: FINAL score surfaced via the enum -------------------------

    def test_final_score_surfaced_nonfinal_null(self) -> None:
        """A FINAL game carries its score (proving the enum, not a raw string);
        a non-final game's scores are null."""
        wk = self._seed_week_row(1)
        self._add_game(
            week_id=wk, week=1, espn_event_id=10,
            kickoff_at=datetime(2026, 9, 5, 18, 0, tzinfo=timezone.utc),
            home_team_id=self.tid[0], away_team_id=self.tid[1],
            status=GameStatus.FINAL, home_score=33, away_score=8,
        )
        self._add_game(
            week_id=wk, week=1, espn_event_id=11,
            kickoff_at=datetime(2026, 9, 6, 18, 0, tzinfo=timezone.utc),
            home_team_id=self.tid[2], away_team_id=self.tid[3],
            status=GameStatus.SCHEDULED,
        )

        resp = self._get("2026-09-01", "2026-09-30")
        self.assertEqual(resp.status_code, 200, resp.text)
        games = resp.json()["games"]
        by_status = {g["status"]: g for g in games}

        final = by_status["FINAL"]
        self.assertEqual(final["home_score"], 33)
        self.assertEqual(final["away_score"], 8)

        sched = by_status["SCHEDULED"]
        self.assertIsNone(sched["home_score"])
        self.assertIsNone(sched["away_score"])

    # -- case 3: inclusive upper bound -------------------------------------

    def test_to_day_is_inclusive(self) -> None:
        """A game late on the `to` day itself IS included (end = to + 1 day)."""
        wk = self._seed_week_row(1)
        self._add_game(
            week_id=wk, week=1, espn_event_id=20,
            kickoff_at=datetime(2026, 9, 30, 23, 0, tzinfo=timezone.utc),
            home_team_id=self.tid[0], away_team_id=self.tid[1],
        )

        resp = self._get("2026-09-01", "2026-09-30")
        self.assertEqual(resp.status_code, 200, resp.text)
        games = resp.json()["games"]
        self.assertEqual(len(games), 1)
        self.assertEqual(
            _aware(datetime.fromisoformat(games[0]["kickoff_at"])),
            datetime(2026, 9, 30, 23, 0, tzinfo=timezone.utc),
        )

    # -- case 4: empty window ----------------------------------------------

    def test_empty_window_returns_empty_games(self) -> None:
        """A window with no games -> 200 + games == [] (a pure read, never 404)."""
        wk = self._seed_week_row(1)
        self._add_game(
            week_id=wk, week=1, espn_event_id=30,
            kickoff_at=datetime(2026, 9, 5, 18, 0, tzinfo=timezone.utc),
            home_team_id=self.tid[0], away_team_id=self.tid[1],
        )

        resp = self._get("2026-11-01", "2026-11-30")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["games"], [])

    # -- case 5: unauthenticated -------------------------------------------

    def test_unauthenticated_rejected_401(self) -> None:
        """GET /api/calendar with no auth -> 401 + error envelope."""
        self._clear_auth()
        resp = self.client.get("/api/calendar?from=2026-09-01&to=2026-09-30")
        self.assertEqual(resp.status_code, 401, resp.text)
        self._assert_envelope(resp.json())

    # -- case 6: malformed date -> clean 4xx -------------------------------

    def test_bad_date_string_returns_422(self) -> None:
        """A malformed `from` date -> a clean 422, never a 500."""
        resp = self._get("not-a-date", "2026-09-30")
        self.assertEqual(resp.status_code, 422, resp.text)
        self._assert_envelope(resp.json())


if __name__ == "__main__":
    unittest.main()
