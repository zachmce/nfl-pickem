"""Offline tests for the current-week context-bar read endpoint.

Covers the thin authenticated HTTP router (:mod:`app.api.current_week`) via an
in-memory ``TestClient`` — mirroring the conventions established by
:mod:`tests.test_results_api`:

* a single shared in-memory SQLite connection (``StaticPool``) so every
  ``Session`` — including the one ``get_current_user`` opens — sees the SAME db,
* ``app.dependency_overrides[get_session]`` routed at that engine so importing
  :mod:`app.main` never opens a real Postgres connection,
* no network of any kind,
* bearer auth for reads (CSRF-exempt) via ``_bearer_headers``; ``_clear_auth``
  for the unauthenticated 401 case; ``_assert_envelope`` for the error shape.

Each window state is driven by seeding ``Game.kickoff_at`` relative to the real
clock (future = now + days, past = now - days) plus explicit ``GameStatus``, so
the four states + the earliest-open selection + the all-closed fallback + the
demo-shift case are deterministic and not clock-flaky. No picks are seeded — this
endpoint does not read picks.

> Note: on this machine there is no bare ``python`` on ``PATH``; run with
> ``backend/.venv/bin/python -m unittest``.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import httpx
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


class CurrentWeekTests(unittest.TestCase):
    """HTTP coverage for the current-week read surface."""

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

    def _seed_week(
        self,
        *,
        week: int,
        kickoffs: list[datetime],
        status: GameStatus = GameStatus.SCHEDULED,
        season: int = SEASON,
        lines_frozen: bool = False,
    ) -> None:
        """Seed a Week row + its games with the given kickoffs and status.

        Home/away teams cycle through the four seeded teams; ``espn_event_id`` is
        kept unique across weeks AND seasons. Games default to SCHEDULED; pass
        FINAL/IN_PROGRESS to drive the locked/closed split. ``season`` defaults to
        ``SEASON`` (2025) so existing single-season callers are unchanged; pass a
        different year to seed a second season for the multi-season case.
        ``lines_frozen`` sets the Week's admin-override freeze column (defaults
        False so existing callers are unchanged; mirrors slate's
        ``_seed_week_row``).
        """
        with self._session() as session:
            week_row = Week(season=season, week=week, lines_frozen=lines_frozen)
            session.add(week_row)
            session.commit()
            session.refresh(week_row)
            assert week_row.id is not None

            for i, ko in enumerate(kickoffs):
                session.add(
                    Game(
                        espn_event_id=season * 100000 + week * 1000 + i,
                        week_id=week_row.id,
                        season=season,
                        week=week,
                        home_team_id=self.tid[i % 2 * 2],
                        away_team_id=self.tid[i % 2 * 2 + 1],
                        kickoff_at=ko,
                        status=status,
                    )
                )
            session.commit()

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

    def _get(self) -> httpx.Response:
        return self.client.get("/api/current-week", headers=self._bearer_headers(self.user_id))

    # -- auth --------------------------------------------------------------

    def test_unauthenticated_rejected_401(self) -> None:
        """GET /api/current-week with no auth -> 401 + error envelope."""
        self._clear_auth()
        resp = self.client.get("/api/current-week")
        self.assertEqual(resp.status_code, 401, resp.text)
        self._assert_envelope(resp.json())

    # -- states ------------------------------------------------------------

    def test_state_open(self) -> None:
        """Week 1 (open_at None) with a FUTURE first kickoff -> open."""
        now = datetime.now(timezone.utc)
        first = now + timedelta(days=2)
        self._seed_week(week=1, kickoffs=[first, first + timedelta(hours=3)])

        resp = self._get()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["season"], SEASON)
        self.assertEqual(body["week"], 1)
        self.assertEqual(body["window_state"], "open")
        self.assertEqual(_aware(datetime.fromisoformat(body["window_closes_at"])), first)
        # A future SCHEDULED kickoff means the season is NOT over.
        self.assertIs(body["season_complete"], False)

    def test_state_not_yet_open(self) -> None:
        """Chosen week's open boundary (prev week's last kickoff + duration) is
        still in the FUTURE -> not_yet_open.

        Week 1 is fully past+FINAL so it is closed; week 2 is the chosen week.
        Week 1's latest kickoff is in the future (+5 days), so week 2's open_at
        (that + ~3.5h) has not been reached, while week 2's close (its first
        kickoff, +6 days) is further out.
        """
        now = datetime.now(timezone.utc)
        # Week 1: closed (first kickoff in the past) but with a FUTURE latest
        # kickoff so it drives week 2's open boundary into the future. Mark FINAL
        # is irrelevant for selection here — week 1's window is already closed.
        self._seed_week(
            week=1,
            kickoffs=[now - timedelta(days=1), now + timedelta(days=5)],
            status=GameStatus.FINAL,
        )
        # Week 2: first kickoff +6 days (window still future-open -> chosen).
        self._seed_week(
            week=2,
            kickoffs=[now + timedelta(days=6), now + timedelta(days=6, hours=3)],
        )

        resp = self._get()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["week"], 2)
        self.assertEqual(body["window_state"], "not_yet_open")

    def test_state_locked(self) -> None:
        """Chosen week's first kickoff is PAST (closed) but a game is non-FINAL
        -> locked."""
        now = datetime.now(timezone.utc)
        self._seed_week(
            week=1,
            kickoffs=[now - timedelta(hours=2), now + timedelta(hours=1)],
            status=GameStatus.IN_PROGRESS,
        )

        resp = self._get()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["week"], 1)
        self.assertEqual(body["window_state"], "locked")

    def test_state_closed(self) -> None:
        """The only week's window is closed AND every game FINAL -> closed,
        chosen via the all-closed fallback to the latest week."""
        now = datetime.now(timezone.utc)
        self._seed_week(
            week=1,
            kickoffs=[now - timedelta(days=2), now - timedelta(days=2, hours=-3)],
            status=GameStatus.FINAL,
        )

        resp = self._get()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["week"], 1)
        self.assertEqual(body["window_state"], "closed")
        # The single week is past + every game FINAL -> the season is complete.
        self.assertIs(body["season_complete"], True)

    def test_current_week_selection_picks_earliest_open(self) -> None:
        """Week 1 fully closed/past, week 2 still future-open -> week 2 chosen."""
        now = datetime.now(timezone.utc)
        self._seed_week(
            week=1,
            kickoffs=[now - timedelta(days=3), now - timedelta(days=3, hours=-3)],
            status=GameStatus.FINAL,
        )
        wk2_first = now + timedelta(days=2)
        self._seed_week(week=2, kickoffs=[wk2_first, wk2_first + timedelta(hours=3)])

        resp = self._get()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["week"], 2)
        self.assertEqual(body["window_state"], "open")
        self.assertEqual(_aware(datetime.fromisoformat(body["window_closes_at"])), wk2_first)

    def test_in_progress_week_stays_current_not_next(self) -> None:
        """Regression (#37): a week whose games are IN PROGRESS stays current even
        when a later week exists.

        Week 1's first kickoff is in the PAST (pick window closed) but a game is
        still non-FINAL — the week is being played out. A future week 2 exists.
        The current week must remain week 1 (state locked), not advance to the
        upcoming week 2. Before the fix the window-open selector dropped week 1
        (now >= close_at) and jumped to week 2.
        """
        now = datetime.now(timezone.utc)
        # Week 1: first game already kicked off (past), a later game still to come
        # (future), whole week IN_PROGRESS -> not all FINAL.
        self._seed_week(
            week=1,
            kickoffs=[now - timedelta(hours=2), now + timedelta(hours=1)],
            status=GameStatus.IN_PROGRESS,
        )
        # Week 2: entirely in the future (would be the wrongly-chosen "open" week).
        self._seed_week(
            week=2,
            kickoffs=[now + timedelta(days=7), now + timedelta(days=7, hours=3)],
        )

        resp = self._get()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["week"], 1)
        self.assertEqual(body["window_state"], "locked")

    def test_current_week_advances_once_week_all_final(self) -> None:
        """The current week rolls forward only once every game of the earlier week
        is FINAL — the complement of the #37 regression above.

        Week 1 is fully FINAL (played out); week 2 is upcoming. The current week
        must now be week 2.
        """
        now = datetime.now(timezone.utc)
        # Week 1's last kickoff is far enough in the past that week 2's open
        # boundary (that + ~3.5h) has already passed -> week 2 is open, not
        # merely not_yet_open.
        self._seed_week(
            week=1,
            kickoffs=[now - timedelta(hours=8), now - timedelta(hours=5)],
            status=GameStatus.FINAL,
        )
        wk2_first = now + timedelta(days=5)
        self._seed_week(week=2, kickoffs=[wk2_first, wk2_first + timedelta(hours=3)])

        resp = self._get()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["week"], 2)
        self.assertEqual(body["window_state"], "open")

    def test_demo_like_shift(self) -> None:
        """Kickoffs shifted into the FUTURE (simulating the demo time-shift),
        computed against real now -> earliest week reports a future open window.

        Proves no IS_DEMO_DATA branch is needed: the state falls out of
        real-now-vs-persisted-(shifted)-kickoffs.
        """
        now = datetime.now(timezone.utc)
        wk1_first = now + timedelta(days=1)  # season "starts" ~24h out
        self._seed_week(week=1, kickoffs=[wk1_first, wk1_first + timedelta(hours=3)])
        self._seed_week(
            week=2,
            kickoffs=[now + timedelta(days=8), now + timedelta(days=8, hours=3)],
        )

        resp = self._get()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # The earliest (week 1) is chosen and is open (open_at None) with a
        # future close.
        self.assertEqual(body["week"], 1)
        self.assertEqual(body["window_state"], "open")
        closes = _aware(datetime.fromisoformat(body["window_closes_at"]))
        self.assertEqual(closes, wk1_first)
        self.assertGreater(closes, now)

    def test_multi_season_resolves_max_season_no_longer_raises(self) -> None:
        """Regression: a multi-season DB no longer raises NotFoundError.

        Before 260630-h96 the endpoint raised "expected exactly one season ..."
        whenever >1 ``Game.season`` existed. Now it resolves ``max(Game.season)``
        and computes the window over THAT season only. Seed an older season (2024,
        all FINAL/past) plus the newer season (2025) holding a future-open week —
        the response must be 200 with ``season == 2025`` and that week open.
        """
        now = datetime.now(timezone.utc)
        # Older season 2024: a fully past + FINAL week (would otherwise be the
        # "all closed -> latest week" fallback if it leaked into the math).
        self._seed_week(
            week=1,
            kickoffs=[now - timedelta(days=10), now - timedelta(days=10, hours=-3)],
            status=GameStatus.FINAL,
            season=2024,
        )
        # Newer season 2025: week 1 with a FUTURE first kickoff -> open.
        wk1_first = now + timedelta(days=2)
        self._seed_week(
            week=1,
            kickoffs=[wk1_first, wk1_first + timedelta(hours=3)],
            season=2025,
        )

        resp = self._get()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # Resolves the MAX season, not a raise.
        self.assertEqual(body["season"], 2025)
        self.assertEqual(body["week"], 1)
        # Window state + close time reflect the chosen season's week only.
        self.assertEqual(body["window_state"], "open")
        self.assertEqual(_aware(datetime.fromisoformat(body["window_closes_at"])), wk1_first)
        # The 2025 week has a future SCHEDULED kickoff -> season not complete.
        self.assertIs(body["season_complete"], False)

    # -- odds_frozen -------------------------------------------------------

    def test_current_week_reports_odds_frozen_false_before_freeze(self) -> None:
        """A single open week whose first kickoff is FAR in the future (computed
        freeze_at still ahead of real now) -> odds_frozen is False.

        Mirrors slate case 9: the computed predicate branch, not the override.
        """
        now = datetime.now(timezone.utc)
        first = now + timedelta(days=30)  # freeze_at (<= first kickoff) still future
        self._seed_week(week=1, kickoffs=[first, first + timedelta(hours=3)])

        resp = self._get()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["week"], 1)
        self.assertIs(body["odds_frozen"], False)

    def test_current_week_reports_odds_frozen_true_via_override(self) -> None:
        """A single week with lines_frozen=True hits the clock-independent override
        branch of is_odds_frozen -> odds_frozen is True.

        Mirrors slate case 10.
        """
        now = datetime.now(timezone.utc)
        first = now + timedelta(days=2)
        self._seed_week(
            week=1,
            kickoffs=[first, first + timedelta(hours=3)],
            lines_frozen=True,
        )

        resp = self._get()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["week"], 1)
        self.assertIs(body["odds_frozen"], True)


if __name__ == "__main__":
    unittest.main()
