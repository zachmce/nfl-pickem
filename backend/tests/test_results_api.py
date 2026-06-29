"""Offline tests for the results / season-scoreboard read surface.

Covers both the user-agnostic standings service (:mod:`app.services.standings`)
at the function level and the thin authenticated HTTP router
(:mod:`app.api.results`) via an in-memory ``TestClient`` — mirroring the
conventions established by :mod:`tests.test_picks_api`:

* a single shared in-memory SQLite connection (``StaticPool``) so every
  ``Session`` — including the one ``get_current_user`` opens — sees the SAME db,
* ``app.dependency_overrides[get_session]`` routed at that engine so importing
  :mod:`app.main` never opens a real Postgres connection,
* no network of any kind,
* bearer auth for reads (CSRF-exempt) via ``_bearer_headers``; ``_clear_auth``
  for the unauthenticated 401 cases; ``_assert_envelope`` for the error shape.

The scoreboard is a SHARED read: any authenticated user reads ALL users' data
(unlike /api/picks which is strictly self-scoped). FINAL games are seeded with
real scores/odds so ``grade_pick`` produces non-UNGRADEABLE outcomes, and picks
are seeded directly (reads need no open window).

> Note: on this machine there is no bare ``python`` on ``PATH``; run with
> ``backend/.venv/bin/python -m unittest``.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.db import get_session
from app.main import app
from app.models import (
    Game,
    GameStatus,
    Pick,
    PickType,
    Team,
    User,
    Week,
)
from app.services.auth import create_session_cookie, hash_password
from app.services.scoring import GradeOutcome, score_week
from app.services.standings import (
    season_is_complete,
    season_standings,
    week_results,
)

SEASON = 2025
WEEK = 1

# Kickoffs relative to the real clock. The scored week is in the PAST and FINAL
# (reads do not gate on the window), so grade_pick produces real outcomes.
_PAST = timedelta(days=2)


def _aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from SQLite."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class ResultsTests(unittest.TestCase):
    """Service-level + HTTP coverage for the results read surface."""

    week_id: int
    user_a_id: int
    user_b_id: int
    user_c_id: int
    game_fav_id: int
    game_total_id: int

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        now = datetime.now(timezone.utc)
        with Session(self.engine) as session:
            # --- Teams (FK targets) ------------------------------------------
            teams = [
                Team(espn_team_id=i, abbreviation=f"T{i}", display_name=f"Team {i}")
                for i in range(1, 5)
            ]
            session.add_all(teams)
            session.commit()
            for t in teams:
                session.refresh(t)
            tid = [t.id for t in teams]

            # --- Week --------------------------------------------------------
            week = Week(season=SEASON, week=WEEK)
            session.add(week)
            session.commit()
            session.refresh(week)
            assert week.id is not None
            self.week_id = week.id

            # --- FINAL games with real scores + odds ------------------------
            # Favorite (home, tid[0]) is favored by 3.5 and wins by 7 -> the
            # favorite COVERS; underdog does NOT.
            game_fav = Game(
                espn_event_id=1001,
                week_id=week.id,
                season=SEASON,
                week=WEEK,
                home_team_id=tid[0],
                away_team_id=tid[1],
                kickoff_at=now - _PAST,
                status=GameStatus.FINAL,
                home_score=24,
                away_score=17,
                spread=Decimal("3.5"),
                total=Decimal("44.5"),
                favorite_team_id=tid[0],
                underdog_team_id=tid[1],
            )
            # Combined 24+17 = 41 < total 44.5 -> UNDER wins, OVER loses.
            game_total = Game(
                espn_event_id=1002,
                week_id=week.id,
                season=SEASON,
                week=WEEK,
                home_team_id=tid[2],
                away_team_id=tid[3],
                kickoff_at=now - _PAST + timedelta(hours=3),
                status=GameStatus.FINAL,
                home_score=20,
                away_score=14,
                spread=Decimal("6.5"),
                total=Decimal("41.0"),
                favorite_team_id=tid[2],
                underdog_team_id=tid[3],
            )
            session.add_all([game_fav, game_total])
            session.commit()
            session.refresh(game_fav)
            session.refresh(game_total)
            assert game_fav.id is not None and game_total.id is not None
            self.game_fav_id = game_fav.id
            self.game_total_id = game_total.id

            # --- Users -------------------------------------------------------
            pw = hash_password("correct horse battery staple")
            user_a = User(display_name="alice", password_hash=pw, is_active=True)
            user_b = User(display_name="bob", password_hash=pw, is_active=True)
            user_c = User(display_name="carol", password_hash=pw, is_active=True)
            session.add_all([user_a, user_b, user_c])
            session.commit()
            session.refresh(user_a)
            session.refresh(user_b)
            session.refresh(user_c)
            assert (
                user_a.id is not None
                and user_b.id is not None
                and user_c.id is not None
            )
            self.user_a_id = user_a.id
            self.user_b_id = user_b.id
            self.user_c_id = user_c.id

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

    def _seed_pick(
        self,
        *,
        user_id: int,
        game_id: int,
        pick_type: PickType,
        is_mortal_lock: bool = False,
    ) -> int:
        with self._session() as session:
            pick = Pick(
                user_id=user_id,
                game_id=game_id,
                week_id=self.week_id,
                pick_type=pick_type,
                is_mortal_lock=is_mortal_lock,
            )
            session.add(pick)
            session.commit()
            session.refresh(pick)
            assert pick.id is not None
            return pick.id

    def _set_discord_identity(
        self, user_id: int, *, discord_id: int | None, avatar_hash: str | None
    ) -> None:
        """Set a user's Discord avatar identity (discord_id + avatar hash)."""
        with self._session() as session:
            user = session.get(User, user_id)
            assert user is not None
            user.discord_id = discord_id
            user.discord_avatar_hash = avatar_hash
            session.add(user)
            session.commit()

    def _open_the_window(self) -> None:
        """Push EVERY week-1 game's kickoff into the future so the window is OPEN.

        The setUp games are FINAL with PAST kickoffs, so the week's earliest
        kickoff is in the past and ``compute_window(week_games).close_at`` is
        already CLOSED. The week-level visibility gate keys off that single
        boundary, so to exercise the WINDOW-OPEN state (other users hidden) every
        kickoff in the week must be in the FUTURE. This rewrites the in-store
        kickoffs (the gate reads them via ``compute_window``) and is explicit so
        the window-open tests don't depend on accidental ordering.
        """
        future = datetime.now(timezone.utc) + timedelta(days=2)
        with self._session() as session:
            games = list(
                session.exec(select(Game).where(Game.week_id == self.week_id)).all()
            )
            for offset, game in enumerate(games):
                game.kickoff_at = future + timedelta(hours=offset)
                session.add(game)
            session.commit()

    def _seed_future_game(self) -> int:
        """Insert a NOT-yet-locked (future-kickoff) game in the scored week.

        The setUp games are FINAL with PAST kickoffs (the week's earliest kickoff
        is in the past, so the week-level window is already CLOSED); this adds a
        SCHEDULED game whose kickoff is in the FUTURE, used to prove that once the
        window is closed even a not-yet-kicked-off game's other-user pick is now
        VISIBLE. It reuses the existing teams (tid 1 & 2) and carries spread/
        total/favorite/underdog like the FINAL games so a pick on it is
        well-formed.
        """
        future = datetime.now(timezone.utc) + timedelta(days=2)
        with self._session() as session:
            teams = list(session.exec(select(Team)).all())
            tid = [t.id for t in sorted(teams, key=lambda t: t.id or 0)]
            game = Game(
                espn_event_id=1003,
                week_id=self.week_id,
                season=SEASON,
                week=WEEK,
                home_team_id=tid[0],
                away_team_id=tid[1],
                kickoff_at=future,
                status=GameStatus.SCHEDULED,
                spread=Decimal("3.5"),
                total=Decimal("44.5"),
                favorite_team_id=tid[0],
                underdog_team_id=tid[1],
            )
            session.add(game)
            session.commit()
            session.refresh(game)
            assert game.id is not None
            return game.id

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

    # -- service-level (Task 1) -------------------------------------------

    def test_season_standings_ordered_by_total_then_name(self) -> None:
        """season_standings ranks ALL picking users by (-season_total, name).

        alice: FAVORITE_COVER (win, +1) + UNDER mortal lock (win, +2) = 3.
        carol: FAVORITE_COVER (win, +1)                                = 1.
        bob:   FAVORITE_COVER (win, +1)                                = 1 (tie
               with carol -> broken by display_name 'bob' < 'carol').
        """
        self._seed_pick(
            user_id=self.user_a_id,
            game_id=self.game_fav_id,
            pick_type=PickType.FAVORITE_COVER,
        )
        self._seed_pick(
            user_id=self.user_a_id,
            game_id=self.game_total_id,
            pick_type=PickType.UNDER,
            is_mortal_lock=True,
        )
        self._seed_pick(
            user_id=self.user_b_id,
            game_id=self.game_fav_id,
            pick_type=PickType.FAVORITE_COVER,
        )
        self._seed_pick(
            user_id=self.user_c_id,
            game_id=self.game_fav_id,
            pick_type=PickType.FAVORITE_COVER,
        )

        with self._session() as session:
            standings, identities = season_standings(session, season=SEASON)

        names = [r.display_name for r in standings.results]
        totals = [r.season_total for r in standings.results]
        self.assertEqual(names, ["alice", "bob", "carol"])
        self.assertEqual(totals, [3, 1, 1])
        # alice's per-week score is the FINAL week-1 total.
        alice = standings.results[0]
        self.assertEqual(alice.weekly_scores, {WEEK: 3})
        # The identity map is keyed by the unique display_name and carries the
        # avatar-identity fields for every standing row.
        self.assertEqual(set(identities), {"alice", "bob", "carol"})
        for name in ("alice", "bob", "carol"):
            self.assertIsNone(identities[name].discord_id)
            self.assertIsNone(identities[name].discord_avatar_hash)

    def test_season_standings_excludes_users_with_no_picks(self) -> None:
        """A user with zero picks in the season does not appear at all."""
        self._seed_pick(
            user_id=self.user_a_id,
            game_id=self.game_fav_id,
            pick_type=PickType.FAVORITE_COVER,
        )
        with self._session() as session:
            standings, _ = season_standings(session, season=SEASON)
        self.assertEqual([r.display_name for r in standings.results], ["alice"])

    def test_season_is_complete_all_final(self) -> None:
        """All seeded games are FINAL -> the season is complete."""
        with self._session() as session:
            self.assertTrue(season_is_complete(session, season=SEASON))

    def test_season_is_complete_false_with_non_final_game(self) -> None:
        """Any non-FINAL game makes the season incomplete.

        Flip one seeded FINAL game to IN_PROGRESS; the season is no longer
        complete even though every other game is FINAL.
        """
        with self._session() as session:
            game = session.exec(
                select(Game).where(Game.id == self.game_fav_id)
            ).one()
            game.status = GameStatus.IN_PROGRESS
            session.add(game)
            session.commit()
        with self._session() as session:
            self.assertFalse(season_is_complete(session, season=SEASON))

    def test_season_is_complete_false_for_empty_season(self) -> None:
        """A season with zero games is NOT complete (the empty-season rule)."""
        with self._session() as session:
            self.assertFalse(season_is_complete(session, season=SEASON + 1))

    def test_week_results_shape_and_scores(self) -> None:
        """week_results carries each user's graded picks + weekly_score.

        Each user's weekly_score equals score_week over the same picks and
        equals the sum of its graded picks' points; graded picks carry a real
        (non-UNGRADEABLE) outcome + points.
        """
        self._seed_pick(
            user_id=self.user_a_id,
            game_id=self.game_fav_id,
            pick_type=PickType.FAVORITE_COVER,
        )
        self._seed_pick(
            user_id=self.user_a_id,
            game_id=self.game_total_id,
            pick_type=PickType.UNDER,
            is_mortal_lock=True,
        )
        self._seed_pick(
            user_id=self.user_b_id,
            game_id=self.game_total_id,
            pick_type=PickType.OVER,
        )

        with self._session() as session:
            results = week_results(session, season=SEASON, week=WEEK)
            # Build the same games index score_week would use, for a cross-check.
            from app.services.standings import _season_games_by_pk

            games_by_pk = _season_games_by_pk(session, season=SEASON)
            from app.models import Pick as _Pick
            from sqlmodel import select as _select

            picks_a = list(
                session.exec(
                    _select(_Pick).where(
                        _Pick.user_id == self.user_a_id,
                        _Pick.week_id == self.week_id,
                    )
                ).all()
            )
            expected_a = score_week(games_by_pk, picks_a)

        # Ordered (-weekly_score, name): alice (1+2=3) then bob (OVER loses, 0).
        self.assertEqual([r.display_name for r in results], ["alice", "bob"])
        alice = results[0]
        self.assertEqual(alice.weekly_score, 3)
        self.assertEqual(alice.weekly_score, expected_a)
        self.assertEqual(alice.weekly_score, sum(p.points for p in alice.picks))
        # alice's UNDER mortal lock won (+2); the FAVORITE_COVER won (+1).
        outcomes = {p.pick_type: (p.outcome, p.points) for p in alice.picks}
        self.assertEqual(outcomes[PickType.UNDER], (GradeOutcome.WIN.value, 2))
        self.assertEqual(
            outcomes[PickType.FAVORITE_COVER], (GradeOutcome.WIN.value, 1)
        )
        # bob's OVER lost on a FINAL game -> LOSS, 0 points.
        bob = results[1]
        self.assertEqual(bob.weekly_score, 0)
        self.assertEqual(bob.picks[0].outcome, GradeOutcome.LOSS.value)

    def test_week_results_empty_for_unknown_week(self) -> None:
        """A week with no Week row / no picks yields an empty list (pure read)."""
        with self._session() as session:
            self.assertEqual(week_results(session, season=SEASON, week=99), [])
            self.assertEqual(week_results(session, season=SEASON, week=WEEK), [])

    # -- privacy gate (leak-gate) -----------------------------------------

    def _picks_for(self, results, display_name):
        """The graded picks of the named user in a week_results list."""
        for r in results:
            if r.display_name == display_name:
                return r
        self.fail(f"{display_name} not present in results")

    def test_gate_hides_other_users_unlocked_pick(self) -> None:
        """WINDOW OPEN: another user's whole-week picks are hidden from the caller.

        With every week-1 kickoff pushed into the future the week's pick window is
        OPEN (now < the week's earliest kickoff), so the week-level gate hides ALL
        of bob's picks while the caller (alice) still sees her own.
        """
        self._open_the_window()
        self._seed_pick(
            user_id=self.user_b_id,
            game_id=self.game_fav_id,
            pick_type=PickType.FAVORITE_COVER,
        )
        self._seed_pick(
            user_id=self.user_a_id,
            game_id=self.game_total_id,
            pick_type=PickType.OVER,
        )

        with self._session() as session:
            results = week_results(
                session, season=SEASON, week=WEEK, caller_user_id=self.user_a_id
            )

        bob = self._picks_for(results, "bob")
        self.assertEqual(bob.picks, ())  # other user's picks hidden while open
        alice = self._picks_for(results, "alice")
        self.assertIn(self.game_total_id, {p.game_id for p in alice.picks})

    def test_gate_shows_other_users_locked_pick(self) -> None:
        """WINDOW CLOSED: another user's whole-week picks are visible to the caller.

        The default setUp window is already CLOSED (game_fav kicked off in the
        past), so the reveal trigger ("window closed") fires and bob's pick is
        visible to caller alice.
        """
        self._seed_pick(
            user_id=self.user_b_id,
            game_id=self.game_fav_id,
            pick_type=PickType.FAVORITE_COVER,
        )

        with self._session() as session:
            results = week_results(
                session, season=SEASON, week=WEEK, caller_user_id=self.user_a_id
            )

        bob = self._picks_for(results, "bob")
        self.assertIn(self.game_fav_id, {p.game_id for p in bob.picks})

    def test_gate_window_closed_reveals_later_unkicked_pick(self) -> None:
        """CRUX: once the window is CLOSED, a later not-yet-kicked-off game's
        other-user pick is VISIBLE (the OLD per-game gate hid it).

        The default setUp window is CLOSED because game_fav kicked off in the
        past. bob picks a SCHEDULED future game in the SAME week — under the new
        week-level rule that future-game pick is revealed to caller alice even
        though it has not kicked off, because all picking for the week is frozen.
        """
        future_game_id = self._seed_future_game()
        self._seed_pick(
            user_id=self.user_b_id,
            game_id=future_game_id,
            pick_type=PickType.FAVORITE_COVER,
        )

        with self._session() as session:
            results = week_results(
                session, season=SEASON, week=WEEK, caller_user_id=self.user_a_id
            )

        bob = self._picks_for(results, "bob")
        self.assertIn(future_game_id, {p.game_id for p in bob.picks})

    def test_gate_caller_always_sees_own_unlocked_pick(self) -> None:
        """The caller sees their OWN pick on a not-yet-kicked-off game (bypass).

        Works in either window state; pinned here to the WINDOW-OPEN state (all
        kickoffs future) so the caller bypass is what reveals the pick.
        """
        self._open_the_window()
        future_game_id = self._seed_future_game()
        self._seed_pick(
            user_id=self.user_a_id,
            game_id=future_game_id,
            pick_type=PickType.FAVORITE_COVER,
        )

        with self._session() as session:
            results = week_results(
                session, season=SEASON, week=WEEK, caller_user_id=self.user_a_id
            )

        alice = self._picks_for(results, "alice")
        self.assertIn(future_game_id, {p.game_id for p in alice.picks})

    def test_gate_preserves_weekly_score_under_redaction(self) -> None:
        """WINDOW OPEN: a fully-redacted other user keeps their full weekly_score.

        With the window OPEN (all kickoffs future) bob's picks are hidden from the
        caller, but his ``weekly_score`` is still computed over his FULL persisted
        pick set, so redaction never changes a score.
        """
        self._open_the_window()
        self._seed_pick(
            user_id=self.user_b_id,
            game_id=self.game_fav_id,
            pick_type=PickType.FAVORITE_COVER,
        )
        self._seed_pick(
            user_id=self.user_b_id,
            game_id=self.game_total_id,
            pick_type=PickType.UNDER,
        )

        with self._session() as session:
            results = week_results(
                session, season=SEASON, week=WEEK, caller_user_id=self.user_a_id
            )
            from app.services.standings import _season_games_by_pk

            games_by_pk = _season_games_by_pk(session, season=SEASON)
            picks_b = list(
                session.exec(
                    select(Pick).where(
                        Pick.user_id == self.user_b_id,
                        Pick.week_id == self.week_id,
                    )
                ).all()
            )
            expected_b = score_week(games_by_pk, picks_b)

        bob = self._picks_for(results, "bob")
        # bob's picks are fully redacted for the caller (alice) while open…
        self.assertEqual(bob.picks, ())
        # …but the score is computed over bob's FULL pick set (unchanged).
        self.assertEqual(bob.weekly_score, expected_b)

    def test_gate_http_redacts_other_users_unlocked_pick(self) -> None:
        """Over the wire (WINDOW OPEN): alice does not see any of bob's picks.

        Pushes every week kickoff into the future so the window is OPEN, then
        asserts bob's picks are absent over the wire and no ``user_id`` leaks.
        """
        self._open_the_window()
        self._seed_pick(
            user_id=self.user_b_id,
            game_id=self.game_total_id,
            pick_type=PickType.UNDER,
        )
        self._seed_pick(
            user_id=self.user_b_id,
            game_id=self.game_fav_id,
            pick_type=PickType.FAVORITE_COVER,
        )

        resp = self.client.get(
            "/api/results/week",
            params={"season": SEASON, "week": WEEK},
            headers=self._bearer_headers(self.user_a_id),
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        rows = resp.json()["results"]
        bob = next(r for r in rows if r["display_name"] == "bob")
        self.assertNotIn("user_id", bob)
        game_ids = {p["game_id"] for p in bob["picks"]}
        self.assertNotIn(self.game_total_id, game_ids)  # hidden while open
        self.assertNotIn(self.game_fav_id, game_ids)  # hidden while open

    # -- HTTP (Task 3) -----------------------------------------------------

    def test_unauthenticated_week_and_standings_rejected_401(self) -> None:
        """Unauthenticated GETs to both endpoints -> 401 envelope."""
        self._clear_auth()
        wk = self.client.get(
            "/api/results/week", params={"season": SEASON, "week": WEEK}
        )
        self.assertEqual(wk.status_code, 401, wk.text)
        self._assert_envelope(wk.json())

        self._clear_auth()
        st = self.client.get("/api/results/standings", params={"season": SEASON})
        self.assertEqual(st.status_code, 401, st.text)
        self._assert_envelope(st.json())

    def test_week_results_http_shape_for_authenticated_user(self) -> None:
        """GET /api/results/week returns each user's graded picks + score.

        A FINAL week is seeded for multiple users; the response shape carries
        per-user weekly_score (== score_week) and graded picks with outcome +
        points. Any authenticated user may read it (shared scoreboard).
        """
        self._seed_pick(
            user_id=self.user_a_id,
            game_id=self.game_fav_id,
            pick_type=PickType.FAVORITE_COVER,
        )
        self._seed_pick(
            user_id=self.user_a_id,
            game_id=self.game_total_id,
            pick_type=PickType.UNDER,
            is_mortal_lock=True,
        )
        self._seed_pick(
            user_id=self.user_b_id,
            game_id=self.game_total_id,
            pick_type=PickType.OVER,
        )
        # Give alice a Discord identity so the row carries a real hash; bob has
        # none, so his row must report null avatar fields.
        self._set_discord_identity(
            self.user_a_id, discord_id=4242, avatar_hash="abc123hash"
        )

        # carol (who did not pick) can still read the shared scoreboard.
        resp = self.client.get(
            "/api/results/week",
            params={"season": SEASON, "week": WEEK},
            headers=self._bearer_headers(self.user_c_id),
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["season"], SEASON)
        self.assertEqual(body["week"], WEEK)
        results = body["results"]
        # Ordered (-weekly_score, name): alice (3) then bob (0).
        self.assertEqual([r["display_name"] for r in results], ["alice", "bob"])
        self.assertNotIn("user_id", results[0])
        alice = results[0]
        self.assertEqual(alice["weekly_score"], 3)
        self.assertEqual(
            alice["weekly_score"], sum(p["points"] for p in alice["picks"])
        )
        for p in alice["picks"]:
            self.assertIn(p["outcome"], {o.value for o in GradeOutcome})
            self.assertNotEqual(p["outcome"], GradeOutcome.UNGRADEABLE.value)
        # Avatar identity threads through per-row: alice has a hash, bob null.
        self.assertEqual(alice["discord_id"], 4242)
        self.assertEqual(alice["discord_avatar_hash"], "abc123hash")
        bob = next(r for r in results if r["display_name"] == "bob")
        self.assertIsNone(bob["discord_id"])
        self.assertIsNone(bob["discord_avatar_hash"])

    def test_standings_http_ordering_with_tiebreak(self) -> None:
        """GET /api/results/standings ranks all users by (-total, name).

        alice 3, then a bob/carol tie at 1 broken by display_name.
        """
        self._seed_pick(
            user_id=self.user_a_id,
            game_id=self.game_fav_id,
            pick_type=PickType.FAVORITE_COVER,
        )
        self._seed_pick(
            user_id=self.user_a_id,
            game_id=self.game_total_id,
            pick_type=PickType.UNDER,
            is_mortal_lock=True,
        )
        self._seed_pick(
            user_id=self.user_b_id,
            game_id=self.game_fav_id,
            pick_type=PickType.FAVORITE_COVER,
        )
        self._seed_pick(
            user_id=self.user_c_id,
            game_id=self.game_fav_id,
            pick_type=PickType.FAVORITE_COVER,
        )
        # alice gets a Discord identity (hash present); bob/carol stay null.
        self._set_discord_identity(
            self.user_a_id, discord_id=9001, avatar_hash="deadbeefhash"
        )

        resp = self.client.get(
            "/api/results/standings",
            params={"season": SEASON},
            headers=self._bearer_headers(self.user_b_id),
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["season"], SEASON)
        # setUp seeds two FINAL games (no non-FINAL game) -> season complete.
        self.assertTrue(body["season_complete"])
        rows = body["standings"]
        self.assertEqual(
            [r["display_name"] for r in rows], ["alice", "bob", "carol"]
        )
        self.assertEqual([r["season_total"] for r in rows], [3, 1, 1])
        self.assertNotIn("user_id", rows[0])
        # weekly_scores is keyed by week number (JSON stringifies int keys).
        self.assertEqual(rows[0]["weekly_scores"], {str(WEEK): 3})
        # Avatar identity threads through per-row: alice has a hash, the others
        # (no Discord identity) report null for both avatar fields.
        self.assertEqual(rows[0]["discord_id"], 9001)
        self.assertEqual(rows[0]["discord_avatar_hash"], "deadbeefhash")
        for other in rows[1:]:
            self.assertIsNone(other["discord_id"])
            self.assertIsNone(other["discord_avatar_hash"])


if __name__ == "__main__":
    unittest.main()
