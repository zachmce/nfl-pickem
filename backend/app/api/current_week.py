"""Authenticated current-week READ endpoint for the SPA context bar.

A thin GET-only router mirroring :mod:`app.api.results`: it authenticates the
caller, reads the persisted games, reuses the pure
:mod:`app.services.pick_window` logic, and shapes the response. No window math is
re-implemented here — :func:`~app.services.pick_window.compute_window` /
:func:`~app.services.pick_window.is_pick_open` remain the single source of truth;
the router only selects the current week and derives the four-state.

Security — deliberate shared-read posture
------------------------------------------

The endpoint requires authentication (``get_current_user`` -> 401 envelope when
unauthenticated) but is intentionally **NOT user-scoped**: the context bar is the
same for every member (season, week, window state, close time — no per-user
data). This is a reviewed, deliberate choice (same posture as
:mod:`app.api.results`), not an IDOR oversight. The response carries no
``user_id`` and no per-user surrogate.

Demo correctness — no ``IS_DEMO_DATA`` branch
---------------------------------------------

The four-state is computed from the REAL clock (``datetime.now(timezone.utc)``)
against the PERSISTED kickoffs. In demo mode the seed rigidly time-shifts those
kickoffs forward (via the shared anchor/offset), so the same real-now-vs-persisted
comparison is automatically demo-correct — byte-for-byte the pattern
``refresh_games`` / ``submit_picks`` / ``standings`` already use. This module
therefore does NOT import :mod:`app.config`, ``settings.is_demo_data``, or any
:mod:`app.demo` module.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.api.deps import get_current_user
from app.db import get_session
from app.exceptions import NotFoundError
from app.models import Game, GameStatus, User
from app.schemas.current_week import CurrentWeekResponse, PickWindowState
from app.services.pick_window import PickWindow, compute_window, is_pick_open
from app.services.standings import active_season, season_is_complete

router = APIRouter(prefix="/api/current-week", tags=["current-week"])


def _as_aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from the store.

    ``DateTime(timezone=True)`` round-trips NAIVE on SQLite (Postgres preserves
    tz). The window math compares against tz-aware values, so this normalizes a
    naive value to UTC for the comparison ONLY — the normalized copy is never
    persisted back (mirrors :func:`app.services.refresh._as_aware`).
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _normalized(games: list[Game]) -> list[Game]:
    """Shallow copies of ``games`` with tz-aware kickoffs for window math.

    Never mutates the live store-read rows (which would dirty/persist them):
    each copy re-attaches UTC to a naive ``kickoff_at`` so
    :func:`compute_window` can be called safely.
    """
    return [
        Game(
            espn_event_id=g.espn_event_id,
            week_id=g.week_id,
            season=g.season,
            week=g.week,
            home_team_id=g.home_team_id,
            away_team_id=g.away_team_id,
            kickoff_at=_as_aware(g.kickoff_at),
            status=g.status,
        )
        for g in games
    ]


@router.get("", response_model=CurrentWeekResponse)
def read_current_week(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> CurrentWeekResponse:
    """Resolve the current week + its pick-window state for the context bar.

    Shared read: authenticated but NOT user-scoped (see module docstring). The
    active season is the newest persisted one
    (:func:`app.services.standings.active_season` -> ``max(Game.season)``); a
    multi-season DB resolves the newest season rather than raising. Within that
    season the current week is the earliest week whose window has not yet closed
    (``now < close_at``); if every week is closed, the latest week. The
    four-state is derived from real ``now`` vs the persisted (possibly demo
    time-shifted) kickoffs — no demo branch.
    """
    now = datetime.now(timezone.utc)

    all_games = list(session.exec(select(Game)).all())
    if not all_games:
        raise NotFoundError("no seeded games to derive the current week from")

    # The newest persisted season is active (shared selector). all_games is
    # non-empty here, so season is non-None — no redundant raise needed.
    season = active_season(session)

    # Restrict every downstream computation to the chosen season's games, so a
    # stray other-season week cannot leak into the window math or the all-FINAL
    # check. Group THIS season's games by week number.
    season_games = [g for g in all_games if g.season == season]
    by_week: dict[int, list[Game]] = {}
    for g in season_games:
        by_week.setdefault(g.week, []).append(g)
    week_numbers = sorted(by_week)

    # Compute each week's window, reusing the pure pick_window service. Weeks
    # with no kickoff to close on (compute_window ValueError) are skipped.
    windows: dict[int, PickWindow] = {}
    for idx, wk in enumerate(week_numbers):
        this_games = _normalized(by_week[wk])
        prev_games = (
            _normalized(by_week[week_numbers[idx - 1]]) if idx > 0 else None
        )
        try:
            windows[wk] = compute_window(this_games, prev_games)
        except ValueError:
            continue

    if not windows:
        raise NotFoundError("no week has a kickoff to derive a pick window from")

    # Current week: the smallest week number still open (now < close_at);
    # otherwise the largest week that produced a window.
    open_weeks = [wk for wk in sorted(windows) if now < windows[wk].close_at]
    chosen_week = open_weeks[0] if open_weeks else max(windows)
    window = windows[chosen_week]

    # Derive the four-state from the pure window logic + this week's game status.
    if window.open_at is not None and now < window.open_at:
        state = PickWindowState.NOT_YET_OPEN
    elif is_pick_open(window, now):
        state = PickWindowState.OPEN
    else:
        # Window is closed (now >= close_at): locked while any game is non-FINAL,
        # closed once every game in the week is FINAL.
        all_final = all(g.status == GameStatus.FINAL for g in by_week[chosen_week])
        state = PickWindowState.CLOSED if all_final else PickWindowState.LOCKED

    complete = season_is_complete(session, season=season)

    return CurrentWeekResponse(
        season=season,
        week=chosen_week,
        window_state=state,
        window_closes_at=window.close_at,
        season_complete=complete,
    )
