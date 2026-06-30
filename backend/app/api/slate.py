"""Authenticated pickable-slate READ endpoint for the My Picks page.

A thin GET-only router mirroring :mod:`app.api.current_week` / :mod:`app.api.results`:
it authenticates the caller, reads the week's persisted games + their teams, reuses
the pure eligibility + lock logic, and shapes the response. No eligibility or lock
rule is re-implemented here — :func:`app.services.pick_validation.is_pick_type_eligible`
and :func:`app.services.pick_window.is_game_locked` remain the single source of
truth; the router only selects the week's games, resolves team identity, and maps
the values onto the wire schema.

This is the "choices" half of the My Picks page: the page learns the week from
``GET /api/current-week``, fetches the slate (the options) here, and separately
fetches the user's own picks from ``GET /api/picks``. The slate carries no picks.

Security — deliberate shared-read posture
------------------------------------------

The endpoint requires authentication (``get_current_user`` -> 401 envelope when
unauthenticated) but is intentionally **NOT user-scoped**: the slate is the same
for every member (the week's games, lines, lock + eligibility — no per-user data).
This is a reviewed, deliberate choice (same posture as :mod:`app.api.results` /
:mod:`app.api.current_week`), not an IDOR oversight. The response carries no
``user_id`` and no per-user surrogate; the only identity surfaced is public Team
reference data.

Demo correctness — no ``IS_DEMO_DATA`` branch
---------------------------------------------

The per-game ``locked`` bool AND the week-level ``odds_frozen`` bool are both
computed from the REAL clock (``datetime.now(timezone.utc)``) against the
PERSISTED kickoffs (``odds_frozen`` via the shared
:func:`app.services.odds.is_odds_frozen` predicate — no freeze math is
re-implemented here). In demo mode the seed rigidly time-shifts those kickoffs
forward (via the shared anchor/offset), so the same real-now-vs-persisted
comparison is automatically demo-correct — the pattern ``refresh_games`` /
``current_week`` already use. This module therefore does NOT import
:mod:`app.config`, ``settings.is_demo_data``, or any :mod:`app.demo` module.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.api.deps import get_current_user
from app.db import get_session
from app.models import Game, PickType, Team, User, Week
from app.schemas.slate import SlateGame, SlateResponse, SlateTeam
from app.services.odds import is_odds_frozen
from app.services.pick_validation import is_pick_type_eligible
from app.services.pick_window import is_game_locked

router = APIRouter(prefix="/api/slate", tags=["slate"])


def _as_aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from the store.

    ``DateTime(timezone=True)`` round-trips NAIVE on SQLite (Postgres preserves
    tz). The lock math compares against tz-aware values, so this normalizes a
    naive value to UTC for the comparison ONLY — the normalized copy is never
    persisted back (mirrors :func:`app.api.current_week._as_aware`).
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@router.get("", response_model=SlateResponse)
def read_slate(
    season: int,
    week: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> SlateResponse:
    """The pickable slate (the OPTIONS) for ``{season, week}``.

    Shared read: authenticated but NOT user-scoped (see module docstring). Returns
    one entry per game in the week — identity (game_id, kickoff, home/away team),
    the persisted line, a per-game ``locked`` bool, and per-``PickType``
    eligibility — ordered by kickoff then game_id. An empty/unknown ``{season,
    week}`` yields an empty ``games`` list (a pure read, never a 404), matching
    ``/api/results/week``. No picks are read.
    """
    now = datetime.now(timezone.utc)

    games = list(session.exec(select(Game).where(Game.season == season, Game.week == week)).all())

    # Week-level computed freeze predicate. REUSE app.services.odds.is_odds_frozen
    # against the SAME real-clock now (no freeze math re-implemented; same posture
    # as ``locked`` — real now vs persisted kickoffs, no IS_DEMO_DATA branch).
    # An unknown/kickoff-less week must stay a clean 200 (the slate is a pure read
    # that returns empty games for unknown weeks), so default to False and never
    # let it raise to the caller:
    #   * no Week row for {season, week}  -> False
    #   * is_odds_frozen raises ValueError (e.g. no kickoff to reason about) -> False
    week_row = session.exec(
        select(Week).where(Week.season == season, Week.week == week)
    ).one_or_none()
    odds_frozen = False
    if week_row is not None:
        try:
            odds_frozen = is_odds_frozen(week_row, games, now=now)
        except ValueError:
            odds_frozen = False

    # Resolve referenced teams in one query (no N+1), keyed by id.
    team_ids = {g.home_team_id for g in games} | {g.away_team_id for g in games}
    teams_by_id: dict[int, Team] = {}
    if team_ids:
        for t in session.exec(select(Team).where(Team.id.in_(team_ids))).all():
            assert t.id is not None
            teams_by_id[t.id] = t

    def _team(team_id: int) -> SlateTeam:
        t = teams_by_id[team_id]
        assert t.id is not None
        return SlateTeam(
            team_id=t.id,
            abbreviation=t.abbreviation,
            display_name=t.display_name,
        )

    # Stable order: kickoff then game_id (null kickoffs sort last).
    _MAX_KO = datetime.max.replace(tzinfo=timezone.utc)
    games.sort(key=lambda g: (_as_aware(g.kickoff_at) or _MAX_KO, g.id or 0))

    slate_games: list[SlateGame] = []
    for g in games:
        # Lock math on a tz-aware shallow copy so the live row is never dirtied.
        locked_game = Game(
            espn_event_id=g.espn_event_id,
            week_id=g.week_id,
            season=g.season,
            week=g.week,
            home_team_id=g.home_team_id,
            away_team_id=g.away_team_id,
            kickoff_at=_as_aware(g.kickoff_at),
            status=g.status,
        )
        assert g.id is not None
        slate_games.append(
            SlateGame(
                game_id=g.id,
                kickoff_at=g.kickoff_at,
                home_team=_team(g.home_team_id),
                away_team=_team(g.away_team_id),
                spread=g.spread,
                total=g.total,
                favorite_team_id=g.favorite_team_id,
                underdog_team_id=g.underdog_team_id,
                locked=is_game_locked(locked_game, now),
                eligibility={t: is_pick_type_eligible(g, t) for t in PickType},
            )
        )

    return SlateResponse.from_games(
        season=season, week=week, games=slate_games, odds_frozen=odds_frozen
    )
