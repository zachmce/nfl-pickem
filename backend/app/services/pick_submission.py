"""Pick submission + read service — the DB-writing core behind /api/picks.

This is build-order step #3: the first user-facing write surface of the pick'em
domain. It turns the proven pure-logic layer (:mod:`app.services.pick_window` and
:mod:`app.services.pick_validation`) into an auth-guarded, conflict-safe write.

Design — service-layer, DB-focused, caller-owns-commit (mirrors
:mod:`app.services.refresh`):

* The business logic lives here and operates on a **passed-in** ``Session``. The
  router (:mod:`app.api.picks`) is a thin wrapper that resolves the session +
  current user and owns the commit. This module is unit-testable without HTTP.
* ``user_id`` is always supplied by the caller (the router derives it from the
  verified session, NEVER from the request body) — there is no user field in any
  argument that comes from untrusted input.
* ``now`` is injected (defaulting to the real UTC clock per PROJECT.md's
  "no virtual clock") and used only for window/lock decisions.

Enforcement order on every submit (rejected BEFORE any write):

  1. **Window** — :func:`pick_window.is_pick_open`; closed window rejects the
     whole submit (no rows persisted).
  2. **Per-game lock** — :func:`pick_window.is_game_locked`; a pick on a game
     that has kicked off is rejected.
  3. **Conflict / roster** — :func:`pick_validation.check_new_pick` against the
     user's existing picks for the week, accumulating newly-accepted picks so a
     batch is validated against itself too.

First-pick precedence (PROJECT.md: "existing pick takes precedence over a
conflicting new one"): because ``check_new_pick`` treats the existing picks as
authoritative, an incoming pick that conflicts with one the user already has is
REJECTED (existing wins). A legitimate **replace** of the user's OWN
non-conflicting base pick — same ``(week_id, pick_type, is_mortal_lock=false)``
slot but a DIFFERENT game — is an upsert: the existing row's ``game_id`` is
updated rather than inserting a duplicate (the DB partial unique index
``uq_pick_user_week_type_base`` enforces one base type per user/week).

Exception mapping lives HERE, using ONLY :mod:`app.exceptions` (the service never
imports from ``app.api``): :func:`violation_to_exception` maps each
:class:`~app.services.pick_validation.ViolationCode` to the typed exception the
global handler turns into the ``{"error": {"code", "message", "reason"}}``
envelope. Window-closed / game-locked rejections raise :class:`ConflictError`
directly. Every rejection is a structured 4xx — never a raw 500.

Purity boundary: imports only :mod:`app.models`, :mod:`app.exceptions`, the two
pure services, sqlmodel, and stdlib. It does NOT import :mod:`app.config`, the
ESPN adapter, or any network layer, and does NOT modify ``scoring.py`` /
``pick_validation.py`` / ``pick_window.py``.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Session, select

from app.exceptions import ApiException, ConflictError, NotFoundError, ValidationError
from app.models import Game, Pick, PickType, Week
from app.schemas.picks import PickItem
from app.services.pick_validation import ViolationCode, check_new_pick
from app.services.pick_window import compute_window, is_game_locked, is_pick_open

# ViolationCode → typed exception. Conflict/roster rejections are 409; a spread
# pick on a true pick'em is a 422 eligibility failure. ``reason`` is the raw
# ViolationCode value so the envelope carries the machine-readable cause.
_CONFLICT_CODES = frozenset(
    {
        ViolationCode.DUPLICATE_PICK,
        ViolationCode.CONTRADICTORY_PICK,
        ViolationCode.MULTIPLE_MORTAL_LOCKS,
    }
)


def violation_to_exception(code: ViolationCode, message: str) -> ApiException:
    """Map a :class:`ViolationCode` to its typed envelope exception.

    DUPLICATE/CONTRADICTORY/MULTIPLE_MORTAL_LOCKS -> :class:`ConflictError` (409);
    PICKEM_SPREAD_INELIGIBLE -> :class:`ValidationError` (422). The ``reason`` is
    the ViolationCode value so callers/clients get the machine-readable cause.
    """
    if code in _CONFLICT_CODES:
        return ConflictError(message, reason=code.value)
    if code is ViolationCode.PICKEM_SPREAD_INELIGIBLE:
        return ValidationError(message, reason=code.value)
    # Defensive: an unmapped code is still a structured 4xx, never a raw 500.
    return ConflictError(message, reason=code.value)


def _as_aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from the store.

    ``DateTime(timezone=True)`` round-trips NAIVE on SQLite (Postgres preserves
    tz). The window/lock math compares against tz-aware values, so this
    normalizes for the comparison ONLY — the normalized copy is never persisted,
    leaving production-on-Postgres unaffected. Mirrors
    :func:`app.services.refresh._as_aware`.
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _normalized_game(game: Game) -> Game:
    """A shallow copy of ``game`` with a tz-aware kickoff for window/lock math.

    Avoids mutating (and thus dirtying/persisting) the live row. Mirrors
    :func:`app.services.refresh._normalized_games` but for a single game.
    """
    return Game(
        espn_event_id=game.espn_event_id,
        week_id=game.week_id,
        season=game.season,
        week=game.week,
        home_team_id=game.home_team_id,
        away_team_id=game.away_team_id,
        kickoff_at=_as_aware(game.kickoff_at),
        status=game.status,
        spread=game.spread,
        total=game.total,
        favorite_team_id=game.favorite_team_id,
        underdog_team_id=game.underdog_team_id,
    )


def _resolve_week(session: Session, season: int, week: int) -> Week:
    """Load the Week row for ``{season, week}`` or raise :class:`NotFoundError`."""
    row = session.exec(
        select(Week).where(Week.season == season, Week.week == week)
    ).first()
    if row is None:
        raise NotFoundError(
            f"No week {week} for season {season}.",
            reason="week_not_found",
        )
    return row


def _load_week_games(session: Session, season: int, week: int) -> list[Game]:
    """All Game rows for ``{season, week}`` (live rows, not normalized copies)."""
    return list(
        session.exec(
            select(Game).where(Game.season == season, Game.week == week)
        ).all()
    )


def read_picks(
    session: Session, *, user_id: int, season: int, week: int
) -> list[Pick]:
    """Return ONLY ``user_id``'s Pick rows for ``{season, week}``.

    Scoped to the caller by construction — there is no parameter to ask for
    another user's picks (IDOR-safe). Resolves the Week so an unknown
    ``{season, week}`` raises :class:`NotFoundError` rather than silently
    returning an empty list.
    """
    week_row = _resolve_week(session, season, week)
    return list(
        session.exec(
            select(Pick).where(
                Pick.user_id == user_id, Pick.week_id == week_row.id
            )
        ).all()
    )


def submit_picks(
    session: Session,
    *,
    user_id: int,
    season: int,
    week: int,
    items: list[PickItem],
    now: datetime | None = None,
) -> list[Pick]:
    """Validate then persist ``user_id``'s picks for ``{season, week}``.

    Enforces, IN ORDER and BEFORE any write: window open -> per-game lock ->
    conflict/roster (first-pick precedence). On any rejection raises a typed
    structured 4xx (no rows persisted). The service ``session.add(...)``s
    accepted picks but does NOT commit — the caller (router/test) owns the
    commit, matching :mod:`app.services.refresh`'s "caller commits" contract.

    Newly-accepted picks in this batch are folded into the in-progress existing
    set, so a batch is validated against itself too. A non-conflicting replace of
    the user's own base slot (same ``pick_type``/non-lock, different game)
    upserts the existing row's ``game_id`` rather than inserting a duplicate.

    Returns the persisted (added/updated) Pick rows, in submission order.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    week_row = _resolve_week(session, season, week)
    assert week_row.id is not None  # a persisted week always has an id

    week_games = _load_week_games(session, season, week)
    games_by_id = {g.id: g for g in week_games}
    norm_by_id = {g.id: _normalized_game(g) for g in week_games}

    # Previous week's games (if any) for the window-open boundary.
    prev_games = _load_week_games(session, season, week - 1) if week > 1 else []
    prev_norm = [_normalized_game(g) for g in prev_games] or None

    # 1) Window — reject the whole submit if the week is closed (no writes).
    window = compute_window(list(norm_by_id.values()), prev_norm)
    if not is_pick_open(window, now):
        raise ConflictError(
            f"The pick window for season {season} week {week} is closed.",
            reason="window_closed",
        )

    # Validate every item's game belongs to this week before deciding locks.
    for item in items:
        if item.game_id not in games_by_id:
            raise NotFoundError(
                f"Game {item.game_id} is not part of season {season} week {week}.",
                reason="game_not_in_week",
            )

    # 2) Per-game lock — reject a pick whose game has already kicked off.
    for item in items:
        if is_game_locked(norm_by_id[item.game_id], now):
            raise ConflictError(
                f"Game {item.game_id} has kicked off; picks on it are locked.",
                reason="game_locked",
            )

    # 3) Conflict / roster — validate each item against the user's existing
    # picks for the week, accumulating accepted picks so the batch is checked
    # against itself too. Use normalized games for pick'em eligibility.
    existing = list(
        session.exec(
            select(Pick).where(
                Pick.user_id == user_id, Pick.week_id == week_row.id
            )
        ).all()
    )
    accepted: list[Pick] = list(existing)
    persisted: list[Pick] = []

    for item in items:
        candidate = Pick(
            user_id=user_id,
            game_id=item.game_id,
            week_id=week_row.id,
            pick_type=item.pick_type,
            is_mortal_lock=item.is_mortal_lock,
        )
        decision = check_new_pick(candidate, accepted, norm_by_id)
        if not decision.ok:
            v = decision.violations[0]
            raise violation_to_exception(v.code, v.message)

        # Non-conflicting own-slot replace: same base slot (week, type, non-lock)
        # but a DIFFERENT game -> upsert the existing row's game_id instead of
        # inserting a duplicate (DB partial unique index enforces one base type).
        replaced = _find_replaceable_base_slot(accepted, item)
        if replaced is not None:
            replaced.game_id = item.game_id
            session.add(replaced)
            persisted.append(replaced)
        else:
            session.add(candidate)
            accepted.append(candidate)
            persisted.append(candidate)

    return persisted


def _find_replaceable_base_slot(
    accepted: list[Pick], item: PickItem
) -> Pick | None:
    """Find an existing OWN base pick this item should replace (upsert), if any.

    A replace applies only to a BASE (non-mortal-lock) slot: same ``pick_type``
    and ``is_mortal_lock=False`` as the incoming item, on a DIFFERENT game. The
    earlier ``check_new_pick`` pass guarantees the item does not conflict, so the
    only reason it would otherwise collide is the per-(user, week, base-type)
    unique index — which a replace satisfies by moving the existing row's game.
    """
    if item.is_mortal_lock:
        return None
    for pick in accepted:
        if (
            not pick.is_mortal_lock
            and pick.pick_type == item.pick_type
            and pick.game_id != item.game_id
        ):
            return pick
    return None
