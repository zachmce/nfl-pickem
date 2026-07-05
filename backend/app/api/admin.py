"""Authenticated web admin user-management router (QT-C).

The web counterpart to the Discord ``/admin`` cog: list every user and
deactivate / reactivate / grant-admin / revoke-admin / delete any OTHER user.
Every route sits behind :func:`app.api.deps.require_admin` (non-admin -> 403,
unauthenticated -> 401) and resolves the acting admin from the verified session
(``admin.id``) — there is NO caller/actor field in the path or body beyond the
target ``{user_id}``, so there is no IDOR and no privilege-escalation surface.

All legality enforcement (self-guards, last-admin guard, existence/state checks)
lives in :mod:`app.services.admin`; this router only translates HTTP <-> service.
The service raises ``ValueError`` whose leading whitespace-delimited token is a
stable machine code, which the router splits off and maps to a typed exception:

* ``user_not_found`` -> :class:`~app.exceptions.NotFoundError` (404)
* everything else (``cannot_act_on_self`` / ``last_admin`` / ``already_*`` /
  ``not_admin``) -> :class:`~app.exceptions.ConflictError` (409)

The stable code is passed as ``reason=`` and the service's human message as the
envelope ``message``, so the global handler emits
``{"error": {"code", "message", "reason"}}``.

The mutating POST/DELETE routes work WITH the existing double-submit CSRF
middleware exactly as :mod:`app.api.picks` relies on (cookie auth requires
``X-CSRF-Token``; bearer is exempt) — there is no CSRF special-casing here.

Response contract (pinned — Task 3 asserts it): the four POST mutations all
return HTTP 200 with the updated target as :class:`AdminUserRead` (including its
refreshed ``pick_count``); DELETE returns 204 with no body.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.api.deps import require_admin
from app.db import get_session
from app.exceptions import ConflictError, NotFoundError
from app.models import Game, PickResult, PickType, Team, User
from app.schemas.admin import (
    AdminUserListResponse,
    AdminUserRead,
    BotPersonalityRead,
    FreezeWeekRequest,
    IngestSeasonRequest,
    SetBotPersonalityRequest,
)
from app.schemas.admin_picks import AdminMiscGradeRequest, AdminPickSetRequest
from app.schemas.picks import PickRead
from app.tasks import freeze_week_task, ingest_season_task
from app.services.admin import (
    AdminUserRow,
    deactivate_user,
    delete_user,
    grant_admin,
    list_users,
    reactivate_user,
    revoke_admin,
)
from app.services.admin_picks import (
    admin_clear_pick,
    admin_grade_misc,
    admin_set_pick,
)
from app.services.notifications import (
    admin_pick_cleared_event,
    admin_pick_set_event,
    misc_graded_event,
    pick_log_detail,
    publish_event,
)
from app.services.app_settings import (
    get_bot_personality,
    set_bot_personality,
)
from app.bot.personality import available_personality_ids
from app.services.pick_submission import (
    _load_week_games,
    _normalized_game,
    read_picks,
)
from app.services.pick_window import compute_window, is_pick_open

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _abbr(session: Session, team_id: int | None) -> str | None:
    """Resolve a team id to its display abbreviation, or None when absent."""
    if team_id is None:
        return None
    team = session.get(Team, team_id)
    return team.abbreviation if team is not None else None


def _target_display_name(session: Session, user_id: int) -> str:
    """Resolve the override target's display_name (server-resolved, never client)."""
    target = session.get(User, user_id)
    return target.display_name if target is not None else str(user_id)


def _week_window_closed(session: Session, season: int, week: int) -> bool:
    """Whether ``{season, week}``'s pick window is CLOSED at now — best-effort.

    Reuses the PURE window predicates the same way standings.py / pick_submission.py
    do (load this week's games + the previous week's for the open boundary, normalize
    each to a tz-aware copy, build the window with :func:`compute_window`, and check
    :func:`is_pick_open` at now) — it does NOT re-implement window math. The whole
    computation is wrapped so an empty / kickoff-less week (which would raise out of
    ``compute_window``) degrades to ``False`` ("treat as NOT closed → do not publish")
    rather than breaking the caller — the leak-safe default is to suppress the chat
    publish (T-w9w-01 / T-w9w-03).
    """
    try:
        week_games = _load_week_games(session, season, week)
        norm = [_normalized_game(g) for g in week_games]
        prev_games = _load_week_games(session, season, week - 1) if week > 1 else []
        prev_norm = [_normalized_game(g) for g in prev_games] or None
        window = compute_window(norm, prev_norm)
        return not is_pick_open(window, datetime.now(timezone.utc))
    except Exception:
        return False


def _raise_for_service_error(exc: ValueError) -> None:
    """Map a service ``ValueError`` (leading stable code) to a typed exception.

    ``user_not_found`` -> 404; every other guard code -> 409. The full human
    message is preserved as the envelope message; the stable code becomes
    ``reason=``.
    """
    message = str(exc)
    code = message.split(":", 1)[0].split(None, 1)[0].strip()
    if code == "user_not_found":
        raise NotFoundError(message, reason=code) from exc
    raise ConflictError(message, reason=code) from exc


@router.get("/users", response_model=AdminUserListResponse)
def get_users(
    admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> AdminUserListResponse:
    """List every user (admin only): identity, role/state flags, and pick_count."""
    return AdminUserListResponse.from_rows(list_users(session))


def _read(row: AdminUserRow) -> AdminUserRead:
    return AdminUserRead.from_row(row)


@router.post("/users/{user_id}/deactivate", response_model=AdminUserRead)
def deactivate(
    user_id: int,
    admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> AdminUserRead:
    """Deactivate another user. 409 on self / last-active-admin / already-inactive."""
    assert admin.id is not None
    try:
        return _read(deactivate_user(session, caller_id=admin.id, user_id=user_id))
    except ValueError as exc:
        _raise_for_service_error(exc)
        raise  # unreachable; satisfies the type checker


@router.post("/users/{user_id}/reactivate", response_model=AdminUserRead)
def reactivate(
    user_id: int,
    admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> AdminUserRead:
    """Reactivate a deactivated user. 409 on already-active; 404 on missing."""
    try:
        return _read(reactivate_user(session, user_id=user_id))
    except ValueError as exc:
        _raise_for_service_error(exc)
        raise


@router.post("/users/{user_id}/grant-admin", response_model=AdminUserRead)
def grant(
    user_id: int,
    admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> AdminUserRead:
    """Grant admin to a user. 409 on already-admin; 404 on missing."""
    try:
        return _read(grant_admin(session, user_id=user_id))
    except ValueError as exc:
        _raise_for_service_error(exc)
        raise


@router.post("/users/{user_id}/revoke-admin", response_model=AdminUserRead)
def revoke(
    user_id: int,
    admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> AdminUserRead:
    """Revoke admin from another user. 409 on self / last-admin / not-admin."""
    assert admin.id is not None
    try:
        return _read(revoke_admin(session, caller_id=admin.id, user_id=user_id))
    except ValueError as exc:
        _raise_for_service_error(exc)
        raise


@router.delete("/users/{user_id}", status_code=204)
def delete(
    user_id: int,
    admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> None:
    """Delete another user (their picks cascade). 409 on self / last-admin; 404 missing."""
    assert admin.id is not None
    try:
        delete_user(session, caller_id=admin.id, user_id=user_id)
    except ValueError as exc:
        _raise_for_service_error(exc)
    return None


# --------------------------------------------------------------------------- #
# Admin pick-override (QT-1)
#
# GET/PUT/DELETE /api/admin/users/{user_id}/picks let an admin read/set/clear ANY
# user's pick for a week — past or upcoming — bypassing the window/lock but KEEPING
# roster integrity, recording every override in PickEditAudit. Acting on ANOTHER
# user is the whole point (admin convenience for a small group of friends): the
# caller is the session admin (``admin.id``), the target is the path ``{user_id}``
# — NOT IDOR, but it MUST stay require_admin-gated (T-m66-01/T-m66-02).
#
# Unlike the user-management routes above (whose service raises ``ValueError`` with
# a leading stable code), :mod:`app.services.admin_picks` raises typed
# ``ApiException`` subclasses directly (NotFoundError/ConflictError/ValidationError),
# each already carrying status + reason — so they propagate straight to the global
# handler with no per-route mapping.
# --------------------------------------------------------------------------- #


@router.get("/users/{user_id}/picks", response_model=list[PickRead])
def get_user_picks(
    user_id: int,
    season: int,
    week: int,
    admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> list[PickRead]:
    """List the PATH user's picks for ``{season, week}`` (admin only)."""
    picks = read_picks(session, user_id=user_id, season=season, week=week)
    return [PickRead.from_orm_pick(p) for p in picks]


@router.put("/users/{user_id}/picks", response_model=PickRead)
def set_user_pick(
    user_id: int,
    season: int,
    week: int,
    payload: AdminPickSetRequest,
    admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> PickRead:
    """Set/add/change the PATH user's slot (window/lock bypassed, roster kept)."""
    assert admin.id is not None
    pick = admin_set_pick(
        session,
        caller_id=admin.id,
        target_user_id=user_id,
        season=season,
        week=week,
        game_id=payload.game_id,
        pick_type=payload.pick_type,
        is_mortal_lock=payload.is_mortal_lock,
        misc_text=payload.misc_text,
    )
    session.commit()
    session.refresh(pick)

    # Post-commit, best-effort pickem-logger publish (AFTER the commit succeeds).
    # Resolve the TARGET user's display_name + the game's team abbreviations for a
    # concise admin-override line. publish_event swallows Redis errors so logging
    # can never break the override.
    game = session.get(Game, pick.game_id)
    detail = pick_log_detail(
        pick.pick_type,
        pick.is_mortal_lock,
        pick.misc_text,
        favorite_abbr=_abbr(session, game.favorite_team_id) if game else None,
        underdog_abbr=_abbr(session, game.underdog_team_id) if game else None,
        home_abbr=_abbr(session, game.home_team_id) if game else None,
        away_abbr=_abbr(session, game.away_team_id) if game else None,
    )
    publish_event(
        admin_pick_set_event(
            target=_target_display_name(session, user_id),
            week=week,
            detail=detail,
        )
    )
    return PickRead.from_orm_pick(pick)


@router.put("/users/{user_id}/picks/misc-grade", response_model=PickRead)
def grade_user_misc_pick(
    user_id: int,
    season: int,
    week: int,
    payload: AdminMiscGradeRequest,
    admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> PickRead:
    """Grade the PATH user's MISC pick (mark correct/incorrect + set points).

    Writes ``Pick.result`` / ``Pick.points`` (authoritative for MISC) plus one
    PickEditAudit row in the same txn. Window/lock are bypassed (grading is
    post-hoc). The service's typed exceptions propagate to the global handler.
    """
    assert admin.id is not None
    pick = admin_grade_misc(
        session,
        caller_id=admin.id,
        target_user_id=user_id,
        season=season,
        week=week,
        result=payload.result,
        points=payload.points,
    )
    session.commit()
    session.refresh(pick)

    # Post-commit, best-effort pickem-CHAT publish (AFTER the commit succeeds),
    # mirroring the set/clear routes' "publish in the ROUTE, never in the pure
    # service" pattern. LEAK GUARD (T-w9w-01): misc_text is hidden-until-lock, so
    # publish the prediction to the public chat ONLY once the week's pick window is
    # CLOSED — reusing the pure compute_window/is_pick_open predicates (no
    # re-implemented window math). While the window is still OPEN, skip the publish
    # entirely (lossy is acceptable; a grade during an open window is rare). The
    # verdict word is derived from the graded PickResult; publish_event swallows
    # Redis errors so it can never break the grade response.
    if _week_window_closed(session, season, week):
        verdict = "correct" if pick.result is PickResult.WIN else "incorrect"
        publish_event(
            misc_graded_event(
                actor=_target_display_name(session, user_id),
                week=week,
                prediction=pick.misc_text,
                verdict=verdict,
                points=pick.points,
                grader=admin.display_name,
            )
        )
    return PickRead.from_orm_pick(pick)


@router.delete("/users/{user_id}/picks", status_code=204)
def clear_user_pick(
    user_id: int,
    season: int,
    week: int,
    pick_type: PickType,
    is_mortal_lock: bool = False,
    admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> None:
    """Clear the PATH user's ``{pick_type, lock}`` slot (window/lock bypassed)."""
    assert admin.id is not None
    # Resolve the target display_name BEFORE the commit (the row is loaded now);
    # after clearing, publish post-commit best-effort. The slot label is the
    # pick_type value (the row is deleted, so there is no game/side to resolve).
    target_display_name = _target_display_name(session, user_id)
    admin_clear_pick(
        session,
        caller_id=admin.id,
        target_user_id=user_id,
        season=season,
        week=week,
        pick_type=pick_type,
        is_mortal_lock=is_mortal_lock,
    )
    session.commit()
    publish_event(
        admin_pick_cleared_event(
            target=target_display_name,
            week=week,
            slot=pick_type.value,
        )
    )
    return None


# --------------------------------------------------------------------------- #
# Admin worker triggers (QT-2)
#
# POST /api/admin/ingest-season and POST /api/admin/freeze-week let an admin (a)
# bootstrap a season's Week+Game skeleton and (b) lock a week's DraftKings lines
# on demand before they vanish. Both DISPATCH the existing/new Celery tasks via
# ``.delay(...)`` and return 202 with the AsyncResult id — they do NOT run the
# (~18 synchronous ESPN-call) ingest in the request thread. The gated source
# resolution lives in the task wrappers (:mod:`app.tasks`), never here.
#
# Both sit behind Depends(require_admin) (401 anon / 403 non-admin) exactly like
# every other /api/admin route, and there is NO actor field in the body, so there
# is no privilege-escalation / IDOR surface (T-h2v-01). They are mutating POSTs,
# so the existing double-submit CSRF middleware applies to cookie auth exactly as
# the other admin POSTs (bearer is exempt; no special-casing — T-h2v-02).
# --------------------------------------------------------------------------- #


@router.post("/ingest-season", status_code=202)
def trigger_ingest_season(
    payload: IngestSeasonRequest,
    admin: User = Depends(require_admin),
) -> dict:
    """Dispatch a season-bootstrap ingest (admin only). 202 + the task id."""
    result = ingest_season_task.delay(payload.season)
    return {"task_id": result.id, "season": payload.season}


@router.post("/freeze-week", status_code=202)
def trigger_freeze_week(
    payload: FreezeWeekRequest,
    admin: User = Depends(require_admin),
) -> dict:
    """Dispatch a manual line-freeze for one week (admin only). 202 + the task id."""
    result = freeze_week_task.delay(payload.season, payload.week)
    return {
        "task_id": result.id,
        "season": payload.season,
        "week": payload.week,
    }


# --------------------------------------------------------------------------- #
# Bot personality (260627-xbb)
#
# GET/POST /api/admin/bot-personality read/set the app-wide LLM chat voice. Both
# sit behind Depends(require_admin) (401 anon / 403 non-admin) exactly like every
# other /api/admin route; there is NO actor field in the body, so there is no
# privilege-escalation / IDOR surface (T-xbb-01). The POST is a mutating request,
# so the existing double-submit CSRF middleware applies to cookie auth as usual
# (bearer is exempt; no special-casing). The service validates the id against the
# personality registry and raises ``ValueError("unknown_personality: ...")`` on a
# miss, which ``_raise_for_service_error`` maps to 409 (T-xbb-02).
# --------------------------------------------------------------------------- #


@router.get("/bot-personality", response_model=BotPersonalityRead)
def get_bot_personality_setting(
    admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> BotPersonalityRead:
    """Return the active bot personality id + the registry's selectable ids."""
    return BotPersonalityRead(
        active_id=get_bot_personality(session),
        available_ids=available_personality_ids(),
    )


@router.post("/bot-personality", response_model=BotPersonalityRead)
def set_bot_personality_setting(
    payload: SetBotPersonalityRequest,
    admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> BotPersonalityRead:
    """Set the active bot personality. 409 on an unknown id (unknown_personality)."""
    try:
        active = set_bot_personality(session, payload.personality_id)
    except ValueError as exc:
        _raise_for_service_error(exc)
        raise  # unreachable; satisfies the type checker
    return BotPersonalityRead(
        active_id=active,
        available_ids=available_personality_ids(),
    )
