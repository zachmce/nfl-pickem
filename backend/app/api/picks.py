"""Authenticated pick submission + read endpoints.

Thin router mirroring :mod:`app.api.auth` / :mod:`app.api.proof`: it resolves the
acting user from the verified session (NEVER the body), calls the pick-submission
service, and shapes the response. All legality enforcement and the
ViolationCode -> envelope mapping live in :mod:`app.services.pick_submission`; the
router only translates HTTP <-> service and lets the global exception handler emit
the ``{"error": {"code", "message", "reason"}}`` envelope for every rejection.

Security:

* ``user_id`` is always ``user.id`` from :func:`app.api.deps.get_current_user`
  (signed session cookie or bearer) — never from the request body, and there is
  no user parameter on either endpoint, so a user can only read/write their own
  picks (no IDOR). Unauthenticated requests are rejected 401 by the dependency.
* The mutating POST works WITH the existing double-submit CSRF middleware exactly
  as :func:`app.api.proof.echo` does (cookie auth requires ``X-CSRF-Token``;
  bearer is exempt) — there is no CSRF special-casing here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.api.deps import get_current_user
from app.db import get_session
from app.models import Game, Pick, PickType, Team, User
from app.schemas.picks import PickRead, PickSubmitRequest
from app.services.notifications import (
    COOLDOWN_TTL_SECONDS,
    claim_cooldown,
    misc_picked_event,
    pick_cleared_event,
    pick_event,
    pick_log_detail,
    publish_event,
    roster_complete_event,
)
from app.services.pick_submission import (
    clear_pick,
    main_picks_complete,
    read_picks,
    submit_picks,
)

router = APIRouter(prefix="/api/picks", tags=["picks"])


def _abbr(session: Session, team_id: int | None) -> str | None:
    """Resolve a team id to its abbreviation (display data), or None.

    A tiny read seam used to build the resolved side/team ``detail`` for the
    pickem-logger events. Returns None for a missing/absent id (e.g. a true
    pick'em has no favorite/underdog), which ``pick_log_detail`` tolerates.
    """
    if team_id is None:
        return None
    team = session.get(Team, team_id)
    return team.abbreviation if team is not None else None


def _resolve_pick_detail(session: Session, pick: Pick) -> str:
    """Build the resolved side/team ``detail`` for one persisted pick.

    Loads the pick's Game and its four team abbreviations and delegates to the
    pure :func:`app.services.notifications.pick_log_detail` (which owns the
    favorite/underdog/over/under/MISC + mortal-lock mapping). Display data only.
    """
    game = session.get(Game, pick.game_id)
    if game is None:
        # Should not happen for a just-persisted pick, but never let logging break
        # the request — fall back to the bare pick-type label.
        return pick.pick_type.value
    return pick_log_detail(
        pick.pick_type,
        pick.is_mortal_lock,
        pick.misc_text,
        favorite_abbr=_abbr(session, game.favorite_team_id),
        underdog_abbr=_abbr(session, game.underdog_team_id),
        home_abbr=_abbr(session, game.home_team_id),
        away_abbr=_abbr(session, game.away_team_id),
    )


@router.post("", response_model=list[PickRead])
def submit(
    payload: PickSubmitRequest,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> list[PickRead]:
    """Submit one or more picks for ``{season, week}`` as the current user.

    Window/lock/conflict rejections surface as a structured 4xx envelope (409 for
    window/lock/conflict/roster, 422 for pick'em-spread-ineligible) via the typed
    exceptions the service raises — never a raw 500. On success the picks are
    committed and returned.
    """
    assert user.id is not None  # an authenticated user always has an id
    picks = submit_picks(
        session,
        user_id=user.id,
        season=payload.season,
        week=payload.week,
        items=payload.picks,
    )
    session.commit()
    for pick in picks:
        session.refresh(pick)

    # Post-commit, best-effort pickem-logger publish (one per resulting pick) —
    # AFTER the commit succeeds (caller-owns-commit, mirrors the QT-1 login site).
    # publish_event swallows any Redis error, so a logging hiccup can never break
    # the submit. submit_picks upserts a slot in place, so we cannot cleanly tell
    # an insert from an update from the returned rows — default to ``pick.changed``
    # for the upsert (documented plan choice).
    for pick in picks:
        publish_event(
            pick_event(
                "pick.changed",
                actor=user.display_name,
                week=payload.week,
                detail=_resolve_pick_detail(session, pick),
            )
        )

        # QT-3 pickem-CHAT (260628-itg): when THIS submit set/updated a MISC
        # prediction, fire ONE leak-safe misc.picked line announcing only THAT
        # the player made their misc call — never the misc_text. Gated behind a
        # ~5-min cooldown per (user, season, week) so repeated edits to the
        # prediction within the window do not re-spam the chat feed. Done AFTER
        # the pick.changed publish so the logger event is unaffected; the
        # cooldown is fail-open and publish_event swallows, so it stays
        # post-commit + best-effort.
        if pick.pick_type is PickType.MISC:
            misc_key = (
                f"pickem:misc_picked_cd:{payload.season}:{payload.week}:{user.id}"
            )
            if claim_cooldown(misc_key, COOLDOWN_TTL_SECONDS):
                publish_event(
                    misc_picked_event(actor=user.display_name, week=payload.week)
                )

    # QT-3 pickem-CHAT: when THIS submit results in the user holding their full
    # standard card for the week — all four base bet types plus a mortal lock —
    # fire ONE roster.complete (display_name only) to the chat feed — post-commit
    # + best-effort (publish_event swallows). Adding the mortal lock can now be
    # the submit that completes the card (no longer a no-op for completion). Gated
    # behind a ~5-min cooldown per (user, season, week) so an immediate second
    # completing submit (e.g. re-setting a slot while the card stays complete)
    # does NOT re-post the roster line (260628-itg). A submit that leaves the card
    # incomplete fires none; the cooldown is fail-open so a Redis hiccup can never
    # suppress the milestone.
    if main_picks_complete(
        session, user_id=user.id, season=payload.season, week=payload.week
    ):
        roster_key = (
            f"pickem:roster_complete_cd:{payload.season}:{payload.week}:{user.id}"
        )
        if claim_cooldown(roster_key, COOLDOWN_TTL_SECONDS):
            publish_event(
                roster_complete_event(actor=user.display_name, week=payload.week)
            )
    return [PickRead.from_orm_pick(p) for p in picks]


@router.get("", response_model=list[PickRead])
def read(
    season: int,
    week: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> list[PickRead]:
    """Read the current user's own picks for ``{season, week}``.

    Scoped to the session user by construction — no user parameter is accepted,
    so reading another user's picks is impossible.
    """
    assert user.id is not None
    picks = read_picks(session, user_id=user.id, season=season, week=week)
    return [PickRead.from_orm_pick(p) for p in picks]


@router.delete("", status_code=204)
def clear(
    season: int,
    week: int,
    pick_type: PickType,
    is_mortal_lock: bool = False,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> None:
    """Clear (delete) the current user's pick for one ``{season, week, pick_type,
    is_mortal_lock}`` slot — the un-pick counterpart of POST.

    Security / user-scoping: the acting user is always ``user.id`` from
    :func:`app.api.deps.get_current_user` (signed session cookie or bearer) — there
    is NO user parameter in the path/query/body, and :func:`clear_pick` filters by
    ``user_id``, so a user can only ever delete their OWN pick (no IDOR).
    Unauthenticated requests are rejected 401 by the dependency.

    Enforcement: the SAME pick-window / per-game-lock rules as POST apply, delegated
    entirely to :func:`clear_pick` (no duplicated window math). A closed window or a
    kicked-off game surfaces as a 409 envelope; a missing slot as a 404
    (``pick_not_found``) — and nothing is deleted on any rejection.

    Shape choice (D-01): the slot identifiers are QUERY PARAMETERS, not a request
    body. DELETE-with-body is fragile across the httpx TestClient / proxies and
    FastAPI treats a DELETE body as non-idiomatic, whereas query params are the
    idiomatic REST shape for a DELETE and keep the auth/CSRF surface identical to the
    other endpoints. The slot fields mirror the POST contract exactly (``PickType``
    enum value for ``pick_type``, bool for ``is_mortal_lock``; ``is_mortal_lock``
    defaults to ``False`` so clearing a base slot needs only season/week/pick_type).

    The mutating DELETE works WITH the existing double-submit CSRF middleware exactly
    as POST does (cookie auth requires ``X-CSRF-Token``; bearer is exempt) — there is
    no CSRF special-casing here. On success FastAPI emits 204 No Content (no body).
    """
    assert user.id is not None  # an authenticated user always has an id
    clear_pick(
        session,
        user_id=user.id,
        season=season,
        week=week,
        pick_type=pick_type,
        is_mortal_lock=is_mortal_lock,
    )
    session.commit()

    # Post-commit, best-effort pickem-logger publish. The row is already deleted,
    # so we have no Game to resolve a team abbreviation against — name the cleared
    # SLOT from the pick_type/lock the caller passed (pick_log_detail falls back to
    # the bare side label, e.g. "OVER" / "Favorite", when abbrs are absent).
    detail = pick_log_detail(
        pick_type,
        is_mortal_lock,
        None,
        favorite_abbr=None,
        underdog_abbr=None,
        home_abbr=None,
        away_abbr=None,
    )
    publish_event(
        pick_cleared_event(actor=user.display_name, week=week, detail=detail)
    )
    return None
