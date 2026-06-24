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
from app.models import PickType, User
from app.schemas.picks import PickRead, PickSubmitRequest
from app.services.pick_submission import clear_pick, read_picks, submit_picks

router = APIRouter(prefix="/api/picks", tags=["picks"])


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
    return None
