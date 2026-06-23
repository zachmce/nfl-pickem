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
from app.models import User
from app.schemas.picks import PickRead, PickSubmitRequest
from app.services.pick_submission import read_picks, submit_picks

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
