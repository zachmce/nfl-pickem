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

from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.api.deps import require_admin
from app.db import get_session
from app.exceptions import ConflictError, NotFoundError
from app.models import User
from app.schemas.admin import AdminUserListResponse, AdminUserRead
from app.services.admin import (
    AdminUserRow,
    deactivate_user,
    delete_user,
    grant_admin,
    list_users,
    reactivate_user,
    revoke_admin,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


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
