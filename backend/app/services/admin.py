"""user_id-keyed admin user-management service (the web admin surface, QT-C).

This is the web counterpart to the discord_id-keyed bot management functions in
:mod:`app.services.auth` (``deactivate_user_by_discord_id`` etc.). The bot keys
every action by the Discord snowflake; the web admin keys by the surrogate
``User.id`` so it can also manage the NULL-discord_id bootstrap-seed admin (QT-B),
which is unreachable by the discord_id path by design.

Guards (locked decisions in .planning/notes/admin-area-design.md):

* **self-guard** — an admin may NOT revoke-admin / delete / deactivate their OWN
  account (checked BEFORE the DB lookup, mirroring ``revoke_admin_by_discord_id``).
  Reactivate and grant-admin have NO self-guard (they cannot lock anyone out).
* **last-admin guard** — no action may leave the system with zero admins:
  - revoke-admin / delete count ALL ``is_admin=True`` rows (regardless of
    is_active) and reject when the target is the only one.
  - deactivate counts ACTIVE admins (``is_admin=True AND is_active=True``) and
    rejects when the target is the only active admin (a deactivated admin can no
    longer log in to manage the system, so "active" is the right denominator).

Every rejection raises ``ValueError`` whose FIRST whitespace-delimited token is a
STABLE machine code (``cannot_act_on_self``, ``user_not_found``, ``already_inactive``,
``already_active``, ``already_admin``, ``not_admin``, ``last_admin``) so the router
(:mod:`app.api.admin`) can split it off and map it to a typed exception with a
``reason=`` field, exactly the way the bot reuses the human message.

The caller id is a PARAMETER only — the service never reads request state, so the
acting identity always comes from the verified session at the router (no IDOR).

This module does NOT touch :mod:`app.services.auth` and adds no ``IS_DEMO_DATA``
branch. ``delete_user`` relies on the QT-A ``ON DELETE CASCADE`` on
``Pick.user_id`` to remove the user's picks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import structlog
from sqlalchemy import func
from sqlmodel import Session, select

from app.models import Pick, User

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AdminUserRow:
    """A flat, password_hash-free view of a user for the admin list/mutation rows.

    Deliberately NOT an ORM ``User``: it carries only the public-to-admin fields
    plus a derived ``pick_count`` and can never leak ``password_hash``. The router
    shapes this into the ``AdminUserRead`` response schema.
    """

    id: int
    display_name: str
    discord_id: int | None
    discord_avatar_hash: str | None
    is_admin: bool
    is_active: bool
    created_at: datetime
    pick_count: int


def _pick_counts(session: Session) -> dict[int, int]:
    """Return ``{user_id: pick_count}`` via ONE grouped query (no N+1).

    Users with zero picks are simply absent from the map; callers default to 0.
    """
    rows = session.exec(
        select(Pick.user_id, func.count(Pick.id)).group_by(Pick.user_id)
    ).all()
    return {user_id: count for user_id, count in rows}


def _row_for(session: Session, user: User, counts: dict[int, int] | None = None) -> AdminUserRow:
    """Build an :class:`AdminUserRow` for one user (pick_count from ``counts``)."""
    assert user.id is not None
    if counts is None:
        counts = _pick_counts(session)
    return AdminUserRow(
        id=user.id,
        display_name=user.display_name,
        discord_id=user.discord_id,
        discord_avatar_hash=user.discord_avatar_hash,
        is_admin=user.is_admin,
        is_active=user.is_active,
        created_at=user.created_at,
        pick_count=counts.get(user.id, 0),
    )


def list_users(session: Session) -> list[AdminUserRow]:
    """Return every user as a flat, password_hash-free row with pick_count.

    One grouped count query feeds every row's ``pick_count`` (no N+1); users with
    no picks report 0. Ordered by ``id`` for a stable listing.
    """
    counts = _pick_counts(session)
    users = session.exec(select(User).order_by(User.id)).all()
    return [_row_for(session, u, counts) for u in users]


def _get_user_or_raise(session: Session, user_id: int) -> User:
    user: User | None = session.get(User, user_id)
    if user is None:
        raise ValueError(f"user_not_found: no account found for user_id {user_id}")
    return user


def _count_admins(session: Session, *, active_only: bool = False) -> int:
    stmt = select(func.count(User.id)).where(User.is_admin == True)  # noqa: E712
    if active_only:
        stmt = stmt.where(User.is_active == True)  # noqa: E712
    return session.exec(stmt).one()


def deactivate_user(session: Session, caller_id: int, user_id: int) -> AdminUserRow:
    """Deactivate (soft-disable) another user's account.

    Self-guard (rejects caller acting on self), missing-user, already-inactive,
    and the ACTIVE-admin last-admin guard all raise ``ValueError`` with a stable
    leading code. On success sets ``is_active=False`` and returns the updated row.
    """
    if caller_id == user_id:
        raise ValueError("cannot_act_on_self: cannot deactivate your own account")
    user = _get_user_or_raise(session, user_id)
    if not user.is_active:
        raise ValueError("already_inactive: account is already deactivated")
    # Last-admin guard counts ACTIVE admins: deactivating the only active admin
    # would leave nobody able to log in and manage the system.
    if user.is_admin and _count_admins(session, active_only=True) == 1:
        raise ValueError("last_admin: cannot deactivate the only remaining active admin")

    user.is_active = False
    session.add(user)
    session.commit()
    session.refresh(user)
    logger.info("admin_user_deactivated", user_id=user_id, caller_id=caller_id)
    return _row_for(session, user)


def reactivate_user(session: Session, user_id: int) -> AdminUserRow:
    """Reactivate a previously deactivated account. No self/last-admin guard."""
    user = _get_user_or_raise(session, user_id)
    if user.is_active:
        raise ValueError("already_active: account is already active")

    user.is_active = True
    session.add(user)
    session.commit()
    session.refresh(user)
    logger.info("admin_user_reactivated", user_id=user_id)
    return _row_for(session, user)


def grant_admin(session: Session, user_id: int) -> AdminUserRow:
    """Grant admin privileges to a user. No self/last-admin guard (cannot lock out)."""
    user = _get_user_or_raise(session, user_id)
    if user.is_admin:
        raise ValueError("already_admin: user is already an admin")

    user.is_admin = True
    session.add(user)
    session.commit()
    session.refresh(user)
    logger.info("admin_granted", user_id=user_id)
    return _row_for(session, user)


def revoke_admin(session: Session, caller_id: int, user_id: int) -> AdminUserRow:
    """Revoke admin privileges from another user.

    Self-guard, missing-user, target-not-admin, and the last-admin guard (counts
    ALL ``is_admin=True`` regardless of is_active) all raise ``ValueError`` with a
    stable leading code. On success sets ``is_admin=False`` and returns the row.
    """
    if caller_id == user_id:
        raise ValueError("cannot_act_on_self: cannot remove your own admin access")
    user = _get_user_or_raise(session, user_id)
    if not user.is_admin:
        raise ValueError("not_admin: target user is not an admin")
    if _count_admins(session) == 1:
        raise ValueError("last_admin: cannot remove the only remaining admin")

    user.is_admin = False
    session.add(user)
    session.commit()
    session.refresh(user)
    logger.info("admin_revoked", user_id=user_id, caller_id=caller_id)
    return _row_for(session, user)


def delete_user(session: Session, caller_id: int, user_id: int) -> None:
    """Hard-delete a user; their picks cascade via the QT-A ON DELETE CASCADE.

    Self-guard, missing-user, and the last-admin guard (when the target is an
    admin and is the only ``is_admin=True`` row) all raise ``ValueError`` with a
    stable leading code. On success ``session.delete(user)`` removes the row and
    the DB cascade removes that user's picks; commits.
    """
    if caller_id == user_id:
        raise ValueError("cannot_act_on_self: cannot delete your own account")
    user = _get_user_or_raise(session, user_id)
    if user.is_admin and _count_admins(session) == 1:
        raise ValueError("last_admin: cannot delete the only remaining admin")

    session.delete(user)
    session.commit()
    logger.info("admin_user_deleted", user_id=user_id, caller_id=caller_id)
    return None
