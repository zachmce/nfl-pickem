"""Pydantic response schemas for the web admin user-management API (QT-C).

Mirrors the Pydantic v2 ``BaseModel`` + ``ConfigDict(extra="forbid")`` style of
:mod:`app.schemas.slate` / :mod:`app.schemas.results`: the schemas are built
explicitly from the already-shaped :class:`~app.services.admin.AdminUserRow`
rows via ``from_row`` / ``from_rows`` classmethods rather than coupling to ORM
``User`` rows — so ``password_hash`` can never appear in the response.

Privacy posture: ``AdminUserRead`` is ``extra="forbid"`` and lists every field
explicitly; there is no ``password_hash`` field, and the source rows are the
flat service dataclass (not the ORM model), so the hash cannot leak.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.services.admin import AdminUserRow


class AdminUserRead(BaseModel):
    """One user as seen by an admin: identity, role/state flags, and pick_count.

    Returned both as the list element (GET /api/admin/users) and as the body of
    the four POST mutation routes (the updated target, with a refreshed
    ``pick_count``). Never includes ``password_hash``.
    """

    model_config = ConfigDict(extra="forbid")

    id: int
    display_name: str
    discord_id: int | None
    is_admin: bool
    is_active: bool
    created_at: datetime
    pick_count: int

    @classmethod
    def from_row(cls, row: AdminUserRow) -> "AdminUserRead":
        """Shape one service result row into the response model."""
        return cls(
            id=row.id,
            display_name=row.display_name,
            discord_id=row.discord_id,
            is_admin=row.is_admin,
            is_active=row.is_active,
            created_at=row.created_at,
            pick_count=row.pick_count,
        )


class AdminUserListResponse(BaseModel):
    """The full user list for the admin page: one :class:`AdminUserRead` per user."""

    model_config = ConfigDict(extra="forbid")

    users: list[AdminUserRead]

    @classmethod
    def from_rows(cls, rows: list[AdminUserRow]) -> "AdminUserListResponse":
        """Shape the service's result rows into the list response."""
        return cls(users=[AdminUserRead.from_row(r) for r in rows])
