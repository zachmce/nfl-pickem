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

from app.schemas.types import DiscordId
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
    discord_id: DiscordId
    # Discord avatar hash (mirrors AdminUserRow). None for web-origin / seeded
    # accounts or Discord users without a custom avatar; the admin table builds
    # the CDN avatar URL from this and falls back to initials.
    discord_avatar_hash: str | None
    is_admin: bool
    is_active: bool
    # The break-glass marker (the bootstrap admin). True only for the protected
    # account whose Deactivate / Revoke admin / Delete controls the UI disables to
    # mirror the server-side "protected" guard. extra="forbid" requires this field
    # to be listed for the new AdminUserRow attribute to surface.
    is_protected: bool
    created_at: datetime
    pick_count: int

    @classmethod
    def from_row(cls, row: AdminUserRow) -> "AdminUserRead":
        """Shape one service result row into the response model."""
        return cls(
            id=row.id,
            display_name=row.display_name,
            discord_id=row.discord_id,
            discord_avatar_hash=row.discord_avatar_hash,
            is_admin=row.is_admin,
            is_active=row.is_active,
            is_protected=row.is_protected,
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


# --------------------------------------------------------------------------- #
# Admin trigger requests (QT-2)
#
# Request bodies for the two admin-only worker triggers under /api/admin:
# ``POST /ingest-season`` (bootstrap a season's Week+Game skeleton) and
# ``POST /freeze-week`` (re-snapshot + lock one week's lines NOW). The acting
# admin is the verified session — there is NO actor field in the body, so there
# is no spoofable caller and no IDOR surface. Both endpoints DISPATCH a Celery
# task and return a small 202 body carrying the dispatched task id.
# --------------------------------------------------------------------------- #


class IngestSeasonRequest(BaseModel):
    """Body for ``POST /api/admin/ingest-season``: just the season to ingest."""

    model_config = ConfigDict(extra="forbid")

    season: int


class FreezeWeekRequest(BaseModel):
    """Body for ``POST /api/admin/freeze-week``: the season + week to freeze."""

    model_config = ConfigDict(extra="forbid")

    season: int
    week: int


# --------------------------------------------------------------------------- #
# Bot personality (260627-xbb)
#
# GET /api/admin/bot-personality returns the active id plus the registry's
# available ids (so the admin selector can render the options); POST sets it. The
# acting admin is the verified session — there is NO actor field in the body, so
# there is no spoofable caller / IDOR surface.
# --------------------------------------------------------------------------- #


class BotPersonalityRead(BaseModel):
    """The active bot personality id + the list of selectable ids."""

    model_config = ConfigDict(extra="forbid")

    active_id: str
    available_ids: list[str]


class SetBotPersonalityRequest(BaseModel):
    """Body for ``POST /api/admin/bot-personality``: just the id to make active."""

    model_config = ConfigDict(extra="forbid")

    personality_id: str
