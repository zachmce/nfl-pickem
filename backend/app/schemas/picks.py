"""Pydantic request/response models for the pick submission API.

Mirrors the Pydantic v2 ``BaseModel`` style used in :mod:`app.schemas.auth`. The
incoming shape is ONE clean list shape: a submit request carries a ``{season,
week}`` plus a non-empty list of pick items (the single-pick case is just a list
of length 1 — no second singular-vs-list shape). Read items are built explicitly
from ORM ``Pick`` rows and never expose the owning ``User``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models import Pick, PickResult, PickType


class PickItem(BaseModel):
    """A single incoming pick on one game.

    ``user_id`` is deliberately ABSENT — the acting user is always derived from
    the session by the router, never from the request body (no IDOR).
    """

    model_config = ConfigDict(extra="forbid")

    game_id: int
    pick_type: PickType
    is_mortal_lock: bool = False


class PickSubmitRequest(BaseModel):
    """Submit one or more picks for a single ``{season, week}``."""

    model_config = ConfigDict(extra="forbid")

    season: int
    week: int
    # Non-empty: at least one pick item. A single pick is a list of length 1.
    picks: list[PickItem] = Field(min_length=1)


class PickRead(BaseModel):
    """A persisted pick as returned to its owner.

    Built explicitly from an ORM ``Pick`` via :meth:`from_orm_pick`; never
    exposes ``user_id`` or the ``User`` relationship.
    """

    model_config = ConfigDict(extra="forbid")

    id: int
    game_id: int
    week_id: int
    pick_type: PickType
    is_mortal_lock: bool
    result: PickResult
    points: int

    @classmethod
    def from_orm_pick(cls, pick: Pick) -> "PickRead":
        """Build a read item from an ORM ``Pick`` row (explicit field copy)."""
        assert pick.id is not None  # a persisted/flushed pick always has an id
        return cls(
            id=pick.id,
            game_id=pick.game_id,
            week_id=pick.week_id,
            pick_type=pick.pick_type,
            is_mortal_lock=pick.is_mortal_lock,
            result=pick.result,
            points=pick.points,
        )
