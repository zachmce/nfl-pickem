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

# Anti-abuse request bounds (T-nus-03) — reject overlong / oversized input as a
# 422 at validation, before any DB write, instead of a Postgres 500.
#
# MISC_TEXT_MAX mirrors the VARCHAR(280) Pick.misc_text column. PICKS_BATCH_MAX is
# a generous-but-finite cap: a real week has only a single-digit number of pick
# slots (the base types + one mortal lock + MISC), so 64 comfortably exceeds any
# legitimate submission yet rejects an abusive 10k-item batch. MISC_POINTS_* is an
# anti-abuse guard on the admin MISC grade — scoring intentionally does NOT clamp
# MISC points (any admin-set int is a legitimate grade), so this range is chosen
# generously enough to cover every real grade while rejecting absurd values; it is
# NOT a scoring rule. These live here so the schemas and their tests share one
# source of truth.
MISC_TEXT_MAX = 280
PICKS_BATCH_MAX = 64
MISC_POINTS_MIN = -100
MISC_POINTS_MAX = 100


class PickItem(BaseModel):
    """A single incoming pick on one game.

    ``user_id`` is deliberately ABSENT — the acting user is always derived from
    the session by the router, never from the request body (no IDOR).
    """

    model_config = ConfigDict(extra="forbid")

    game_id: int
    pick_type: PickType
    is_mortal_lock: bool = False
    # Free-text prediction — REQUIRED for a MISC pick, rejected for any other type
    # (enforced in ``app.services.pick_submission.submit_picks``). Capped at the
    # VARCHAR(280) column width so overlong text is a 422, not a DB error.
    misc_text: str | None = Field(default=None, max_length=MISC_TEXT_MAX)


class PickSubmitRequest(BaseModel):
    """Submit one or more picks for a single ``{season, week}``."""

    model_config = ConfigDict(extra="forbid")

    season: int
    week: int
    # Non-empty: at least one pick item. A single pick is a list of length 1. The
    # max_length cap is an anti-abuse bound — a real week has only a single-digit
    # number of slots, so PICKS_BATCH_MAX (64) comfortably exceeds any legitimate
    # submission while rejecting an abusive oversized batch as a 422.
    picks: list[PickItem] = Field(min_length=1, max_length=PICKS_BATCH_MAX)


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
    # The owner always sees their OWN misc_text (NULL for non-MISC picks).
    misc_text: str | None = None
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
            misc_text=pick.misc_text,
            result=pick.result,
            points=pick.points,
        )
