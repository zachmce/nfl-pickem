"""Pydantic response schema for the unauthenticated client-config read API.

Mirrors the Pydantic v2 ``BaseModel`` + ``ConfigDict(extra="forbid")`` style of
:mod:`app.schemas.current_week`. The shaping is trivial (two scalar fields, no
service value-object mapping), so there is deliberately NO ``from_*`` classmethod
— the router builds the response directly.

This is the pre-auth signal the bare login page reads to decide whether to render
the loud demo banner, so the payload is intentionally non-sensitive: only the
demo flag + the (public) season number.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConfigResponse(BaseModel):
    """The unauthenticated client-config payload.

    ``is_demo`` reflects ``settings.is_demo_data``; ``season`` is the single
    distinct seeded ``Game.season`` (or 0 when no games are seeded — the login
    page must still render its banner decision without a 404).
    """

    model_config = ConfigDict(extra="forbid")

    is_demo: bool
    season: int
