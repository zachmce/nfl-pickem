"""Pydantic response schema for the current-week context-bar read API.

Mirrors the Pydantic v2 ``BaseModel`` + ``ConfigDict(extra="forbid")`` style of
:mod:`app.schemas.picks` and :mod:`app.schemas.results`. Unlike ``results``, the
shaping here is trivial (four scalar fields, no service value-object mapping), so
there is deliberately NO ``from_*`` classmethod — the router builds the response
directly.

``window_state`` is the four-state pick-window enum the SPA's persistent context
bar renders ("Season 2025 · Week 5 · picks open · closes Sun 1:00pm"). Its values
are the lowercase API-contract strings (not the model-layer UPPERCASE enums), so
the wire payload reads ``"window_state": "open"`` rather than a member name.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class PickWindowState(str, Enum):
    """The four mutually-exclusive states of a week's pick window.

    A ``(str, Enum)`` (mirroring the model enums in :mod:`app.models`) whose
    VALUES are lowercase API-contract strings: serializing the enum yields the
    value, never the member name.
    """

    NOT_YET_OPEN = "not_yet_open"
    OPEN = "open"
    LOCKED = "locked"
    CLOSED = "closed"


class CurrentWeekResponse(BaseModel):
    """The current-week context-bar payload.

    ``window_closes_at`` is the chosen week's first kickoff
    (``compute_window().close_at``), serialized as ISO-8601.
    """

    model_config = ConfigDict(extra="forbid")

    season: int
    week: int
    window_state: PickWindowState
    window_closes_at: datetime
