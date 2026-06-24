"""Unauthenticated client-config READ endpoint.

A thin GET-only router exposing the pre-auth signal the SPA's bare login page
reads to decide whether to render the loud demo banner.

Security — deliberate UNAUTHENTICATED posture
---------------------------------------------

Unlike :mod:`app.api.results` / :mod:`app.api.current_week` (which require
``get_current_user`` and 401 anonymous callers), this endpoint is intentionally
PUBLIC: the login page renders before any session exists, so it must be able to
read the demo flag without auth. The exposure is deliberately limited to two
non-sensitive scalars — the demo boolean + the (public) season number — shaped
by :class:`~app.schemas.config.ConfigResponse` with ``extra="forbid"``. No user
data, no secrets, no other settings cross this boundary.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.config import settings
from app.db import get_session
from app.models import Game
from app.schemas.config import ConfigResponse

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("", response_model=ConfigResponse)
def read_config(session: Session = Depends(get_session)) -> ConfigResponse:
    """Return the unauthenticated client config: ``{is_demo, season}``.

    ``is_demo`` reflects ``settings.is_demo_data``. ``season`` is derived from the
    seeded games the same way :mod:`app.api.current_week` does: the single distinct
    ``Game.season``; if there is not exactly one (zero seeded games), fall back to
    0 rather than raising — the login page must still render its banner decision.
    """
    seasons = {s for s in session.exec(select(Game.season).distinct()).all()}
    season = next(iter(seasons)) if len(seasons) == 1 else 0
    return ConfigResponse(is_demo=settings.is_demo_data, season=season)
