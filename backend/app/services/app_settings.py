"""Keyed app-wide settings service (260627-xbb).

Pure session functions over the :class:`~app.models.AppSetting` keyed table: a
generic ``get_setting`` / ``set_setting`` (upsert) pair plus the bot-personality
wrappers the admin route and the bot use. There is NO request state here — every
function takes an explicit ``Session`` (mirroring :mod:`app.services.admin`), so
the same code runs under the FastAPI request session AND the bot's
``task_session`` thread.

Validation convention mirrors :mod:`app.services.admin`: an invalid personality
id raises ``ValueError`` whose FIRST whitespace-delimited token is the STABLE
machine code ``unknown_personality`` so the router can split it off and map it to
a typed 409 via ``_raise_for_service_error``.

Best-effort default (T-xbb-04): :func:`get_bot_personality` returns the registry
DEFAULT (sarcastic) when the key is unset or blank, so the bot keeps its current
voice on a missing setting.
"""

from __future__ import annotations

import structlog
from sqlmodel import Session, select

from app.bot.personality import (
    DEFAULT_PERSONALITY_ID,
    available_personality_ids,
)
from app.models import AppSetting, _utcnow

logger = structlog.get_logger(__name__)

# The single key under which the active bot personality id is stored.
BOT_PERSONALITY_KEY = "bot_personality"


def get_setting(session: Session, key: str) -> str | None:
    """Return the stored value for ``key``, or ``None`` when no row exists."""
    row = session.exec(
        select(AppSetting).where(AppSetting.setting_key == key)
    ).first()
    return row.setting_value if row is not None else None


def set_setting(session: Session, key: str, value: str) -> str:
    """Upsert ``key`` -> ``value`` (one row per key) and return the stored value.

    Inserts a new row when the key is absent; otherwise updates the existing
    row's value and bumps ``updated_at``. Commits the change.
    """
    row = session.exec(
        select(AppSetting).where(AppSetting.setting_key == key)
    ).first()
    if row is None:
        row = AppSetting(setting_key=key, setting_value=value)
        session.add(row)
    else:
        row.setting_value = value
        row.updated_at = _utcnow()
        session.add(row)
    session.commit()
    return value


def get_bot_personality(session: Session) -> str:
    """Return the active bot personality id, or the sarcastic default when unset.

    Best-effort: an absent OR blank ``bot_personality`` row resolves to
    :data:`app.bot.personality.DEFAULT_PERSONALITY_ID` so the bot keeps its
    current voice when the setting was never chosen.
    """
    value = get_setting(session, BOT_PERSONALITY_KEY)
    if not value or not value.strip():
        return DEFAULT_PERSONALITY_ID
    return value


def set_bot_personality(session: Session, personality_id: str) -> str:
    """Validate + persist the active bot personality id, returning the stored id.

    Rejects an id not in the personality registry with
    ``ValueError("unknown_personality: ...")`` (leading stable code) following the
    :mod:`app.services.admin` convention; on a valid id upserts the
    ``bot_personality`` setting and returns it.
    """
    if personality_id not in available_personality_ids():
        raise ValueError(
            f"unknown_personality: {personality_id!r} is not a known personality"
        )
    return set_setting(session, BOT_PERSONALITY_KEY, personality_id)
