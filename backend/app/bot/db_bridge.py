"""Async/sync boundary for the Discord bot — wraps every auth service function
in asyncio.to_thread over a task_session() context manager.

Contract invariants:
  - No discord module import — this module is a thin boundary with zero Discord coupling.
  - task_session() is opened INSIDE the _sync closure (worker thread), never on
    the event loop.
  - Each wrapper returns only plain Python values (tuple / str / None); the ORM
    User never escapes the task_session() scope (prevents DetachedInstanceError).
  - No business logic lives here — all validation / commit / structlog is in
    app.services.auth.
  - task_session() is the non-HTTP context manager (not the FastAPI HTTP dep).
"""

from __future__ import annotations

import asyncio

from app.db import task_session
from app.services.notifications import player_registered_event, publish_event
from app.services.notifications_read import (
    current_season,
    get_history_pick_keys,
    get_recap_context,
    get_week_pick_keys,
)
from app.services.auth import (
    deactivate_user_by_discord_id,
    delete_user_by_id,
    get_account_by_discord_id,
    grant_admin_by_discord_id,
    is_admin_by_discord_id,
    provision_user,
    reactivate_user_by_discord_id,
    reset_password_for_discord,
    revoke_admin_by_discord_id,
)


async def provision_user_async(discord_id: int, discord_handle: str) -> tuple[int, str, str]:
    """Async wrapper: runs provision_user() in a thread via task_session().

    Returns (user_id, display_name, plain_password).
    Raises ValueError if discord_id already has an account.
    """

    def _sync() -> tuple[int, str, str]:
        with task_session() as session:
            result = provision_user(session, discord_id, discord_handle)
            # provision_user COMMITS internally, so this publish is post-commit by
            # construction. Best-effort pickem-logger notice carrying ONLY the
            # returned display_name (result[1]) — NEVER the returned plain_password
            # (result[2]) (HARD RULE T-kvi-01). publish_event hits Redis only (no
            # discord import), so db_bridge stays Discord-free.
            _, display_name, _plain_password = result
            publish_event(player_registered_event(display_name))
            return result

    return await asyncio.to_thread(_sync)


async def reset_password_async(discord_id: int) -> str:
    """Async wrapper: runs reset_password_for_discord() in a thread via task_session().

    Returns plain_password (str) — returned once for the bot to DM.
    Raises ValueError if discord_id has no account or account is deactivated.
    """

    def _sync() -> str:
        with task_session() as session:
            return reset_password_for_discord(session, discord_id)

    return await asyncio.to_thread(_sync)


async def deactivate_user_async(discord_id: int) -> None:
    """Async wrapper: runs deactivate_user_by_discord_id() in a thread via task_session().

    Raises ValueError if discord_id has no account or is already deactivated.
    """

    def _sync() -> None:
        with task_session() as session:
            return deactivate_user_by_discord_id(session, discord_id)

    return await asyncio.to_thread(_sync)


async def reactivate_user_async(discord_id: int) -> None:
    """Async wrapper: runs reactivate_user_by_discord_id() in a thread via task_session().

    Raises ValueError if discord_id has no account or is already active.
    """

    def _sync() -> None:
        with task_session() as session:
            return reactivate_user_by_discord_id(session, discord_id)

    return await asyncio.to_thread(_sync)


async def grant_admin_async(discord_id: int) -> None:
    """Async wrapper: runs grant_admin_by_discord_id() in a thread via task_session().

    Raises ValueError if discord_id has no account or is already an admin.
    """

    def _sync() -> None:
        with task_session() as session:
            return grant_admin_by_discord_id(session, discord_id)

    return await asyncio.to_thread(_sync)


async def revoke_admin_async(caller_discord_id: int, target_discord_id: int) -> None:
    """Async wrapper: runs revoke_admin_by_discord_id() in a thread via task_session().

    Raises ValueError if caller == target (self-demote guard), target has no
    account, or target is not an admin.
    """

    def _sync() -> None:
        with task_session() as session:
            return revoke_admin_by_discord_id(session, caller_discord_id, target_discord_id)

    return await asyncio.to_thread(_sync)


async def get_account_async(discord_id: int) -> str | None:
    """Async wrapper: runs get_account_by_discord_id() in a thread via task_session().

    Returns display_name (str) if an account exists for discord_id, None otherwise.
    Never raises. Only display_name returned, no ORM object escapes the thread.
    """

    def _sync() -> str | None:
        with task_session() as session:
            return get_account_by_discord_id(session, discord_id)

    return await asyncio.to_thread(_sync)


async def is_admin_async(discord_id: int) -> bool:
    """Async wrapper: runs is_admin_by_discord_id() in a thread via task_session().

    Returns True only when a row with that discord_id has is_admin=True.
    Returns False for unknown discord_ids or non-admin rows.
    """

    def _sync() -> bool:
        with task_session() as session:
            return is_admin_by_discord_id(session, discord_id)

    return await asyncio.to_thread(_sync)


async def delete_user_async(user_id: int) -> None:
    """Async wrapper: runs delete_user_by_id() in a thread via task_session().

    Hard-deletes a user row by primary key.
    Raises ValueError if user_id is absent (never silently no-ops on a missing row).
    """

    def _sync() -> None:
        with task_session() as session:
            return delete_user_by_id(session, user_id)

    return await asyncio.to_thread(_sync)


async def get_week_picks_async(week: int) -> dict[str, list[dict]]:
    """Async wrapper: this week's locked pick-keys as a plain {display_name: [keys]} dict.

    Resolves the active season via ``current_season`` then runs
    ``get_week_pick_keys`` inside a thread over ``task_session()``. Returns ``{}``
    when the season is ambiguous/empty. NO ORM escapes the thread; this module
    stays Discord-free (no business logic here — the season-resolve + key
    derivation live in :mod:`app.services.notifications_read`).
    """

    def _sync() -> dict[str, list[dict]]:
        with task_session() as session:
            season = current_season(session)
            if season is None:
                return {}
            return get_week_pick_keys(session, season, week)

    return await asyncio.to_thread(_sync)


async def get_pick_history_async(week: int, weeks_back: int = 6) -> dict[str, list[dict]]:
    """Async wrapper: recent-weeks pick-keys as a plain {display_name: [keys]} dict.

    Same posture as :func:`get_week_picks_async`: resolves the season then runs
    ``get_history_pick_keys`` in a thread over ``task_session()`` (covering ``week``
    and the prior ``weeks_back - 1`` weeks). Returns ``{}`` on an ambiguous/empty
    season. Plain dict out only; Discord-free.
    """

    def _sync() -> dict[str, list[dict]]:
        with task_session() as session:
            season = current_season(session)
            if season is None:
                return {}
            return get_history_pick_keys(session, season, week, weeks_back=weeks_back)

    return await asyncio.to_thread(_sync)


async def get_recap_context_async(week: int) -> dict:
    """Async wrapper: the display-only recap context as a plain dict.

    Same posture as :func:`get_week_picks_async`: resolves the active season via
    ``current_season`` then runs ``get_recap_context`` inside a thread over
    ``task_session()``. Returns ``{"week": week, "weekly_scores": [],
    "season_standings": []}`` when the season is ambiguous/empty. Plain dict out
    only; NO ORM escapes the thread; this module stays Discord-free (no business
    logic here — the season-resolve + shaping live in
    :mod:`app.services.notifications_read`).
    """

    def _sync() -> dict:
        with task_session() as session:
            season = current_season(session)
            if season is None:
                return {"week": week, "weekly_scores": [], "season_standings": []}
            return get_recap_context(session, season, week)

    return await asyncio.to_thread(_sync)
