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
    get_game_final_context,
    get_history_pick_keys,
    get_leaders_context,
    get_recap_context,
    get_roster_complete_context,
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
    upsert_avatar_hash_by_discord_id,
)


async def provision_user_async(
    discord_id: int, discord_handle: str, avatar_hash: str | None = None
) -> tuple[int, str, str]:
    """Async wrapper: runs provision_user() in a thread via task_session().

    ``avatar_hash`` is the invoking member's Discord avatar hash (None for a
    default avatar), captured inline so a new account has its hash before the
    first sweep tick. Returns (user_id, display_name, plain_password).
    Raises ValueError if discord_id already has an account.
    """

    def _sync() -> tuple[int, str, str]:
        with task_session() as session:
            result = provision_user(session, discord_id, discord_handle, avatar_hash)
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


async def upsert_avatar_hash_async(discord_id: int, avatar_hash: str | None) -> bool:
    """Async wrapper: runs upsert_avatar_hash_by_discord_id() in a thread.

    Sets (or clears, when avatar_hash is None) the Discord avatar hash on the row
    keyed by discord_id. Returns True if a row matched and was updated, False when
    no account exists for that discord_id (the sweep visits members who may not
    have registered) — never raises on a miss. Plain bool out only; Discord-free.
    """

    def _sync() -> bool:
        with task_session() as session:
            return upsert_avatar_hash_by_discord_id(session, discord_id, avatar_hash)

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
    "season_standings": [], "storylines": []}`` when the season is ambiguous/empty
    (the ``storylines`` key mirrors the populated context shape — 260703-jun). Plain
    dict out only; NO ORM escapes the thread; this module stays Discord-free (no
    business logic here — the season-resolve + shaping live in
    :mod:`app.services.notifications_read`).
    """

    def _sync() -> dict:
        with task_session() as session:
            season = current_season(session)
            if season is None:
                return {
                    "week": week,
                    "weekly_scores": [],
                    "season_standings": [],
                    "storylines": [],
                }
            return get_recap_context(session, season, week)

    return await asyncio.to_thread(_sync)


async def get_game_final_context_async(week: int, away_abbr: str, home_abbr: str) -> dict:
    """Async wrapper: the display-only game.final context as a plain dict.

    Same posture as :func:`get_recap_context_async`: resolves the active season
    via ``current_season`` then runs ``get_game_final_context`` inside a thread
    over ``task_session()``. Returns a not-found-shaped dict (``found`` False, line
    results ``None``, impacts ``[]``) when the season is ambiguous/empty so the
    caller falls back. Plain dict out only; NO ORM escapes the thread; Discord-free
    (the season-resolve + shaping live in
    :mod:`app.services.notifications_read`).
    """

    def _sync() -> dict:
        with task_session() as session:
            season = current_season(session)
            if season is None:
                return {
                    "found": False,
                    "away": None,
                    "home": None,
                    "away_score": None,
                    "home_score": None,
                    "spread_result": None,
                    "total_result": None,
                    "pick_impacts": [],
                    "narrative": {},
                }
            return get_game_final_context(
                session, season, week, away_abbr=away_abbr, home_abbr=home_abbr
            )

    return await asyncio.to_thread(_sync)


async def get_roster_complete_context_async(week: int, actor: str) -> dict:
    """Async wrapper: the display-only roster.complete context as a plain dict.

    Same posture as :func:`get_recap_context_async`: resolves the active season
    then runs ``get_roster_complete_context`` inside a thread over
    ``task_session()``. Returns a safe empty-shaped dict (counts zeroed, ``rank``
    ``None``) on an ambiguous/empty season. COUNT-only by construction — the
    builder never returns outstanding names or pick content. Plain dict out only;
    Discord-free.
    """

    def _sync() -> dict:
        with task_session() as session:
            season = current_season(session)
            if season is None:
                return {
                    "actor": actor,
                    "rank": None,
                    "season_total": 0,
                    "completed_count": 0,
                    "total_players": 0,
                    "outstanding_count": 0,
                }
            return get_roster_complete_context(session, season, week, actor=actor)

    return await asyncio.to_thread(_sync)


async def get_bot_personality_async() -> str:
    """Async wrapper: the active bot personality id as a plain str.

    Resolves the active personality inside a thread over ``task_session()`` (the
    same async/thread seam every other DB read uses). Best-effort: returns the
    sarcastic DEFAULT on an unset setting OR any read failure, so the bot keeps its
    current voice and NEVER raises into the notifier loop. Plain str out only;
    Discord-free.
    """
    from app.bot.personality import DEFAULT_PERSONALITY_ID
    from app.services.app_settings import get_bot_personality

    def _sync() -> str:
        with task_session() as session:
            return get_bot_personality(session)

    try:
        return await asyncio.to_thread(_sync)
    except Exception:  # pragma: no cover - defensive best-effort
        return DEFAULT_PERSONALITY_ID


async def resolve_active_voice_async() -> str:
    """Async wrapper: the active personality's VOICE PREAMBLE as a plain str.

    Resolves the active personality id via :func:`get_bot_personality_async` then
    maps it to its voice preamble through the registry. Best-effort: any miss /
    unknown id / read failure falls back to the sarcastic voice (the registry's
    ``voice_for`` already defaults unknown ids, and the id resolver already
    defaults on a DB miss). This is the ONLY place the active voice is read for the
    prompt builders — the pure ``llm_client.phrase`` layer never touches the DB.
    """
    from app.bot.personality import voice_for

    personality_id = await get_bot_personality_async()
    return voice_for(personality_id)


async def get_leaders_context_async() -> dict:
    """Async wrapper: the display-only season-leaders context as a plain dict.

    Same posture as :func:`get_recap_context_async`: resolves the active season
    then runs ``get_leaders_context`` inside a thread over ``task_session()``.
    Returns a safe empty-shaped dict (leader ``None``) on an ambiguous/empty
    season. Plain dict out only; Discord-free.
    """

    def _sync() -> dict:
        with task_session() as session:
            season = current_season(session)
            if season is None:
                return {
                    "leader": None,
                    "leader_total": 0,
                    "runner_up": None,
                    "runner_up_total": None,
                    "gap": None,
                }
            return get_leaders_context(session, season)

    return await asyncio.to_thread(_sync)
