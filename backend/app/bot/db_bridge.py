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
from datetime import datetime

from app.db import task_session
from app.services.notifications import player_registered_event, publish_event
from app.services.notifications_read import (
    current_season,
    get_current_week_event_id_for_team,
    get_current_week_weather_target_for_team,
    get_game_final_context,
    get_history_pick_keys,
    get_leaders_context,
    get_lines_slate,
    get_pick_status_for_user,
    get_real_team_tokens,
    get_recap_context,
    get_roster_complete_context,
    get_week_pick_keys,
    get_week_recap_context,
    get_week_scores,
    resolve_current_week,
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


async def get_week_recap_context_async(week: int) -> dict:
    """Async wrapper: the display-only week.recap "closing ceremony" context.

    Same posture as :func:`get_recap_context_async`: resolves the active season via
    ``current_season`` then runs
    :func:`app.services.notifications_read.get_week_recap_context` inside a thread
    over ``task_session()``. Returns the safe empty shape
    ``{standings: [], best_call: None, biggest_bust: None, mortal_locks: []}`` on an
    ambiguous/empty season. Display-only by construction (display_name + ints +
    abbrs + a spread STRING — never a user_id). Plain dict out only; NO ORM escapes
    the thread; this module stays Discord-free.
    """

    def _sync() -> dict:
        with task_session() as session:
            season = current_season(session)
            if season is None:
                return {
                    "standings": [],
                    "best_call": None,
                    "biggest_bust": None,
                    "mortal_locks": [],
                }
            return get_week_recap_context(session, season, week)

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


# --------------------------------------------------------------------------- #
# 260709-k5w — inbound @mention Q&A async seams (Path A v1). Same posture as the
# wrappers above (asyncio.to_thread + task_session() + current_season inside the
# worker thread; plain values out; NO discord import). Each resolves the
# season/current-week internally and returns the safe empty shape on an
# ambiguous/empty season.
# --------------------------------------------------------------------------- #


async def get_real_team_tokens_async() -> set[str]:
    """Async wrapper: the real 32-team token set for the Q&A validator.

    Abbreviations + display-name tokens (see
    :func:`app.services.notifications_read.get_real_team_tokens`). Returns an empty
    set on an unseeded DB. Plain set out only; Discord-free.
    """

    def _sync() -> set[str]:
        with task_session() as session:
            return get_real_team_tokens(session)

    return await asyncio.to_thread(_sync)


async def get_pick_status_async(discord_id: int) -> dict:
    """Async wrapper: ASKER-ONLY pick status for ``discord_id`` (leak-safe).

    Resolves the active season + current week internally, then reads ONLY the
    caller's own status via
    :func:`app.services.notifications_read.get_pick_status_for_user` (never another
    user). Returns ``{registered: False}`` on an ambiguous/empty season or an
    unregistered discord_id. Plain dict out only; Discord-free.
    """

    def _sync() -> dict:
        with task_session() as session:
            season = current_season(session)
            if season is None:
                return {"registered": False}
            week = resolve_current_week(session, season)
            if week is None:
                return {"registered": False}
            return get_pick_status_for_user(session, season, week, discord_id=discord_id)

    return await asyncio.to_thread(_sync)


async def get_lines_slate_async(team_abbr: str | None = None) -> dict:
    """Async wrapper: this week's lines/slate, optionally narrowed to one team.

    Resolves the season + current week internally then delegates to
    :func:`app.services.notifications_read.get_lines_slate`. Returns the safe empty
    shape ``{week: None, close_at: None, games: []}`` on an ambiguous/empty season.
    Plain dict out only; Discord-free.
    """

    def _sync() -> dict:
        with task_session() as session:
            season = current_season(session)
            if season is None:
                return {"week": None, "close_at": None, "games": []}
            week = resolve_current_week(session, season)
            if week is None:
                return {"week": None, "close_at": None, "games": []}
            return get_lines_slate(session, season, week, team_abbr=team_abbr)

    return await asyncio.to_thread(_sync)


async def get_week_scores_async() -> dict:
    """Async wrapper: this week's final + in-progress scores.

    Resolves the season + current week internally then delegates to
    :func:`app.services.notifications_read.get_week_scores`. Returns the safe empty
    shape ``{week: None, games: []}`` on an ambiguous/empty season. Plain dict out
    only; Discord-free.
    """

    def _sync() -> dict:
        with task_session() as session:
            season = current_season(session)
            if season is None:
                return {"week": None, "games": []}
            week = resolve_current_week(session, season)
            if week is None:
                return {"week": None, "games": []}
            return get_week_scores(session, season, week)

    return await asyncio.to_thread(_sync)


async def get_injuries_event_id_async(team_abbr: str) -> tuple[int, str] | None:
    """Async seam: resolve ``team_abbr`` to its current-week ``(event_id, abbr)``.

    Same posture as the other Q&A seams (asyncio.to_thread + task_session() +
    current_season + resolve_current_week inside the worker thread). Delegates to
    :func:`app.services.notifications_read.get_current_week_event_id_for_team` and
    returns the ``(espn_event_id, canonical_abbreviation)`` tuple, or ``None`` on an
    ambiguous/empty season, an unresolved current week, or a team that does not
    resolve to exactly one game with a stored event id. Plain values out only;
    Discord-free.
    """

    def _sync() -> tuple[int, str] | None:
        with task_session() as session:
            season = current_season(session)
            if season is None:
                return None
            week = resolve_current_week(session, season)
            if week is None:
                return None
            return get_current_week_event_id_for_team(session, season, week, team_abbr=team_abbr)

    return await asyncio.to_thread(_sync)


async def get_weather_target_async(team_abbr: str) -> tuple[str, datetime] | None:
    """Async seam: resolve ``team_abbr`` to its current-week ``(home_abbr, kickoff_at)``.

    Same posture as :func:`get_injuries_event_id_async` (asyncio.to_thread +
    task_session() + current_season + resolve_current_week inside the worker thread).
    Delegates to
    :func:`app.services.notifications_read.get_current_week_weather_target_for_team` and
    returns the ``(home_team_abbreviation, kickoff_at)`` tuple, or ``None`` on an
    ambiguous/empty season, an unresolved current week, or a team that does not resolve
    to exactly one game with a known home abbr + kickoff. Plain values out only;
    Discord-free.
    """

    def _sync() -> tuple[str, datetime] | None:
        with task_session() as session:
            season = current_season(session)
            if season is None:
                return None
            week = resolve_current_week(session, season)
            if week is None:
                return None
            return get_current_week_weather_target_for_team(
                session, season, week, team_abbr=team_abbr
            )

    return await asyncio.to_thread(_sync)
