"""Best-effort Discord event publisher (QT-1 — the load-bearing transport).

This is the backend SIDE of the Discord notification pipe: the FastAPI process
serializes a small event and PUBLISHes it to a shared Redis channel. The bot
process subscribes (see :mod:`app.bot.notifier`), renders, and posts it into a
Discord channel. Backend never imports ``discord``; the bot owns all rendering.

v1 event schema
---------------
Every event is a JSON object::

    {
      "v": 1,                       # schema version (int)
      "type": "<str>",              # e.g. "user.login"
      "targets": ["logger"|"chat"], # which Discord surfaces should render it
      ...display-data-only fields   # e.g. "actor": "<display_name>"
    }

v1 event types
--------------
* ``user.login``        (QT-1) — ``actor``
* ``pick.created``      — ``actor``, ``week``, ``detail``
* ``pick.changed``      — ``actor``, ``week``, ``detail``
* ``pick.cleared``      — ``actor``, ``week``, ``detail``
* ``admin.pick_set``    — ``target``, ``week``, ``detail``
* ``admin.pick_cleared``— ``target``, ``week``, ``slot``
* ``player.registered`` — ``actor`` (display_name ONLY)
* ``ingest.season``     — ``season``, ``weeks``, ``games``, ``failed``
* ``freeze.week``       — ``week``

All seven QT-2 types target ``["logger"]`` only.

QT-3 player-facing pickem-CHAT types (all target ``["chat"]`` only):

* ``roster.complete``   — ``actor``, ``week`` (a user filled all 4 base slots)
* ``window.opened``     — ``week``
* ``window.closed``     — ``week``
* ``game.final``        — ``week``, ``away``, ``home``, ``away_score``, ``home_score``
* ``week.recap``        — ``week``, ``winner``, ``winner_score``, ``leader``,
  ``leader_score`` (week winner + season leader display_name + scores)

The five QT-3 types carry DISPLAY data only (display_name strings, integer
scores, week number, team abbreviations) — never a user_id, password, or token.

HARD RULE: only DISPLAY data is ever published — never passwords, tokens, emails,
session cookies, or any secret. The event crosses a trust boundary into Discord,
so the builders below carry display fields only (T-kd8-02 / T-kvi-01 / T-kvi-02).
In particular ``player_registered_event`` accepts the display_name ONLY and NEVER
the temp/plain password returned alongside it by provisioning.

Publisher contract
------------------
``publish_event`` is BEST-EFFORT: the entire body (client construction + publish)
is wrapped in try/except. On any failure it logs a structlog warning and returns
normally — it MUST NEVER raise. A Redis hiccup must not break a login (T-kd8-01).
The publisher uses the SYNCHRONOUS redis client because it runs inside the request
thread; the async client is the subscriber's concern.
"""

from __future__ import annotations

import json

import structlog

from app.config import settings
from app.models import PickType

logger = structlog.get_logger(__name__)

# The single Redis pub/sub channel for cross-process notification events.
EVENTS_CHANNEL = "pickem:events"


def login_event(display_name: str) -> dict:
    """Build the v1 ``user.login`` event — a pure function, no I/O.

    Returns exactly ``{"v": 1, "type": "user.login", "targets": ["logger"],
    "actor": display_name}``. ``actor`` is the user's DISPLAY name only.
    """
    return {
        "v": 1,
        "type": "user.login",
        "targets": ["logger"],
        "actor": display_name,
    }


# --------------------------------------------------------------------------- #
# QT-2 — granular pickem-logger event builders (pure, no I/O) + side resolver.
#
# Each builder mirrors ``login_event``'s shape: ``{"v": 1, "type": ...,
# "targets": ["logger"], ...display fields}`` and carries DISPLAY data only. The
# bot (``app.bot.notifier._render``) does NO resolution — it only string-joins the
# fields these builders emit, so the resolved side/team ``detail`` is computed
# HERE (at the publish site, which has the open session) via ``pick_log_detail``.
# --------------------------------------------------------------------------- #


def pick_log_detail(
    pick_type: PickType,
    is_mortal_lock: bool,
    misc_text: str | None,
    *,
    favorite_abbr: str | None,
    underdog_abbr: str | None,
    home_abbr: str | None,
    away_abbr: str | None,
) -> str:
    """Resolve a pick into a concise display ``detail`` string — pure, no I/O.

    Maps each :class:`~app.models.PickType` to a finished label, reusing the
    favorite/underdog convention from :mod:`app.services.scoring` (one side is the
    favorite, the other the underdog):

    * ``FAVORITE_COVER`` -> ``"Favorite (KC)"`` (or ``"Favorite"`` if the abbr is
      unknown — a true pick'em has no favorite/underdog side).
    * ``UNDERDOG_COVER`` -> ``"Underdog (LAR)"`` / ``"Underdog"``.
    * ``OVER`` / ``UNDER`` -> ``"OVER LAR@KC"`` / ``"UNDER LAR@KC"`` (away@home
      matchup form; falls back to bare ``"OVER"`` when abbrs are missing).
    * ``MISC`` -> the ``misc_text`` verbatim.

    A mortal-lock slot is annotated with a trailing ``" (ML)"``. Every input is a
    plain value (the call site loads the Game + its four Team abbreviations and
    passes them in), so this stays unit-testable offline.
    """
    if pick_type is PickType.MISC:
        detail = misc_text or "Misc"
    elif pick_type is PickType.FAVORITE_COVER:
        detail = f"Favorite ({favorite_abbr})" if favorite_abbr else "Favorite"
    elif pick_type is PickType.UNDERDOG_COVER:
        detail = f"Underdog ({underdog_abbr})" if underdog_abbr else "Underdog"
    else:
        # OVER / UNDER — the away@home matchup form.
        side = "OVER" if pick_type is PickType.OVER else "UNDER"
        if away_abbr and home_abbr:
            detail = f"{side} {away_abbr}@{home_abbr}"
        else:
            detail = side

    if is_mortal_lock:
        detail = f"{detail} (ML)"
    return detail


def pick_event(type: str, *, actor: str, week: int, detail: str) -> dict:
    """Build a ``pick.created`` / ``pick.changed`` event — pure, no I/O.

    ``type`` is one of ``"pick.created"`` / ``"pick.changed"``; ``actor`` is the
    submitting user's DISPLAY name, ``detail`` the resolved side/team string.
    """
    return {
        "v": 1,
        "type": type,
        "targets": ["logger"],
        "actor": actor,
        "week": week,
        "detail": detail,
    }


def pick_cleared_event(*, actor: str, week: int, detail: str) -> dict:
    """Build a ``pick.cleared`` event — ``actor`` cleared their own slot."""
    return {
        "v": 1,
        "type": "pick.cleared",
        "targets": ["logger"],
        "actor": actor,
        "week": week,
        "detail": detail,
    }


def admin_pick_set_event(*, target: str, week: int, detail: str) -> dict:
    """Build an ``admin.pick_set`` event — an admin set ``target``'s slot.

    ``target`` is the affected user's DISPLAY name (server-resolved from the path
    user, never client free text).
    """
    return {
        "v": 1,
        "type": "admin.pick_set",
        "targets": ["logger"],
        "target": target,
        "week": week,
        "detail": detail,
    }


def admin_pick_cleared_event(*, target: str, week: int, slot: str) -> dict:
    """Build an ``admin.pick_cleared`` event — an admin cleared ``target``'s slot.

    ``slot`` is the cleared pick-type label (e.g. ``"FAVORITE_COVER"``).
    """
    return {
        "v": 1,
        "type": "admin.pick_cleared",
        "targets": ["logger"],
        "target": target,
        "week": week,
        "slot": slot,
    }


def player_registered_event(display_name: str) -> dict:
    """Build a ``player.registered`` event — a new player provisioned.

    HARD RULE (T-kvi-01): carries the DISPLAY name ONLY. The temp/plain password
    returned alongside the display_name at provisioning time NEVER appears here —
    the key set is exactly ``{v, type, targets, actor}``.
    """
    return {
        "v": 1,
        "type": "player.registered",
        "targets": ["logger"],
        "actor": display_name,
    }


def ingest_season_event(*, season: int, weeks: int, games: int, failed: int) -> dict:
    """Build an ``ingest.season`` event from an ingest summary — non-sensitive."""
    return {
        "v": 1,
        "type": "ingest.season",
        "targets": ["logger"],
        "season": season,
        "weeks": weeks,
        "games": games,
        "failed": failed,
    }


def freeze_week_event(week: int) -> dict:
    """Build a ``freeze.week`` event — one week's lines were frozen."""
    return {
        "v": 1,
        "type": "freeze.week",
        "targets": ["logger"],
        "week": week,
    }


# --------------------------------------------------------------------------- #
# QT-3 — player-facing pickem-CHAT event builders (pure, no I/O).
#
# These five milestone/edge events feed the ``pickem-chat`` channel (targets
# ``["chat"]``). They mirror the QT-2 builder shape but carry DISPLAY data ONLY
# — display_name strings, integer scores, a week number, and team abbreviations.
# The chat seam is where a FUTURE local-LLM personality layer will plug in
# (bot-side ``render_chat``); NO LLM/nudge logic lives here. The events never
# carry a user_id, password, token, or any secret (T-llw-01).
# --------------------------------------------------------------------------- #


def roster_complete_event(*, actor: str, week: int) -> dict:
    """Build a ``roster.complete`` event — ``actor`` filled all 4 base slots.

    ``actor`` is the user's DISPLAY name only. Fired once, post-commit, only when
    a submit results in the user holding all four base (non-mortal-lock, non-MISC)
    pick slots for the week.
    """
    return {
        "v": 1,
        "type": "roster.complete",
        "targets": ["chat"],
        "actor": actor,
        "week": week,
    }


def window_opened_event(week: int) -> dict:
    """Build a ``window.opened`` event — ``week``'s pick window just opened."""
    return {
        "v": 1,
        "type": "window.opened",
        "targets": ["chat"],
        "week": week,
    }


def window_closed_event(week: int) -> dict:
    """Build a ``window.closed`` event — ``week``'s pick window just closed."""
    return {
        "v": 1,
        "type": "window.closed",
        "targets": ["chat"],
        "week": week,
    }


def game_final_event(
    *,
    week: int,
    away_abbr: str,
    home_abbr: str,
    away_score: int,
    home_score: int,
) -> dict:
    """Build a ``game.final`` event — one game went FINAL. DISPLAY data only.

    Carries the two team abbreviations and the two integer final scores plus the
    week number — nothing user-identifying.
    """
    return {
        "v": 1,
        "type": "game.final",
        "targets": ["chat"],
        "week": week,
        "away": away_abbr,
        "home": home_abbr,
        "away_score": away_score,
        "home_score": home_score,
    }


def week_recap_event(
    *,
    week: int,
    winner: str,
    winner_score: int,
    leader: str,
    leader_score: int,
) -> dict:
    """Build a ``week.recap`` event — a week's last game just went FINAL.

    ``winner`` is the week winner's DISPLAY name and ``winner_score`` their weekly
    score; ``leader`` is the season leader's DISPLAY name and ``leader_score``
    their season total. Display names + integer scores ONLY — nothing sensitive.
    """
    return {
        "v": 1,
        "type": "week.recap",
        "targets": ["chat"],
        "week": week,
        "winner": winner,
        "winner_score": winner_score,
        "leader": leader,
        "leader_score": leader_score,
    }


def _redis_client():
    """Construct a synchronous redis client from ``settings.redis_url``.

    Isolated as a tiny seam so tests can monkeypatch it without touching a real
    socket, and so the URL is never hardcoded (reuse the celery broker setting).
    """
    import redis

    return redis.Redis.from_url(settings.redis_url)


def publish_event(event: dict) -> None:
    """PUBLISH ``event`` (as JSON) to :data:`EVENTS_CHANNEL` — best-effort.

    Wraps the whole operation (client build + publish) in try/except: on ANY
    failure it logs a structlog warning and returns normally. NEVER raises, so a
    Redis outage can never break the caller (e.g. the login route).
    """
    try:
        client = _redis_client()
        client.publish(EVENTS_CHANNEL, json.dumps(event))
    except Exception:
        logger.warning(
            "notification_publish_failed",
            event_type=event.get("type"),
            channel=EVENTS_CHANNEL,
        )
