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

* ``roster.complete``   — ``actor``, ``week`` (a user filled their full standard
  card — four base bet types + a mortal lock)
* ``window.opened``     — ``week``
* ``window.closed``     — ``week``
* ``game.final``        — ``week``, ``away``, ``home``, ``away_score``, ``home_score``
* ``week.recap``        — ``week``, ``winner``, ``winner_score``, ``leader``,
  ``leader_score`` (week winner + season leader display_name + scores)
* ``misc.graded``       — ``actor``, ``week``, ``prediction``, ``verdict``,
  ``points`` (an admin graded a player's MISC prediction; the player's OWN free
  text + the verdict word + signed points — published ONLY once the week's pick
  window is closed so the prediction is never revealed before lock)
* ``misc.picked``       — ``actor``, ``week`` (a player submitted/set their MISC
  prediction; announces ONLY THAT they made a misc call — LEAK-SAFE: NEVER the
  ``misc_text`` content, NEVER a user_id, since picks are hidden until the window
  closes. Deduped behind a ~5-min :func:`claim_cooldown` window per user/week.)

The QT-3 chat types carry DISPLAY data only (display_name strings, integer
scores, week number, team abbreviations, the player's own prediction text) —
never a user_id, password, or token.

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
from typing import Literal, TypedDict

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
    """Build a ``freeze.week`` event — one week's lines were frozen.

    DUAL-DISPATCHED (260705-jo9): targets BOTH the ops-log (``logger`` — the terse
    "Week N lines frozen" line) AND the chat channel (``chat`` — a LIGHT gold
    "Week N - Lines Locked" card). The freeze is a noteworthy state change worth a
    chat card, but the ops-log line is a useful operational signal we keep too.
    """
    return {
        "v": 1,
        "type": "freeze.week",
        "targets": ["logger", "chat"],
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
    """Build a ``roster.complete`` event — ``actor`` filled their full standard card.

    ``actor`` is the user's DISPLAY name only. Fired once, post-commit, only when
    a submit results in the user holding their full standard card for the week —
    all four base bet types plus a mortal lock.
    """
    return {
        "v": 1,
        "type": "roster.complete",
        "targets": ["chat"],
        "actor": actor,
        "week": week,
    }


def misc_picked_event(*, actor: str, week: int) -> dict:
    """Build a ``misc.picked`` event — ``actor`` submitted/set their MISC pick.

    HARD LEAK RULE (T-itg-01): carries the player's DISPLAY name + the week ONLY.
    It NEVER carries the ``misc_text`` content (predictions are hidden until the
    pick window closes) and NEVER a user_id — the key set is EXACTLY
    ``{v, type, targets, actor, week}``, mirroring :func:`roster_complete_event`.
    The caller fires this once per (user, season, week) cooldown window behind
    :func:`claim_cooldown` so repeated edits to the prediction do not re-spam chat.
    """
    return {
        "v": 1,
        "type": "misc.picked",
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


class GameFinalImpact(TypedDict):
    """One per-user impact on a FINAL game — the ``game.final`` event contract.

    This is the typed shape the Discord embed renderer consumes directly off
    ``event["impacts"]`` (see ``.planning/todos/pending/game-final-embed-render.md``)
    to populate its Busted/Cashed fields. JSON-primitive-only by design: the event
    is ``json.dumps``'d across the Redis pub/sub boundary, so every field is a str
    or bool that survives a round-trip.

    * ``username`` — the impacted player's DISPLAY name (the game is FINAL/public,
      so naming pick winners/losers is intended; the event targets ``["chat"]`` only).
    * ``outcome`` — ``"busted"`` (a LOSS) or ``"cashed"`` (a WIN); the only two
      point-bearing outcomes the embed has fields for.
    * ``was_mortal_lock`` — whether the impacted pick was a mortal lock.
    """

    username: str
    outcome: Literal["busted", "cashed"]
    was_mortal_lock: bool


# The only two POINT-BEARING grade outcomes, mapped to the embed's two words.
# Keyed by the scoring ``GradeOutcome`` VALUE string (``pick_impacts`` carries the
# ``.value``), so a plain string map keeps this builder dependency-light — no import
# of the heavy scoring module. Outcomes absent from this map (PUSH / INELIGIBLE /
# UNGRADEABLE) are DROPPED by :func:`to_game_final_impacts`.
_GRADE_OUTCOME_TO_IMPACT_WORD: dict[str, Literal["busted", "cashed"]] = {
    "WIN": "cashed",
    "LOSS": "busted",
}


def to_game_final_impacts(pick_impacts: list[dict]) -> list[GameFinalImpact]:
    """Map ``get_game_final_context`` pick impacts to the event contract shape.

    Pure, never-raising mapper. Iterates ``pick_impacts`` IN ORDER (the context
    builder emits mortal-lock rows first, then by display_name) and, for each row:

    * skips it when ``display_name`` is falsy (defensive; matches the context builder);
    * looks up the outcome word — ``WIN`` -> ``"cashed"``, ``LOSS`` -> ``"busted"`` —
      and DROPS the row when the outcome is not point-bearing (``PUSH`` /
      ``INELIGIBLE`` / ``UNGRADEABLE`` belong in neither the Busted nor Cashed group);
    * otherwise emits ``{"username": display_name, "outcome": word,
      "was_mortal_lock": bool(is_mortal_lock)}``.

    Order is preserved from the input, so the mortal-lock-first ordering carries
    through to the embed.
    """
    impacts: list[GameFinalImpact] = []
    for impact in pick_impacts:
        display_name = impact.get("display_name")
        if not display_name:
            continue
        outcome = impact.get("outcome")
        word = _GRADE_OUTCOME_TO_IMPACT_WORD.get(outcome) if isinstance(outcome, str) else None
        if word is None:
            continue
        impacts.append(
            {
                "username": display_name,
                "outcome": word,
                "was_mortal_lock": bool(impact.get("is_mortal_lock")),
            }
        )
    return impacts


def game_final_event(
    *,
    week: int,
    away_abbr: str,
    home_abbr: str,
    away_score: int,
    home_score: int,
    impacts: list[GameFinalImpact] | None = None,
) -> dict:
    """Build a ``game.final`` event — one game went FINAL. DISPLAY data only.

    Carries the two team abbreviations and the two integer final scores plus the
    week number, and — since the game is FINAL/public — the ``impacts`` list: the
    per-user (display_name), busted/cashed, mortal-lock-flagged breakdown of EVERY
    graded pick on the game (see :class:`GameFinalImpact`). ``impacts`` defaults to
    ``[]`` so existing callers stay back-compatible. Still targets ``["chat"]`` only
    and carries no user_id/secret — display names on a final game are intended.
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
        "impacts": list(impacts or []),
    }


class RecapStandingsRow(TypedDict):
    """One season-standings row on the ``week.recap`` event (260705-kuv).

    Display-only + JSON-primitive: ``display_name`` plus derived integers
    (``rank``, ``season_total``, ``week_delta`` = the points gained THIS week).
    NEVER a ``user_id``.
    """

    rank: int
    display_name: str
    season_total: int
    week_delta: int


class RecapUpsetImpact(TypedDict):
    """The best-call / biggest-bust upset impact on the ``week.recap`` event.

    Display-only + JSON-primitive: ``display_name`` + the picked team's abbreviation
    + a ``side_label`` + the frozen ``spread`` carried as a STRING (mirrors
    ``get_game_final_context``'s ``spread_result``) + the ``is_mortal_lock`` flag.
    NEVER a ``user_id``.
    """

    display_name: str
    team_abbr: str
    side_label: str
    spread: str
    is_mortal_lock: bool


class RecapMortalLock(TypedDict):
    """One mortal-lock board row on the ``week.recap`` event (260705-kuv).

    Display-only + JSON-primitive: ``display_name`` + the ``hit`` boolean (grade
    outcome was a WIN) + the signed ``points`` + the ``side_label``. NEVER a
    ``user_id``.
    """

    display_name: str
    hit: bool
    points: int
    side_label: str


def week_recap_event(
    *,
    week: int,
    winner: str,
    winner_score: int,
    leader: str,
    leader_score: int,
    standings: list[RecapStandingsRow] | None = None,
    best_call: RecapUpsetImpact | None = None,
    biggest_bust: RecapUpsetImpact | None = None,
    mortal_locks: list[RecapMortalLock] | None = None,
) -> dict:
    """Build a ``week.recap`` event — a week's last game just went FINAL.

    ``winner`` is the week winner's DISPLAY name and ``winner_score`` their weekly
    score; ``leader`` is the season leader's DISPLAY name and ``leader_score``
    their season total.

    The enriched "closing ceremony" blocks (260705-kuv) are OPTIONAL and
    back-compatible — mirroring :func:`game_final_event`'s ``impacts=None -> []``
    pattern — so an existing caller passing only the original four kwargs gets the
    ORIGINAL key set PLUS the new keys defaulted to ``[]`` / ``None``:

    * ``standings`` — full ``[{rank, display_name, season_total, week_delta}, ...]``
      (defaults ``[]``);
    * ``best_call`` — the biggest UNDERDOG_COVER win, or ``None`` (default);
    * ``biggest_bust`` — the biggest FAVORITE_COVER bust, or ``None`` (default);
    * ``mortal_locks`` — the ``[{display_name, hit, points, side_label}, ...]`` board
      (defaults ``[]``).

    DISPLAY DATA ONLY (T-kuv-01): display names + integers/booleans + team
    abbreviations + a spread STRING — NEVER a ``user_id``, password, or token. The
    aggregation lives in :func:`app.services.notifications_read.get_week_recap_context`;
    this pure builder only shapes the display payload.
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
        "standings": list(standings or []),
        "best_call": best_call,
        "biggest_bust": biggest_bust,
        "mortal_locks": list(mortal_locks or []),
    }


def misc_graded_event(
    *,
    actor: str,
    week: int,
    prediction: str,
    verdict: str,
    points: int,
    grader: str | None = None,
) -> dict:
    """Build a ``misc.graded`` event — an admin graded ``actor``'s MISC prediction.

    DISPLAY data ONLY: ``actor`` is the player's DISPLAY name, ``prediction`` is the
    player's OWN free-text ``misc_text`` (bounded ≤280 by the model, same posture as
    ``pick_log_detail``'s MISC branch), ``verdict`` is the plain word ``"correct"`` or
    ``"incorrect"`` (the CALLER derives it from :class:`~app.models.PickResult`; the
    builder just carries the string), and ``points`` is the graded integer (may be
    negative). ``grader`` is the OPTIONAL display name of the admin who graded the
    pick (display-only, for an embed footer); it defaults to ``None`` and, when None,
    is simply carried as ``None`` (the renderer omits the footer). NO user_id, no
    secret — the key set is exactly
    ``{v, type, targets, actor, week, prediction, verdict, points, grader}``.

    HARD RULE (T-w9w-01): ``misc_text`` is hidden-until-lock, so the CALLER must only
    publish this event once the week's pick window is CLOSED — this pure builder does
    not (and cannot) enforce that; it just shapes the display payload.
    """
    return {
        "v": 1,
        "type": "misc.graded",
        "targets": ["chat"],
        "actor": actor,
        "week": week,
        "prediction": prediction,
        "verdict": verdict,
        "points": points,
        "grader": grader,
    }


# Memoized module-level synchronous redis client (None until first real use).
# Building a client per call discarded redis-py's internal connection pool on
# every event; caching one instance here reuses that pool across calls (T9).
_client = None


def _redis_client():
    """Return a MEMOIZED synchronous redis client built from ``settings.redis_url``.

    Isolated as a tiny seam so tests can monkeypatch it without touching a real
    socket, and so the URL is never hardcoded (reuse the celery broker setting).

    The client is constructed ONCE and cached in the module-global ``_client`` so
    redis-py's internal connection pool is reused across calls (no per-event pool
    churn); subsequent calls return the same instance. redis-py transparently
    reconnects a stale pooled connection, so no health-check/reconnect logic is
    needed here. This stays the SINGLE construction point AND the SINGLE monkeypatch
    seam: a test that patches ``notifications._redis_client`` replaces this whole
    function, so the cache branch never runs under those patches.
    """
    global _client
    if _client is None:
        import redis

        _client = redis.Redis.from_url(settings.redis_url)
    return _client


def _reset_redis_client() -> None:
    """Clear the memoized client so the next :func:`_redis_client` call rebuilds it.

    Test-only seam: exists ONLY so the reuse test can drop the cached (fake) client
    between tests and never leak one. NOT part of the public API — not exported.
    """
    global _client
    _client = None


# Default cooldown window (seconds) for the dedup'd chat milestones below
# (roster.complete + misc.picked). ~5 minutes: long enough that a flurry of
# re-submits on the same week collapses to ONE chat line, short enough that a
# genuine re-completion much later can re-announce (accepted).
COOLDOWN_TTL_SECONDS = 300


def claim_cooldown(key: str, ttl_seconds: int = COOLDOWN_TTL_SECONDS) -> bool:
    """Atomically claim ``key`` for ``ttl_seconds`` — best-effort, FAIL-OPEN.

    Runs a Redis ``SET key 1 NX EX ttl_seconds`` via the shared
    :func:`_redis_client` seam (so tests monkeypatch ONE place). Returns:

    * ``True``  — the key was newly claimed (first time within the TTL window):
      the caller SHOULD publish its milestone.
    * ``False`` — the key already existed (a repeat within the window): the
      caller SHOULD suppress the duplicate.

    FAIL-OPEN (T-itg-03): the whole op is wrapped in try/except. On ANY error it
    logs a structlog warning (mirroring :data:`publish_event`'s posture) and
    returns ``True`` — it NEVER raises into the request path and NEVER silently
    swallows the milestone. The events bus is itself Redis, so if Redis is fully
    down nothing posts anyway; failing open here just means we do not let the
    dedup op become a new failure mode for the pick submit.
    """
    try:
        client = _redis_client()
        # redis-py: set(..., nx=True, ex=ttl) returns True on a successful new
        # claim and None when the key already exists (NX rejected the write).
        return bool(client.set(key, "1", nx=True, ex=ttl_seconds))
    except Exception:
        logger.warning(
            "cooldown_claim_failed",
            key=key,
            ttl_seconds=ttl_seconds,
        )
        return True  # FAIL-OPEN — never swallow the milestone.


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
