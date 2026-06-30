"""Pure pick-window logic — when a week is open and when a game is locked.

This is a sibling to the pure ``scoring`` and pick-validation services: it is the
last pure-logic piece before the pick-submission endpoint, encoding *when* picks
may be made. Given a week's games
and the previous week's games — plus a caller-injected ``now`` — it computes the
week-level pick window and a finer-grained per-game lock. Like its siblings it is
deliberately **pure**:

* It performs **no I/O of any kind** — no database session, no network, no file
  access, no writes. The database engine module is never imported.
* It imports only :mod:`app.models` (for :class:`~app.models.Game`) plus the
  standard library (:mod:`datetime`). It does **not** import the sibling
  ``scoring`` or pick-validation service — the pure services stay independent of
  one another (purity layering).
* It **never reads the clock**: no current-time call is ever made here. ``now``
  is always injected by the caller, which keeps every decision deterministic and
  testable.
* It **never mutates** the ``Game`` instances handed to it. It only inspects
  them and returns a value.

Scope is **window + lock computation only**:

* :func:`compute_window` derives the week's :class:`PickWindow` (``open_at``,
  ``close_at``) from kickoffs.
* :func:`is_pick_open` decides whether the week window is open at a given ``now``.
* :func:`is_game_locked` decides whether a single game has kicked off.

Rules (single source of truth — see PROJECT.md "Active" requirements):

* The window **closes** at the current week's **earliest kickoff** (the week's
  first game). Games that kick off later in the same week are guarded
  individually by :func:`is_game_locked`, not by the week-level close.
* The window **opens** after the previous week's last game ends. The schema has
  no explicit game-end timestamp (``Game`` carries only ``kickoff_at``, status,
  and scores), so this boundary is **APPROXIMATED** as the previous week's
  latest kickoff plus :data:`DEFAULT_GAME_DURATION` (~3.5h). See the constant.
* **Week 1** has no previous week, so its lower boundary is unbounded-open
  (``open_at = None``): the window is open from the start and closes at week 1's
  first kickoff.
* Boundary semantics are **half-open**: open includes ``open_at`` and excludes
  ``close_at`` (at exactly the first kickoff the week window is closed). The
  per-game lock is inclusive of kickoff: locked at or after ``kickoff_at``.

Reserved exceptions are for **programmer errors only**, mirroring how the sibling
services reserve ``KeyError`` for a missing game:

* a naive (timezone-unaware) ``now`` or kickoff raises a deliberate, labeled
  :class:`ValueError` (never a bare "can't compare offset-naive and
  offset-aware" ``TypeError``);
* an empty current week, or a current week whose only game has no kickoff,
  raises :class:`ValueError` (there is nothing to close on).

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python`` for any
> commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from app.models import Game

# Approximate NFL game length, used only to derive the previous week's
# window-open boundary. KNOWN LIMITATION: the schema has no explicit game-end
# timestamp (``Game`` has only ``kickoff_at``, ``status`` and scores), so the
# previous week's last game's *end* is approximated as its kickoff plus this
# duration (~3.5h). If a real game-end timestamp field is ever added to ``Game``
# it would be the better source and this approximation should be revisited. The
# constant is overridable per call via ``compute_window(..., game_duration=...)``.
DEFAULT_GAME_DURATION: timedelta = timedelta(hours=3, minutes=30)


@dataclass(frozen=True)
class PickWindow:
    """The immutable computed pick window for one week.

    ``open_at`` of ``None`` means the lower boundary is unbounded-open (e.g.
    week 1, which has no previous week): the window is open from the start and
    closes at ``close_at`` — the week's first kickoff.
    """

    open_at: datetime | None
    close_at: datetime


def _require_aware(dt: datetime, label: str) -> None:
    """Raise a deliberate, labeled ``ValueError`` if ``dt`` is naive.

    Comparing a naive datetime against a tz-aware one would otherwise raise a
    bare ``TypeError`` ("can't compare offset-naive and offset-aware"); this
    turns that into an explicit, well-labeled error so a wrong-timezone window
    decision can never be made silently.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware (got a naive datetime)")


def _valid_kickoffs(games: Iterable[Game], label: str) -> list[datetime]:
    """Collect the non-None, tz-aware kickoffs from ``games``.

    Games with ``kickoff_at is None`` are skipped (no scheduled time to reason
    about); each present kickoff is guarded for tz-awareness so a naive value
    raises a deliberate ``ValueError`` via :func:`_require_aware`.
    """
    kickoffs: list[datetime] = []
    for game in games:
        if game.kickoff_at is None:
            continue
        _require_aware(game.kickoff_at, f"{label} kickoff_at")
        kickoffs.append(game.kickoff_at)
    return kickoffs


def compute_window(
    week_games: Sequence[Game],
    prev_week_games: Iterable[Game] | None = None,
    *,
    game_duration: timedelta = DEFAULT_GAME_DURATION,
) -> PickWindow:
    """Compute the :class:`PickWindow` for one week from kickoffs.

    ``close_at`` is the **earliest** kickoff among ``week_games`` — the week's
    first game. This is the WEEK-LEVEL close; games that kick off later in the
    same week are guarded individually by :func:`is_game_locked`.

    ``open_at`` is the previous week's **latest** kickoff plus ``game_duration``
    (the documented game-end approximation — see :data:`DEFAULT_GAME_DURATION`).
    When ``prev_week_games`` is ``None`` or has no valid kickoffs (week 1, or an
    unscheduled previous week), ``open_at`` is ``None`` (unbounded-open).

    Does not mutate its inputs. Raises :class:`ValueError` when ``week_games`` is
    empty or none of its games has a (tz-aware) kickoff — there is nothing to
    close on — and when any present kickoff is naive.
    """
    current_kickoffs = _valid_kickoffs(week_games, "current-week")
    if not current_kickoffs:
        raise ValueError("current week has no game with a kickoff to close the window on")
    close_at = min(current_kickoffs)

    if prev_week_games is None:
        open_at: datetime | None = None
    else:
        prev_kickoffs = _valid_kickoffs(prev_week_games, "previous-week")
        open_at = max(prev_kickoffs) + game_duration if prev_kickoffs else None

    return PickWindow(open_at=open_at, close_at=close_at)


def is_pick_open(window: PickWindow, now: datetime) -> bool:
    """Whether the week-level pick window is open at ``now``.

    Half-open semantics: the window includes ``open_at`` and excludes
    ``close_at`` — at exactly the first kickoff the week window is closed. An
    ``open_at`` of ``None`` is unbounded-open (any ``now`` strictly before
    ``close_at`` is open). ``now`` must be timezone-aware (a naive value raises
    :class:`ValueError`).
    """
    _require_aware(now, "now")
    return (window.open_at is None or window.open_at <= now) and now < window.close_at


def is_game_locked(game: Game, now: datetime) -> bool:
    """Whether ``game`` has kicked off (and so picks on it are locked) at ``now``.

    Returns ``True`` iff ``now >= game.kickoff_at`` — locked at or after kickoff.
    A game with ``kickoff_at is None`` has no scheduled time to lock on and is
    treated as **not** locked. ``now`` (and the kickoff, when present) must be
    timezone-aware; a naive value raises :class:`ValueError`.

    This is the finer-grained guard that complements the week-level window: the
    window closes at the week's earliest kickoff, but games later in the same
    week stay unlocked until their own kickoff.
    """
    _require_aware(now, "now")
    if game.kickoff_at is None:
        return False
    _require_aware(game.kickoff_at, "game kickoff_at")
    return now >= game.kickoff_at
