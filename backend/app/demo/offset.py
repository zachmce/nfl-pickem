"""Offset-positioning helper for the demo season walkthrough.

The :class:`~app.scoreboard.demo.Demo2025Source` serves the real 2025 fixture
*positioned around the present* via a constructor ``offset``, deriving status
against the REAL clock. This module computes the ``offset`` that lands a target
week on the intended side of the status/window boundaries — so the demo driver
can drive the season "as if live" without any virtual clock.

Two positioning phases (:class:`DemoPhase`):

* ``WINDOW_OPEN_FOR_WEEK`` — position so target week N's earliest game is in the
  FUTURE (week N SCHEDULED, the week-level pick window not yet closed) AND week
  N-1's latest game has ended (so the poller stamps ``window_opens_at`` and
  ``is_pick_open`` is true). Week 1 has no predecessor, so only the
  future-earliest-kickoff condition applies.
* ``ALL_WEEK_FINAL`` — position so every week-N game's
  ``effective_kickoff + GAME_DURATION`` is in the PAST (all derive FINAL).

Design — pure core, no DB/network (mirrors ``pick_window`` / ``derive_status``):

* Imports only :mod:`app.seeds.fixture_2025` (for ``FIXTURE_PATH``),
  :data:`app.scoreboard.demo.GAME_DURATION` (so the FINAL boundary matches the
  source's own duration EXACTLY — never a second hardcoded copy that could
  drift), and the standard library. It opens no DB session and touches no
  network. ``app.db`` / ``app.config`` are never imported.
* ``now`` is always injected (the caller passes the real
  ``datetime.now(timezone.utc)``); this module never reads the clock itself, so
  every computation is deterministic and testable.
* The tz-aware guard is re-declared locally (``_require_aware``) rather than
  imported from the sibling service — exactly how ``demo.py`` re-declares its
  own helpers (purity layering).

Boundary correctness
--------------------

``derive_status`` / ``is_pick_open`` / ``is_game_locked`` use half-open
boundaries (``now < kickoff`` is SCHEDULED/open; ``now >= kickoff + duration``
is FINAL). Positioning a kickoff *exactly* on a boundary is ambiguous against a
real ``now`` that advances between the offset computation and the source's own
``fetch_week`` call. We therefore apply a small :data:`DEFAULT_MARGIN` so the
positioned kickoff lands STRICTLY on the intended side:

* WINDOW_OPEN: target week's earliest positioned kickoff = ``now + margin``
  (strictly future ⇒ SCHEDULED, window open). The margin absorbs the real-clock
  drift between this computation and ``fetch_week``.
* ALL_WEEK_FINAL: target week's latest positioned kickoff + ``GAME_DURATION`` =
  ``now - margin`` (strictly past ⇒ FINAL).

The margin is generous (a full day) precisely because the per-week fixture
spacing is multi-day, so a one-day cushion never crosses into a neighbouring
week while still dwarfing any sub-second clock drift during a test or run.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from app.scoreboard.demo import GAME_DURATION
from app.seeds.fixture_2025 import FIXTURE_PATH

# A generous safety cushion so a positioned kickoff lands STRICTLY on the
# intended side of a half-open boundary, immune to the real clock advancing
# between this computation and the source's own ``fetch_week`` call. A full day
# is far larger than any clock drift yet far smaller than the multi-day spacing
# between fixture weeks, so it never bleeds into a neighbouring week.
DEFAULT_MARGIN: timedelta = timedelta(days=1)


class DemoPhase(str, Enum):
    """Which side of the boundaries to position a target week on."""

    WINDOW_OPEN_FOR_WEEK = "WINDOW_OPEN_FOR_WEEK"
    ALL_WEEK_FINAL = "ALL_WEEK_FINAL"


def _require_aware(dt: datetime, label: str) -> None:
    """Raise a deliberate, labeled ``ValueError`` if ``dt`` is naive.

    Mirrors ``pick_window._require_aware`` / ``demo._require_aware``
    (re-declared locally, not imported): turns a bare "can't compare
    offset-naive and offset-aware" ``TypeError`` into an explicit error so a
    wrong-timezone positioning decision can never be made silently.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware (got a naive datetime)")


def _parse_kickoff(value: Any) -> datetime | None:
    """Parse the fixture's ISO kickoff (e.g. ``2025-09-05T00:20Z``) tz-aware.

    Mirrors ``demo._parse_kickoff`` / ``fixture_2025._parse_kickoff``
    (re-declared locally): normalizes a trailing ``Z`` to ``+00:00`` so the
    result is tz-aware. Returns ``None`` for missing/blank input.
    """
    if not value or not isinstance(value, str):
        return None
    normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
    return datetime.fromisoformat(normalized)


def load_fixture_kickoffs(path: Path | None = None) -> dict[int, list[datetime]]:
    """Load the fixture's per-week kickoffs (tz-aware), keyed by week number.

    Reads the same packaged 2025 fixture the seeds use (``FIXTURE_PATH``),
    parsing each game's kickoff tz-aware. Pure file read — no DB, no network.
    A game with a missing/blank kickoff is skipped.
    """
    fixture_path = path or FIXTURE_PATH
    with open(fixture_path, encoding="utf-8") as fh:
        fixture = json.load(fh)

    weeks: dict[int, list[datetime]] = {}
    for raw in fixture.get("games", []):
        kickoff = _parse_kickoff(raw.get("kickoff"))
        if kickoff is None:
            continue
        week = int(raw["week"])
        weeks.setdefault(week, []).append(kickoff)
    return weeks


def _week_kickoffs(
    weeks_kickoffs: Mapping[int, list[datetime]], week: int, label: str
) -> list[datetime]:
    """Return the tz-aware kickoffs for ``week`` or raise ``ValueError``.

    An absent or empty week has nothing to position on — a deliberate
    ``ValueError`` (a real positioning bug to surface, not paper over). Each
    present kickoff is guarded tz-aware via :func:`_require_aware`.
    """
    kickoffs = list(weeks_kickoffs.get(week, []))
    if not kickoffs:
        raise ValueError(f"{label} week {week} has no kickoffs to position on")
    for ko in kickoffs:
        _require_aware(ko, f"{label} week {week} kickoff")
    return kickoffs


def compute_offset(
    now: datetime,
    *,
    target_week: int,
    phase: DemoPhase,
    weeks_kickoffs: Mapping[int, list[datetime]],
    margin: timedelta = DEFAULT_MARGIN,
) -> timedelta:
    """Compute the ``Demo2025Source(offset=...)`` for ``target_week`` + ``phase``.

    ``now`` (real UTC clock, injected) and every fixture kickoff must be
    tz-aware (a naive value raises :class:`ValueError`). The returned
    ``timedelta`` is added to each fixture kickoff by the source.

    WINDOW_OPEN_FOR_WEEK
        Position so target week N's EARLIEST positioned kickoff = ``now +
        margin`` (strictly future ⇒ week N SCHEDULED, window open). Then ASSERT
        (do not silently fix) that week N-1's LATEST positioned kickoff +
        ``GAME_DURATION`` < ``now`` (predecessor fully ended ⇒ FINAL + open
        boundary stampable). If that invariant cannot hold for the given fixture
        spacing, raise :class:`ValueError` describing the conflict rather than
        returning a wrong offset. Week 1 has no predecessor, so only the
        future-earliest condition applies.

    ALL_WEEK_FINAL
        Position so target week N's LATEST positioned kickoff + ``GAME_DURATION``
        = ``now - margin`` (strictly past ⇒ every week-N game FINAL).
    """
    _require_aware(now, "now")
    target_kickoffs = _week_kickoffs(weeks_kickoffs, target_week, "target")

    if phase is DemoPhase.WINDOW_OPEN_FOR_WEEK:
        earliest = min(target_kickoffs)
        # Earliest positioned kickoff = now + margin  ->  offset = target - src.
        offset = (now + margin) - earliest

        if target_week > 1:
            prev_kickoffs = _week_kickoffs(weeks_kickoffs, target_week - 1, "previous")
            prev_latest_end = max(prev_kickoffs) + offset + GAME_DURATION
            if not prev_latest_end < now:
                raise ValueError(
                    f"cannot position week {target_week} window-open: with the "
                    f"target's earliest kickoff at now+{margin}, the previous "
                    f"week {target_week - 1}'s latest game ends at "
                    f"{prev_latest_end.isoformat()} which is not strictly "
                    f"before now ({now.isoformat()}). The fixture spacing does "
                    "not allow this positioning."
                )
        return offset

    # DemoPhase.ALL_WEEK_FINAL
    latest = max(target_kickoffs)
    # latest + offset + GAME_DURATION = now - margin  ->  solve for offset.
    return (now - margin - GAME_DURATION) - latest
