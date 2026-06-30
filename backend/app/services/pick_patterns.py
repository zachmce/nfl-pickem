"""Pure pick-pattern scanner — the deterministic FACT owner of pickem-chat (260627-nef).

This module owns the TRUTH for the local-LLM personality layer: it detects a
pattern in a player's locked picks from plain dicts and returns a structured
fact, with NO I/O of any kind — no ORM, no Session, no discord, no network, no
clock. The LLM (in :mod:`app.bot.llm_client`) only phrases the fact this module
produces; it never invents a stat. Keeping the scan pure means the FACTS are
unit-testable offline and can never hallucinate.

v1 detector: same ``(team_abbr, side)`` picked in an unbroken run of >= 3
consecutive weeks ending at the target week. A totals pick is keyed on a REAL
team (the caller expands one OVER/UNDER pick into TWO team keys, one per team in
its game), so a totals streak survives a changing opponent — "KC OVER" fires
across KC@DEN, KC@LV, KC@SF.

Pick-key shape (the unit this module consumes)
----------------------------------------------
A *pick key* is a plain ``{"week": int, "team_abbr": str, "side": str}`` dict
where ``side`` is one of ``"FAVORITE" | "UNDERDOG" | "OVER" | "UNDER"`` and
``team_abbr`` is ALWAYS a real team's abbreviation (never an ``away@home``
matchup label). The caller (the read service in
:mod:`app.services.notifications_read`) is responsible for the expansion:
a spread pick yields ONE key, a totals pick yields TWO (one per team in its
game).
"""

from __future__ import annotations

# Streak threshold for the v1 detector: a run shorter than this never fires.
STREAK_THRESHOLD = 3


def scan_streak(
    target_week: int,
    slate_keys: list[dict],
    history_keys: list[dict],
) -> dict | None:
    """Return the longest qualifying ``(team_abbr, side)`` streak, or ``None``.

    A streak QUALIFIES when the SAME ``(team_abbr, side)`` pair appears in an
    unbroken run of consecutive weeks (``target_week``, ``target_week - 1``, ...
    with no gap) ending AT ``target_week``, of length >= :data:`STREAK_THRESHOLD`.

    * ``slate_keys`` — the player's locked keys for ``target_week`` (gates which
      pairs are even considered: a streak must include the target week, so only a
      pair present in the slate can end there).
    * ``history_keys`` — the player's keys across the recent weeks (may include
      the target week too; duplicates are tolerated).

    A single totals pick contributes TWO keys (the caller expands it), so a
    week's entry may carry 2 keys from one pick; each distinct ``(team_abbr,
    side)`` pair is evaluated independently.

    Returns ``{"team_abbr", "side", "streak_weeks": int}`` for the BEST pair
    (deterministic tie-break: longest streak, then alphabetical ``team_abbr``,
    then ``side``), or ``None`` when nothing reaches the threshold.

    Pure: no I/O, no clock, no ORM — plain dicts in, plain dict (or None) out.
    """
    # The pairs eligible to streak are exactly those the player picked THIS week
    # (the run must end at target_week). Anything not in the slate cannot qualify.
    target_pairs = {(k["team_abbr"], k["side"]) for k in slate_keys if k.get("week") == target_week}
    if not target_pairs:
        return None

    # weeks each pair was picked, across the full history (deduped per pair).
    weeks_by_pair: dict[tuple[str, str], set[int]] = {}
    for k in history_keys:
        pair = (k["team_abbr"], k["side"])
        weeks_by_pair.setdefault(pair, set()).add(k["week"])

    best: dict | None = None
    for team_abbr, side in target_pairs:
        weeks = weeks_by_pair.get((team_abbr, side), set())
        if target_week not in weeks:
            continue  # must include the target week to "end at" it
        # Count back from target_week while each prior week is present (no gap).
        streak = 0
        w = target_week
        while w in weeks:
            streak += 1
            w -= 1
        if streak < STREAK_THRESHOLD:
            continue
        candidate = {
            "team_abbr": team_abbr,
            "side": side,
            "streak_weeks": streak,
        }
        if best is None or _is_better(candidate, best):
            best = candidate

    return best


def _is_better(candidate: dict, current: dict) -> bool:
    """Deterministic ranking: longest streak, then alphabetical team_abbr, then side."""
    return (
        -candidate["streak_weeks"],
        candidate["team_abbr"],
        candidate["side"],
    ) < (
        -current["streak_weeks"],
        current["team_abbr"],
        current["side"],
    )


def _detect_streak(target_week: int, slate_keys: list[dict], history_keys: list[dict]):
    """Thin adapter so :data:`PATTERN_DETECTORS` entries share one call shape."""
    return scan_streak(target_week, slate_keys, history_keys)


# Extensible detector catalog. v1 has ONE entry; adding the future "first
# deviation after a long streak" detector (NOT implemented here — it would flag
# the week a player BREAKS an established streak) is a matter of appending a new
# (name, fn) pair, with no change to the callers that iterate this list.
PATTERN_DETECTORS = [
    ("streak>=3", _detect_streak),
]
