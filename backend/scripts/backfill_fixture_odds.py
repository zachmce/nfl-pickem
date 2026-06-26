"""One-shot idempotent backfill of REAL ESPN-core odds for the 2025 fixture's
odds-less games (the wk13 NYG@NE gap game + all of weeks 14-18).

Why this exists: ``gen_2025_fixture.select_odds_item`` only accepts the ESPN BET
provider (name "ESPN BET" / id 58). For finaled wk14-18 games (and one wk13 game)
ESPN BET lines are gone, but the SAME core-API odds endpoint still returns real
DraftKings lines (provider id "100") in the exact shape
``gen_2025_fixture.normalize_odds`` already consumes. A *permissive* selector
unblocks those games so the full 18-week 2025 demo season becomes
pickable/gradeable.

This script is dev/test tooling: it lives under ``backend/scripts/`` (NOT
``backend/app/``) and imports its normalization helpers from
``scripts.gen_2025_fixture`` so the backfilled odds match the existing flat shape
the importer (``app.seeds.fixture_2025``) consumes. It does NOT modify
``gen_2025_fixture.py`` (the generator's ESPN-BET-first behavior and its tests
stay untouched).

Design: pure selection/merge logic (``select_odds_item_permissive``,
``provider_label``, ``merge_odds_into_fixture``) is isolated from impure I/O
(network + file writes in ``main``) so it is fully unit-testable offline.

Idempotency: games that already carry odds are skipped untouched (never
re-fetched, never reordered). Re-running after a successful backfill is a no-op
for already-filled games.

Run from the ``backend/`` directory::

    cd backend && python -m scripts.backfill_fixture_odds

Optional flags: ``--delay SECONDS`` (polite per-game delay, default 0.4),
``--path PATH`` (override the fixture file).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

# Reuse the generator's pure normalization helpers so backfilled odds are
# byte-shape-identical to the wk1-13 lines. We do NOT modify gen_2025_fixture.
from scripts.gen_2025_fixture import (
    ESPN_BET_PROVIDER_ID,
    ESPN_BET_PROVIDER_NAME,
    fetch_event_odds,
    normalize_odds,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default fixture path resolved relative to this module so the script works
# regardless of the current working directory. This file lives at
# backend/scripts/backfill_fixture_odds.py; the packaged fixture is two levels
# up under backend/app/seeds/data/.
DEFAULT_FIXTURE_PATH: Path = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "seeds"
    / "data"
    / "nfl_2025_regular_season.json"
)

_UNKNOWN_PROVIDER = "unknown"


# ---------------------------------------------------------------------------
# Pure selection / labeling (no network — unit-tested offline)
# ---------------------------------------------------------------------------


def _odds_item_is_usable(item: Any) -> bool:
    """True iff ``normalize_odds(item)`` yields ALL FOUR core fields non-null.

    "Usable line" == spread AND total AND favorite_team_id AND underdog_team_id
    are all present. A partial line is NOT usable (the importer would crash on
    ``abs(None)`` / ``int(None)``), so such items are rejected here.
    """
    normalized = normalize_odds(item)
    if normalized is None:
        return False
    return (
        normalized.get("spread") is not None
        and normalized.get("total") is not None
        and normalized.get("favorite_team_id") is not None
        and normalized.get("underdog_team_id") is not None
    )


def select_odds_item_permissive(odds_response: Any) -> dict[str, Any] | None:
    """Pick the best usable odds item from a core-API odds response.

    Mirrors ``gen_2025_fixture.select_odds_item``'s defensive shape-checking but
    uses an ordered preference instead of ESPN-BET-only:

    1. ``provider.name == "ESPN BET"`` (keeps labels consistent with wk1-13).
    2. ``provider.id == 58`` (string or int).
    3. the FIRST item that yields a *usable* line — one whose ``normalize_odds``
       result has non-None spread AND total AND favorite_team_id AND
       underdog_team_id.

    Returns ``None`` when no item qualifies (no items, items not a list, every
    item unusable). The caller then leaves that game ``odds: null`` and reports
    it — a partial/empty line is NEVER returned.
    """
    if not isinstance(odds_response, dict):
        return None
    items = odds_response.get("items")
    if not isinstance(items, list):
        return None

    id_fallback: dict[str, Any] | None = None
    first_usable: dict[str, Any] | None = None

    for item in items:
        if not isinstance(item, dict):
            continue
        provider = item.get("provider")
        if not isinstance(provider, dict):
            continue

        # Preference 1: exact ESPN BET name match wins immediately, but only if
        # it actually carries a usable line (a name match with a partial line is
        # worse than a complete line from another book).
        name = provider.get("name")
        if name == ESPN_BET_PROVIDER_NAME and _odds_item_is_usable(item):
            return item

        provider_id = provider.get("id")
        # Preference 2: ESPN BET by id (string or int), usable line required.
        if (
            id_fallback is None
            and (
                provider_id == ESPN_BET_PROVIDER_ID
                or provider_id == str(ESPN_BET_PROVIDER_ID)
            )
            and _odds_item_is_usable(item)
        ):
            id_fallback = item

        # Preference 3: remember the first usable line from any provider.
        if first_usable is None and _odds_item_is_usable(item):
            first_usable = item

    if id_fallback is not None:
        return id_fallback
    return first_usable


def provider_label(item: Any) -> str:
    """Return an honest provider label for a selected odds item.

    Reads the item's actual ``provider.name`` (e.g. "DraftKings"), falling back
    to the string form of ``provider.id`` if the name is missing, else
    ``"unknown"``. Used so backfilled odds are labeled by their real provider
    rather than a hardcoded constant.
    """
    if not isinstance(item, dict):
        return _UNKNOWN_PROVIDER
    provider = item.get("provider")
    if not isinstance(provider, dict):
        return _UNKNOWN_PROVIDER
    name = provider.get("name")
    if isinstance(name, str) and name.strip():
        return name
    provider_id = provider.get("id")
    if provider_id is not None and str(provider_id).strip():
        return str(provider_id)
    return _UNKNOWN_PROVIDER


# ---------------------------------------------------------------------------
# Pure merge (no network / no file I/O — fetch_odds_fn injected, unit-tested)
# ---------------------------------------------------------------------------


@dataclass
class BackfillReport:
    """Summary of a merge run (pure — no I/O)."""

    games_total: int = 0
    already_had_odds: int = 0
    filled: int = 0
    still_null: list[dict[str, Any]] = field(default_factory=list)
    filled_by_provider: Counter = field(default_factory=Counter)
    filled_by_week: Counter = field(default_factory=Counter)

    @property
    def still_null_count(self) -> int:
        return len(self.still_null)


def _matchup(game: dict[str, Any]) -> str:
    """Human-readable AWAY@HOME label for reporting (defensive)."""
    away = game.get("away") or {}
    home = game.get("home") or {}
    away_abbr = away.get("abbreviation") or "?"
    home_abbr = home.get("abbreviation") or "?"
    return f"{away_abbr}@{home_abbr}"


def merge_odds_into_fixture(
    fixture: dict[str, Any],
    fetch_odds_fn: Callable[[str, str], Any],
) -> BackfillReport:
    """Backfill odds IN PLACE for every game whose ``odds`` is null.

    Iterates ``fixture["games"]`` IN ORDER. For each game already carrying odds,
    skips it untouched (idempotent). For each null-odds game it calls
    ``fetch_odds_fn(espn_event_id, competition_id)``, runs the response through
    ``select_odds_item_permissive`` + ``normalize_odds``, and — only if a USABLE
    line comes back — writes the normalized odds object back onto the game with
    its ``provider`` set to ``provider_label(selected_item)`` (the real provider,
    e.g. "DraftKings"), NOT a hardcoded constant. The negative-favorite spread
    sign convention from ``normalize_odds`` is preserved so the importer's
    ``abs(spread)`` yields a positive magnitude.

    Games whose fetch/selection yields nothing usable are LEFT null and collected
    into ``report.still_null`` (event_id, week, matchup) — never silently dropped
    and never written as a partial line. Game order is never changed and no game
    is added or removed.

    :param fixture: the parsed fixture dict (mutated in place).
    :param fetch_odds_fn: injected ``(event_id, competition_id) -> odds_response``
        so this function is offline-testable with a fake.
    :returns: a :class:`BackfillReport`.
    """
    games = fixture.get("games")
    if not isinstance(games, list):
        return BackfillReport()

    report = BackfillReport(games_total=len(games))

    for game in games:
        if not isinstance(game, dict):
            continue
        if game.get("odds") is not None:
            report.already_had_odds += 1
            continue

        event_id = game.get("espn_event_id")
        competition_id = game.get("competition_id")
        week = game.get("week")

        selected: dict[str, Any] | None = None
        if event_id and competition_id:
            odds_response = fetch_odds_fn(event_id, competition_id)
            selected = select_odds_item_permissive(odds_response)

        if selected is None:
            report.still_null.append(
                {
                    "espn_event_id": event_id,
                    "week": week,
                    "matchup": _matchup(game),
                }
            )
            continue

        normalized = normalize_odds(selected)
        # _odds_item_is_usable already guaranteed completeness inside the
        # selector, but re-guard here so a regression never writes a partial line.
        if normalized is None or not (
            normalized.get("spread") is not None
            and normalized.get("total") is not None
            and normalized.get("favorite_team_id") is not None
            and normalized.get("underdog_team_id") is not None
        ):
            report.still_null.append(
                {
                    "espn_event_id": event_id,
                    "week": week,
                    "matchup": _matchup(game),
                }
            )
            continue

        label = provider_label(selected)
        normalized["provider"] = label
        game["odds"] = normalized
        report.filled += 1
        report.filled_by_provider[label] += 1
        if week is not None:
            report.filled_by_week[week] += 1

    return report


def recompute_metadata(fixture: dict[str, Any]) -> dict[str, int]:
    """Recompute games_total / games_with_odds / games_without_odds from games.

    Updates ``fixture["metadata"]`` in place and returns the recomputed counts.
    """
    games = fixture.get("games")
    games = games if isinstance(games, list) else []
    total = len(games)
    with_odds = sum(1 for g in games if isinstance(g, dict) and g.get("odds") is not None)
    without_odds = total - with_odds

    metadata = fixture.setdefault("metadata", {})
    metadata["games_total"] = total
    metadata["games_with_odds"] = with_odds
    metadata["games_without_odds"] = without_odds
    return {
        "games_total": total,
        "games_with_odds": with_odds,
        "games_without_odds": without_odds,
    }


# ---------------------------------------------------------------------------
# Impure orchestration (network + file I/O — not exercised by unit tests)
# ---------------------------------------------------------------------------

_BACKFILL_NOTE = (
    "Weeks 1-13 carry ESPN BET stand-in lines (labeled per game). Weeks 14-18 "
    "and the wk13 NYG@NE game were backfilled with REAL ESPN-core lines "
    "(predominantly DraftKings) pulled on backfill, each labeled by its actual "
    "provider — superseding the earlier claim that 2025 has no DraftKings odds "
    "anywhere. Any game still without a usable line is reported in "
    "games_without_odds."
)


def _build_fetch_fn(delay: float) -> Callable[[str, str], Any]:
    """Build a real network fetch_odds_fn with a polite per-game delay.

    The delay is applied AFTER each request (matching gen_2025_fixture's cadence)
    so sequential backfill GETs stay polite to the ESPN core API.
    """
    import time

    def fetch(event_id: str, competition_id: str) -> Any:
        response = fetch_event_odds(event_id, competition_id)
        time.sleep(delay)
        return response

    return fetch


def write_fixture(fixture: dict[str, Any], path: Path) -> None:
    """Write the fixture back with the SAME formatting the file already uses.

    Uses ``json.dump(indent=2, ensure_ascii=False)`` + a trailing newline so the
    diff is limited to the filled games + the metadata block.
    """
    text = json.dumps(fixture, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill REAL ESPN-core odds for the 2025 fixture's odds-less games "
            "(wk13 NYG@NE + wk14-18). Idempotent; dev/test tooling only."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="Fixture JSON path (default: backend/app/seeds/data/nfl_2025_regular_season.json).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="Polite delay in seconds between per-game odds requests (default: 0.4).",
    )
    args = parser.parse_args(argv)

    path: Path = args.path
    with open(path, encoding="utf-8") as fh:
        fixture = json.load(fh)

    before = recompute_metadata(dict(fixture))  # snapshot counts before merge
    print(
        f"[start] {before['games_total']} games; "
        f"{before['games_with_odds']} with odds, "
        f"{before['games_without_odds']} without (to backfill)",
        file=sys.stderr,
    )

    fetch_fn = _build_fetch_fn(args.delay)
    report = merge_odds_into_fixture(fixture, fetch_fn)

    # Recompute metadata from the actual games array post-merge.
    counts = recompute_metadata(fixture)
    metadata = fixture["metadata"]
    metadata["note"] = _BACKFILL_NOTE
    metadata["backfilled_at"] = datetime.now(UTC).isoformat()

    write_fixture(fixture, path)

    # Report.
    print(
        f"[done] filled {report.filled} games "
        f"({report.already_had_odds} already had odds, "
        f"{report.still_null_count} still null)",
        file=sys.stderr,
    )
    if report.filled_by_provider:
        providers = ", ".join(
            f"{name}: {n}" for name, n in report.filled_by_provider.most_common()
        )
        print(f"[done] filled by provider: {providers}", file=sys.stderr)
    if report.filled_by_week:
        weeks = ", ".join(
            f"wk{wk}: {n}" for wk, n in sorted(report.filled_by_week.items())
        )
        print(f"[done] filled by week: {weeks}", file=sys.stderr)
    if report.still_null:
        print("[done] STILL NULL (no usable line found):", file=sys.stderr)
        for entry in report.still_null:
            print(
                f"    event {entry['espn_event_id']} "
                f"(wk{entry['week']}, {entry['matchup']})",
                file=sys.stderr,
            )

    print(
        f"Wrote fixture: {path}\n"
        f"Games: {counts['games_total']} "
        f"({counts['games_with_odds']} with odds, "
        f"{counts['games_without_odds']} without)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
