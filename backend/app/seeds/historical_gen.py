"""DEV-ONLY regeneration of the committed historical-games CSV artifact.

This script fetches the nflverse ``games.csv`` over HTTPS, trims it to the final
games we ingest (1999 -> last completed season, with a posted line), validates
every team abbreviation, and writes the reviewed static artifact at
``data/historical_games.csv`` that the startup upsert
(:mod:`app.seeds.historical_games`) consumes.

It is **NOT** run at startup, in the poller, or in tests — it is the ONE-COMMAND,
out-of-band regeneration path, run once per season on a dev machine::

    cd backend
    .venv/bin/python -m app.seeds.historical_gen

Stdlib only (``csv`` / ``urllib.request`` / ``pathlib``). The parsing and filter
semantics mirror the verified spike
``.planning/spikes/002-rating-model-vs-line/backtest.py``.

Caveats baked into the artifact: nflverse ``spread_line`` is a **consensus** number
(not strictly the closing line), and there is NO moneyline (the AusSportsBetting
overlay is future work, out of scope). The artifact stores nflverse abbreviations
(NOT ``Team.id``, which is assigned at seed time) and NO ``result`` column (the
upsert computes ``result = home_score - away_score`` itself).
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from pathlib import Path

from app.seeds.historical_games import ARTIFACT_PATH, NFLVERSE_ABBR_TO_ESPN

# Pinned nflverse raw URL (same source the spike used).
RAW_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

# The exact trimmed header we write (and the upsert / tests assert against).
ARTIFACT_HEADER: tuple[str, ...] = (
    "nflverse_game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "spread_line",
    "total_line",
)


def _fetch_rows(url: str = RAW_URL) -> list[dict[str, str]]:
    """Fetch and parse the nflverse games.csv over HTTPS into string-dict rows."""
    sys.stderr.write(f"fetching {url}\n")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (pinned HTTPS nflverse URL)
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(text.splitlines()))


def _completed_seasons(rows: list[dict[str, str]]) -> set[int]:
    """Seasons that contain a FINAL Super Bowl row (game_type == "SB" with scores).

    Used as the "through the last completed season" bound (locked decision 4):
    a season is only kept if its Super Bowl has been played, which auto-excludes
    the in-progress current season the ESPN poller owns — no manual constant to
    bump each year.
    """
    completed: set[int] = set()
    for r in rows:
        if r.get("game_type") == "SB" and r.get("home_score") and r.get("away_score"):
            try:
                completed.add(int(r["season"]))
            except ValueError:
                continue
    return completed


def build_artifact_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Trim/validate the raw nflverse rows into the committed artifact rows.

    Keeps only final games (both scores present) with a posted ``spread_line``,
    season >= 1999, and only seasons whose Super Bowl has been played. Validates
    every home/away abbreviation against :data:`NFLVERSE_ABBR_TO_ESPN` and raises
    loudly listing any unmapped ones. Sorted by ``(season, week, gameday)`` for a
    stable diff.
    """
    completed = _completed_seasons(raw_rows)

    kept: list[dict[str, str]] = []
    unmapped: set[str] = set()
    for r in raw_rows:
        if not r.get("home_score") or not r.get("away_score"):
            continue
        if not r.get("spread_line"):
            continue
        try:
            season = int(r["season"])
        except KeyError, ValueError:
            continue
        if season < 1999 or season not in completed:
            continue

        home, away = r["home_team"], r["away_team"]
        for abbr in (home, away):
            if abbr not in NFLVERSE_ABBR_TO_ESPN:
                unmapped.add(abbr)

        kept.append(
            {
                "nflverse_game_id": r["game_id"],
                "season": r["season"],
                "week": r["week"],
                "game_type": r["game_type"],
                "gameday": r["gameday"],
                "home_team": home,
                "away_team": away,
                "home_score": r["home_score"],
                "away_score": r["away_score"],
                "spread_line": r["spread_line"],
                "total_line": r.get("total_line", "") or "",
            }
        )

    if unmapped:
        raise ValueError(
            "Unmapped nflverse team abbreviations (add to NFLVERSE_ABBR_TO_ESPN): "
            + ", ".join(sorted(unmapped))
        )

    kept.sort(key=lambda g: (int(g["season"]), int(g["week"]), g["gameday"]))
    return kept


def write_artifact(rows: list[dict[str, str]], path: Path = ARTIFACT_PATH) -> None:
    """Write the trimmed artifact CSV with the fixed :data:`ARTIFACT_HEADER`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ARTIFACT_HEADER))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Fetch nflverse, build the artifact, write it, print a summary.

    The one-command regeneration path::

        cd backend && .venv/bin/python -m app.seeds.historical_gen

    Run out-of-band once per season. The written ``spread_line`` is a consensus
    number (not strictly the closing line) and there is no moneyline.
    """
    raw = _fetch_rows()
    rows = build_artifact_rows(raw)
    write_artifact(rows)
    seasons = sorted({int(r["season"]) for r in rows})
    span = f"{seasons[0]}-{seasons[-1]}" if seasons else "(none)"
    print(f"Wrote {len(rows)} historical games ({span}) to {ARTIFACT_PATH}.")


if __name__ == "__main__":
    main()
