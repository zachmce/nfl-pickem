"""Readers over the static football-history corpus (issue #248).

The CSV files under ``app/seeds/data/history`` come from Wikipedia tables through
``scripts/gen_history_corpus.py`` and are committed. These readers are pure, offline and
never raise: a missing or broken file reads as an empty corpus.
"""

from __future__ import annotations

import csv
import re
from functools import cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent.parent / "seeds" / "data" / "history"

# Each franchise's nicknames across its history: IND finds the Baltimore Colts, TEN
# the Houston Oilers and CHI the 1921 Chicago Staleys.
FRANCHISE_NICKNAMES: dict[str, tuple[str, ...]] = {
    "ARI": ("Cardinals",),
    "ATL": ("Falcons",),
    "BAL": ("Ravens",),
    "BUF": ("Bills",),
    "CAR": ("Panthers",),
    "CHI": ("Bears", "Staleys"),
    "CIN": ("Bengals",),
    "CLE": ("Browns",),
    "DAL": ("Cowboys",),
    "DEN": ("Broncos",),
    "DET": ("Lions", "Portsmouth Spartans"),
    "GB": ("Packers",),
    "HOU": ("Texans",),
    "IND": ("Colts",),
    "JAX": ("Jaguars",),
    "KC": ("Chiefs",),
    "LAC": ("Chargers",),
    "LAR": ("Rams",),
    "LV": ("Raiders",),
    "MIA": ("Dolphins",),
    "MIN": ("Vikings",),
    "NE": ("Patriots",),
    "NO": ("Saints",),
    "NYG": ("Giants",),
    "NYJ": ("Jets",),
    "PHI": ("Eagles",),
    "PIT": ("Steelers",),
    "SEA": ("Seahawks",),
    "SF": ("49ers",),
    "TB": ("Buccaneers",),
    "TEN": ("Titans", "Oilers"),
    "WSH": ("Redskins", "Commanders", "Washington"),
}
# Abbreviations members type that differ from the app's own.
_ABBR_ALIASES = {"WAS": "WSH", "JAC": "JAX", "LA": "LAR", "OAK": "LV", "SD": "LAC", "STL": "LAR"}


@cache
def _rows(name: str) -> tuple[dict[str, str], ...]:
    try:
        with (_DIR / name).open(newline="") as handle:
            lines = (line for line in handle if not line.startswith("#"))
            return tuple(csv.DictReader(lines))
    except OSError, csv.Error:
        return ()


def franchise(team: object) -> str | None:
    """The app's abbreviation for ``team``, or ``None`` when it names no NFL club."""
    if not isinstance(team, str) or not team.strip():
        return None
    token = team.strip().upper()
    token = _ABBR_ALIASES.get(token, token)
    if token in FRANCHISE_NICKNAMES:
        return token
    for abbr, nicknames in FRANCHISE_NICKNAMES.items():
        if any(nick.upper() in token for nick in nicknames):
            return abbr
    return None


def _is_franchise(name: str, abbr: str) -> bool:
    return any(
        re.search(rf"\b{re.escape(nick)}\b", name) for nick in FRANCHISE_NICKNAMES.get(abbr, ())
    )


def championships(*, season: int | None = None, team: str | None = None) -> list[dict]:
    """League title games (and the standings titles of 1920-1932), oldest first."""
    rows = [dict(r) for r in _rows("championships.csv")]
    if season is not None:
        rows = [r for r in rows if r["season"] == str(season)]
    if team is not None:
        rows = [
            r for r in rows if _is_franchise(r["winner"], team) or _is_franchise(r["loser"], team)
        ]
    return rows


def championship_record(team: str) -> dict[str, int]:
    """Titles won and lost by one franchise: Super Bowls and the NFL titles before them."""
    record = {"super_bowls_won": 0, "super_bowls_lost": 0, "nfl_titles_before_1966": 0}
    for r in championships(team=team):
        won = _is_franchise(r["winner"], team)
        if r["game"].startswith("Super Bowl"):
            record["super_bowls_won" if won else "super_bowls_lost"] += 1
        elif won:
            record["nfl_titles_before_1966"] += 1
    return record


def _name_matches(name: str, asked: str) -> bool:
    words = re.findall(r"[a-z0-9]+", asked.casefold())
    have = re.findall(r"[a-z0-9]+", name.casefold())
    return bool(words) and all(w in have for w in words)


def hall_of_fame(
    *, player: str | None = None, team: str | None = None, class_year: int | None = None
) -> list[dict]:
    """Pro Football Hall of Fame inductees, narrowed by name, franchise and class."""
    rows = [dict(r) for r in _rows("hall_of_fame.csv")]
    if player:
        rows = [r for r in rows if _name_matches(r["name"], player)]
    if team is not None:
        rows = [r for r in rows if _is_franchise(r["teams"], team)]
    if class_year is not None:
        rows = [r for r in rows if r["class"] == str(class_year)]
    return rows


AP_AWARDS: tuple[str, ...] = (
    "mvp",
    "super bowl mvp",
    "offensive player of the year",
    "defensive player of the year",
    "offensive rookie of the year",
    "defensive rookie of the year",
    "coach of the year",
    "comeback player of the year",
)


def awards(
    *, season: int | None = None, award: str | None = None, player: str | None = None
) -> list[dict]:
    """AP award winners (Super Bowl MVP included), narrowed by season, award and winner."""
    rows = [dict(r) for r in _rows("ap_awards.csv")]
    if season is not None:
        rows = [r for r in rows if r["season"] == str(season)]
    if award is not None:
        rows = [r for r in rows if r["award"] == award]
    if player:
        rows = [r for r in rows if _name_matches(r["winner"], player)]
    return rows


def latest_season(name: str) -> int | None:
    """The newest season a corpus file holds, so a caller knows where it stops."""
    seasons = [int(r["season"]) for r in _rows(name) if r.get("season", "").isdigit()]
    return max(seasons, default=None)
