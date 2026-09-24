"""Regenerate the static football-history corpus from Wikipedia tables (issue #248).

Writes three CSV files under ``app/seeds/data/history/``: every league champion since
1920 (Super Bowls included), the Pro Football Hall of Fame inductees, and the AP award
winners. Run it once a year, after the NFL Honors and the Hall of Fame class vote:

    uv run --with pandas --with lxml --with beautifulsoup4 --with html5lib \
        python scripts/gen_history_corpus.py

Then review the diff and commit it. The bot reads only the committed files; it never
calls Wikipedia. Source text is CC BY-SA 4.0; each file names its source pages.
"""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "app" / "seeds" / "data" / "history"
API = "https://en.wikipedia.org/w/api.php"
AGENT = {"User-Agent": "nfl-pickem-corpus/1.0 (https://github.com/zachmce/nfl-pickem)"}

BREAK = "¦"
AWARD_PAGES = {
    "mvp": "AP_NFL_Most_Valuable_Player",
    "offensive player of the year": "AP_NFL_Offensive_Player_of_the_Year",
    "defensive player of the year": "AP_NFL_Defensive_Player_of_the_Year",
    "offensive rookie of the year": "AP_NFL_Rookie_of_the_Year",
    "defensive rookie of the year": "AP_NFL_Rookie_of_the_Year",
    "coach of the year": "AP_NFL_Coach_of_the_Year",
    "comeback player of the year": "AP_NFL_Comeback_Player_of_the_Year",
    "super bowl mvp": "Super_Bowl_Most_Valuable_Player_Award",
}


def _html(title: str) -> str:
    query = urllib.parse.urlencode(
        {"action": "parse", "page": title, "prop": "text", "format": "json", "formatversion": 2}
    )
    request = urllib.request.Request(f"{API}?{query}&redirects=1", headers=AGENT)
    with urllib.request.urlopen(request, timeout=30) as response:
        html = json.load(response)["parse"]["text"]
    # Some pages carry rowspan="“2”"; pandas raises on it. A <br> separates co-winners.
    html = re.sub(r"<br\s*/?>", f" {BREAK} ", html, flags=re.I)
    return re.sub(r'(rowspan|colspan)="[^"0-9]*(\d+)[^"0-9]*"', r'\1="\2"', html, flags=re.I)


def _tables(title: str) -> list[pd.DataFrame]:
    return pd.read_html(io.StringIO(_html(title)), flavor="bs4")


def _clean(value: object) -> str:
    """Drop footnotes ``[a]``, win tallies ``(2, 2–0)``, markers and the AFL/NFL suffix."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace(BREAK, " ")
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\(\d+(?:, ?\d+(?:[–-]\d+)?)*\)", "", text)
    text = re.sub(r"[*†‡^§#¤~]+", "", text)
    # The Super Bowl table marks each side's league with a trailing A/N (a/n before 1970).
    text = (
        re.sub(r"(?<=[a-z])[ANan]$", "", text.strip())
        if re.search(r"[a-z]s[ANan]$", text.strip())
        else text
    )
    return " ".join(text.split())


def _split(value: object, count: int | None = None) -> list[str]:
    """A co-winner cell as one value per winner; a single value repeats ``count`` times."""
    parts = (
        [_clean(p) for p in str(value).split(BREAK)] if isinstance(value, str) else [_clean(value)]
    )
    parts = [p for p in parts if p] or [""]
    if count is not None and len(parts) != count:
        return [" ".join(parts) if len(parts) > 1 else parts[0]] * count
    return parts


def _season(value: object) -> int | None:
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def _write(name: str, header: list[str], rows: list[list[object]], sources: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="") as handle:
        handle.write(f"# Source: Wikipedia ({', '.join(sources)}), CC BY-SA 4.0.\n")
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def championships() -> None:
    rows: list[list[object]] = []
    early = _tables("List_of_NFL_champions_(1920–1969)")
    standings = next(t for t in early if "Champion" in t.columns)
    for _, r in standings.iterrows():
        rows.append(
            [
                _season(r["Season"]),
                "NFL title (by standings)",
                "",
                _clean(r["Champion"]),
                "",
                _clean(r["Runner-up"]),
                "",
                "",
                "",
            ]
        )
    games = next(t for t in early if "Winning team" in t.columns)
    for _, r in games.iterrows():
        won, lost = (str(r["Score"]).replace("—", "–").split("–") + [""])[:2]
        rows.append(
            [
                _season(r["Season"]),
                "NFL Championship Game",
                _clean(r["Date"]),
                _clean(r["Winning team"]),
                won.strip(),
                _clean(r["Losing team"]),
                lost.strip(),
                _clean(r["Venue"]),
                _clean(r["City"]),
            ]
        )
    bowls = next(t for t in _tables("List_of_Super_Bowl_champions") if "Winning team" in t.columns)
    for _, r in bowls.iterrows():
        date, _, season = str(r["Date (season)"]).partition("(")
        rows.append(
            [
                _season(season),
                f"Super Bowl {_clean(r['Game'])}",
                _clean(date),
                _clean(r["Winning team"]),
                _clean(r["Score"]),
                _clean(r["Losing team"]),
                _clean(r["Score.1"]),
                _clean(r["Venue"]),
                _clean(r["City"]),
            ]
        )
    rows.sort(key=lambda row: (row[0] or 0, row[1] != "NFL title (by standings)"))
    _write(
        "championships.csv",
        [
            "season",
            "game",
            "date",
            "winner",
            "winner_score",
            "loser",
            "loser_score",
            "venue",
            "city",
        ],
        rows,
        ["List of NFL champions (1920–1969)", "List of Super Bowl champions"],
    )


def hall_of_fame() -> None:
    table = next(
        t
        for t in _tables("List_of_Pro_Football_Hall_of_Fame_inductees")
        if list(t.columns)[:3] == ["Inductee", "Class", "Position"]
    )
    people: dict[tuple[str, int], dict] = {}
    for _, r in table.iterrows():
        raw = str(r["Inductee"])
        key = (_clean(raw), int(r["Class"]))
        person = people.setdefault(
            key, {"position": _clean(r["Position"]), "first_ballot": "**" in raw, "teams": []}
        )
        team = _clean(r["Team(s)"])
        stint = f"{team} ({_clean(r['Years'])})"
        if team and stint not in person["teams"]:
            person["teams"].append(stint)
    rows = [
        [name, year, p["position"], "yes" if p["first_ballot"] else "no", "; ".join(p["teams"])]
        for (name, year), p in sorted(people.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    ]
    _write(
        "hall_of_fame.csv",
        ["name", "class", "position", "first_year_of_eligibility", "teams"],
        rows,
        ["List of Pro Football Hall of Fame inductees"],
    )


def awards() -> None:
    rows: list[list[object]] = []
    cache: dict[str, list[pd.DataFrame]] = {}
    for award, page in AWARD_PAGES.items():
        tables = cache.setdefault(page, _tables(page))
        winners = []
        for t in tables:
            t.columns = [str(c).split("[")[0] for c in t.columns]
            if {"Season", "Year"} & set(t.columns) and {"Player", "Coach", "Winner"} & set(
                t.columns
            ):
                winners.append(t)
        # The rookie page holds the offensive table first, the defensive table second.
        table = winners[1] if award == "defensive rookie of the year" else winners[0]
        for _, r in table.iterrows():
            if "Season" in table.columns:
                season = _season(r["Season"])
            else:
                # The Super Bowl MVP page lists the calendar year of the game.
                year = _season(r["Year"])
                season = year - 1 if year else None
            person = next((r[c] for c in ("Player", "Coach", "Winner") if c in table.columns), "")
            if season is None or not _clean(person):
                continue
            names = _split(person)
            positions = _split(r.get("Position", ""), len(names))
            teams = _split(r.get("Team", ""), len(names))
            for name, position, team in zip(names, positions, teams):
                rows.append([season, award, name, position, team])
    rows.sort(key=lambda row: (row[0], row[1]))
    _write(
        "ap_awards.csv",
        ["season", "award", "winner", "position", "team"],
        rows,
        sorted(set(AWARD_PAGES.values())),
    )


if __name__ == "__main__":
    championships()
    hall_of_fame()
    awards()
