"""Proactive injury alerts (issue #248 item 7).

A beat task reads ESPN's injury report for each game of the current week that has not
kicked off, compares it with the last report it saw (a Redis snapshot per game), and
publishes an ``injury.change`` chat event when a relevant player's status gets worse:
newly Doubtful, or newly Out.

Relevant players: every QB; an RB, WR or TE only when the game is heavily picked, which
the target read reports only after the week's pick window closed. The first read of a
game only stores the snapshot, so turning the feature on never floods the channel.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from app.services import espn_extra
from app.services.notifications import injury_change_event

logger = structlog.get_logger(__name__)

MAX_EVENTS_PER_RUN = 5
SNAPSHOT_TTL_SECONDS = 10 * 24 * 3600
_ALWAYS_POSITIONS = frozenset({"QB"})
_HEAVY_GAME_POSITIONS = frozenset({"RB", "WR", "TE"})
_SEVERITY = {"questionable": 1, "doubtful": 2, "out": 3}
_ALERT_SEVERITY = 2


def _severity(status: object) -> int:
    return _SEVERITY.get(status.strip().lower(), 0) if isinstance(status, str) else 0


def snapshot_key(event_id: int) -> str:
    return f"pickem:injury_watch:{event_id}"


def diff_reports(
    previous: dict[str, dict[str, str | None]], current: dict[str, list[dict]]
) -> list[dict]:
    """Every player whose status is now Doubtful or Out and worse than before.

    ``previous`` maps team -> player -> status. A team missing from ``previous`` is
    new to the watcher, so its report seeds and reports nothing.
    """
    changes: list[dict] = []
    for team, players in current.items():
        before = previous.get(team)
        if before is None:
            continue
        for player in players:
            name, status = player.get("display_name"), player.get("status")
            if not name or not isinstance(status, str):
                continue
            old = before.get(name)
            if _severity(status) >= _ALERT_SEVERITY and _severity(status) > _severity(old):
                changes.append({**player, "team": team, "old_status": old})
    return changes


def is_relevant(change: dict, *, heavy: bool) -> bool:
    position = (change.get("position") or "").upper()
    return position in _ALWAYS_POSITIONS or (heavy and position in _HEAVY_GAME_POSITIONS)


def _report_by_team(payload: Any, teams: tuple[str, str]) -> dict[str, list[dict]]:
    report: dict[str, list[dict]] = {}
    for team in teams:
        players = espn_extra.parse_injuries(payload, team)
        if players is not None:
            report[team] = players
    return report


def _as_snapshot(report: dict[str, list[dict]]) -> dict[str, dict[str, str | None]]:
    return {
        team: {p["display_name"]: p.get("status") for p in players if p.get("display_name")}
        for team, players in report.items()
    }


async def collect_changes(
    targets: list[dict],
    *,
    read_snapshot: Callable[[int], dict | None],
    write_snapshot: Callable[[int, dict], None],
    fetch: Callable[[int], Awaitable[dict | None]] = espn_extra.fetch_injuries,
) -> list[dict]:
    """The ``injury.change`` events for ``targets``, at most :data:`MAX_EVENTS_PER_RUN`.

    ``read_snapshot`` raises when the store cannot be read; that game is then skipped,
    never treated as new, so a store outage cannot flood the channel either.
    """
    events: list[dict] = []
    for target in targets:
        event_id = target["event_id"]
        payload = await fetch(event_id)
        if payload is None:
            continue
        report = _report_by_team(payload, (target["home"], target["away"]))
        if not report:
            continue
        try:
            previous = read_snapshot(event_id)
        except Exception:
            logger.warning("injury_watch_snapshot_read_failed", event_id=event_id)
            continue
        if previous is not None:
            for change in diff_reports(previous, report):
                if not is_relevant(change, heavy=target["heavy"]):
                    continue
                home = change["team"] == target["home"]
                events.append(
                    injury_change_event(
                        week=target["week"],
                        team=change["team"],
                        opponent=target["away"] if home else target["home"],
                        home=home,
                        player=change["display_name"],
                        position=change.get("position"),
                        old_status=change["old_status"],
                        new_status=change["status"],
                        body_part=change.get("body_part"),
                    )
                )
        write_snapshot(event_id, {**(previous or {}), **_as_snapshot(report)})
    if len(events) > MAX_EVENTS_PER_RUN:
        logger.info("injury_watch_events_capped", found=len(events), kept=MAX_EVENTS_PER_RUN)
    return events[:MAX_EVENTS_PER_RUN]


def redis_snapshot_store(
    client,
) -> tuple[Callable[[int], dict | None], Callable[[int, dict], None]]:
    """Read/write callables over a sync Redis client. A failed write is logged only."""

    def read(event_id: int) -> dict | None:
        raw = client.get(snapshot_key(event_id))
        return json.loads(raw) if raw else None

    def write(event_id: int, snapshot: dict) -> None:
        try:
            client.set(snapshot_key(event_id), json.dumps(snapshot), ex=SNAPSHOT_TTL_SECONDS)
        except Exception:
            logger.warning("injury_watch_snapshot_write_failed", event_id=event_id)

    return read, write
