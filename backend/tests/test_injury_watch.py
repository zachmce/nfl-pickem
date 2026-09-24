"""Offline tests for the proactive injury alerts (issue #248 item 7).

Run with: ``backend/.venv/bin/python -m unittest tests.test_injury_watch -v``
"""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import mock

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.bot import chat_personality
from app.bot.notifier import render_chat
from app.models import Game, GameStatus, Pick, PickType, Team, User, Week
from app.services import injury_watch
from app.services.auth import hash_password
from app.services.notifications import injury_change_event
from app.services.notifications_read import get_injury_watch_targets


def _player(name: str, status: str, position: str = "QB") -> dict:
    return {
        "display_name": name,
        "status": status,
        "position": position,
        "body_part": "Ankle",
        "return_date": None,
        "date": "2026-09-24",
    }


def _summary(home: list[dict], away: list[dict]) -> dict:
    return {
        "injuries": [
            {"team": {"abbreviation": "BUF"}, "injuries": [_raw(p) for p in home]},
            {"team": {"abbreviation": "MIA"}, "injuries": [_raw(p) for p in away]},
        ]
    }


def _raw(p: dict) -> dict:
    return {
        "status": p["status"],
        "athlete": {"displayName": p["display_name"], "position": {"abbreviation": p["position"]}},
        "details": {"type": p["body_part"]},
    }


_TARGET = {"event_id": 7, "week": 4, "home": "BUF", "away": "MIA", "heavy": False}


class _Store:
    def __init__(self, snapshot: dict | None = None, *, broken: bool = False) -> None:
        self.data: dict[int, dict] = {} if snapshot is None else {7: snapshot}
        self.broken = broken

    def read(self, event_id: int) -> dict | None:
        if self.broken:
            raise ConnectionError("redis down")
        return self.data.get(event_id)

    def write(self, event_id: int, snapshot: dict) -> None:
        self.data[event_id] = snapshot


def _collect(store: _Store, payload: dict | None, targets=None) -> list[dict]:
    async def _fetch(event_id):
        return payload

    return asyncio.run(
        injury_watch.collect_changes(
            targets or [_TARGET],
            read_snapshot=store.read,
            write_snapshot=store.write,
            fetch=_fetch,
        )
    )


class DiffTests(unittest.TestCase):
    def test_only_a_worse_status_at_doubtful_or_out_counts(self) -> None:
        previous = {"BUF": {"A": "Questionable", "B": "Out", "C": "Doubtful", "D": None}}
        current = {
            "BUF": [
                _player("A", "Out"),  # Questionable -> Out: alert
                _player("B", "Out"),  # unchanged
                _player("C", "Questionable"),  # better
                _player("E", "Doubtful"),  # new to the report: alert
                _player("F", "Questionable"),  # new, but only Questionable
            ]
        }
        changes = injury_watch.diff_reports(previous, current)
        self.assertEqual(
            [(c["display_name"], c["old_status"]) for c in changes],
            [("A", "Questionable"), ("E", None)],
        )

    def test_a_team_new_to_the_snapshot_only_seeds(self) -> None:
        self.assertEqual(injury_watch.diff_reports({}, {"BUF": [_player("A", "Out")]}), [])

    def test_qbs_always_count_skill_players_only_in_a_heavy_game(self) -> None:
        self.assertTrue(injury_watch.is_relevant({"position": "QB"}, heavy=False))
        self.assertFalse(injury_watch.is_relevant({"position": "WR"}, heavy=False))
        self.assertTrue(injury_watch.is_relevant({"position": "wr"}, heavy=True))
        self.assertFalse(injury_watch.is_relevant({"position": "LB"}, heavy=True))
        self.assertFalse(injury_watch.is_relevant({"position": None}, heavy=True))


class CollectTests(unittest.TestCase):
    def test_the_first_read_seeds_and_posts_nothing(self) -> None:
        store = _Store()
        events = _collect(store, _summary([_player("Josh Allen", "Out")], []))
        self.assertEqual(events, [])
        self.assertEqual(store.data[7], {"BUF": {"Josh Allen": "Out"}, "MIA": {}})

    def test_a_worse_qb_status_is_one_event_from_the_team_side(self) -> None:
        store = _Store({"BUF": {}, "MIA": {"Tua": "Questionable"}})
        events = _collect(store, _summary([], [_player("Tua", "Out")]))
        self.assertEqual(
            events,
            [
                injury_change_event(
                    week=4,
                    team="MIA",
                    opponent="BUF",
                    home=False,
                    player="Tua",
                    position="QB",
                    old_status="Questionable",
                    new_status="Out",
                    body_part="Ankle",
                )
            ],
        )
        self.assertEqual(store.data[7]["MIA"], {"Tua": "Out"})
        # The same report again posts nothing.
        self.assertEqual(_collect(store, _summary([], [_player("Tua", "Out")])), [])

    def test_a_skill_player_needs_a_heavy_game(self) -> None:
        payload = _summary([_player("Cook", "Out", "RB")], [])
        light = _collect(_Store({"BUF": {}, "MIA": {}}), payload)
        heavy = _collect(_Store({"BUF": {}, "MIA": {}}), payload, [{**_TARGET, "heavy": True}])
        self.assertEqual(light, [])
        self.assertEqual([e["player"] for e in heavy], ["Cook"])

    def test_events_are_capped_per_run(self) -> None:
        many = [_player(f"QB{i}", "Out") for i in range(9)]
        events = _collect(_Store({"BUF": {}, "MIA": {}}), _summary(many, []))
        self.assertEqual(len(events), injury_watch.MAX_EVENTS_PER_RUN)

    def test_a_failed_read_or_fetch_posts_and_writes_nothing(self) -> None:
        broken = _Store({"BUF": {}}, broken=True)
        self.assertEqual(_collect(broken, _summary([_player("A", "Out")], [])), [])
        self.assertEqual(broken.data[7], {"BUF": {}})
        store = _Store({"BUF": {"A": "Questionable"}})
        self.assertEqual(_collect(store, None), [])
        self.assertEqual(store.data[7], {"BUF": {"A": "Questionable"}})

    def test_the_redis_store_round_trips_with_a_ttl(self) -> None:
        client = mock.Mock()
        client.get.return_value = b'{"BUF": {"A": "Out"}}'
        read, write = injury_watch.redis_snapshot_store(client)
        self.assertEqual(read(7), {"BUF": {"A": "Out"}})
        write(7, {"BUF": {}})
        client.set.assert_called_once_with(
            "pickem:injury_watch:7", '{"BUF": {}}', ex=injury_watch.SNAPSHOT_TTL_SECONDS
        )
        client.set.side_effect = ConnectionError("down")
        write(7, {})  # logged, never raised


class WatchTargetTests(unittest.TestCase):
    """The pick count reaches the watcher only once the week's window has closed."""

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        SQLModel.metadata.create_all(self.engine)
        self.now = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
        with Session(self.engine) as session:
            teams = [
                Team(espn_team_id=i, abbreviation=a, display_name=a)
                for i, a in enumerate(("BUF", "MIA", "KC", "DEN", "SF", "LAR"), start=1)
            ]
            session.add_all(teams)
            week = Week(season=2026, week=4)
            session.add(week)
            session.commit()
            tid = {t.abbreviation: t.id for t in teams if t.id is not None}
            week_id = week.id
            assert week_id is not None

            def game(eid: int, home: str, away: str, kickoff: datetime) -> Game:
                return Game(
                    espn_event_id=eid,
                    week_id=week_id,
                    season=2026,
                    week=4,
                    home_team_id=tid[home],
                    away_team_id=tid[away],
                    kickoff_at=kickoff,
                    status=GameStatus.SCHEDULED,
                    spread=Decimal("3.0"),
                    favorite_team_id=tid[home],
                    underdog_team_id=tid[away],
                )

            first = game(1, "BUF", "MIA", self.now + timedelta(hours=2))
            second = game(2, "KC", "DEN", self.now + timedelta(days=3))
            third = game(3, "SF", "LAR", self.now + timedelta(days=3))
            session.add_all([first, second, third])
            pw = hash_password("correct horse battery staple")
            users = [
                User(display_name=f"u{i}", password_hash=pw, is_active=True, discord_id=i + 1)
                for i in range(3)
            ]
            session.add_all(users)
            session.commit()
            for u in users:
                assert u.id is not None and second.id is not None and third.id is not None
                session.add(
                    Pick(user_id=u.id, game_id=second.id, week_id=week_id, pick_type=PickType.OVER)
                )
            assert users[0].id is not None and third.id is not None
            session.add(
                Pick(
                    user_id=users[0].id,
                    game_id=third.id,
                    week_id=week_id,
                    pick_type=PickType.UNDER,
                )
            )
            session.commit()

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_no_game_is_heavy_while_the_window_is_open(self) -> None:
        with Session(self.engine) as session:
            targets = get_injury_watch_targets(session, 2026, 4, now=self.now)
        self.assertEqual([t["event_id"] for t in targets], [1, 2, 3])
        self.assertEqual({t["heavy"] for t in targets}, {False})

    def test_after_the_close_only_the_heavily_picked_game_is_heavy(self) -> None:
        later = self.now + timedelta(hours=3)  # the first kickoff closed the window
        with Session(self.engine) as session:
            targets = get_injury_watch_targets(session, 2026, 4, now=later)
        # The first game has kicked off and drops out; 3 of 3 pickers took game 2.
        self.assertEqual({t["event_id"]: t["heavy"] for t in targets}, {2: True, 3: False})

    def test_a_kickoff_more_than_a_week_out_is_not_watched(self) -> None:
        early = self.now - timedelta(days=5)
        with Session(self.engine) as session:
            targets = get_injury_watch_targets(session, 2026, 4, now=early)
        self.assertEqual([t["event_id"] for t in targets], [1])


class InjuryLineTests(unittest.TestCase):
    event = injury_change_event(
        week=4,
        team="MIA",
        opponent="BUF",
        home=False,
        player="Tua Tagovailoa",
        position="QB",
        old_status=None,
        new_status="Out",
        body_part=None,
    )

    def test_the_fallback_line_states_the_status_and_the_game(self) -> None:
        self.assertEqual(
            render_chat(self.event),
            "Injury update: Tua Tagovailoa (QB, MIA) is now listed Out for Week 4 at BUF.",
        )

    def test_the_fact_states_every_clause_even_when_a_field_is_empty(self) -> None:
        fact = chat_personality._basic_injury_change_fact(self.event)
        self.assertIn("as Out for the MIA Week 4 game on the road against BUF", fact)
        self.assertIn("He was not on the report before.", fact)
        self.assertIn("The report names no body part.", fact)
        self.assertIn("injury.change", chat_personality._HANDLED_TYPES)
        self.assertIn(
            "never change or soften the status word".lower(),
            chat_personality._INJURY_CHANGE_ROLE.lower(),
        )


class TaskTests(unittest.TestCase):
    def test_the_task_does_nothing_while_the_setting_is_off(self) -> None:
        from app import tasks
        from app.config import settings

        with mock.patch.object(settings, "injury_alerts_enabled", False):
            self.assertEqual(tasks.watch_injuries_task(), {"enabled": False})

    def test_the_beat_runs_the_watcher_every_fifteen_minutes(self) -> None:
        from app.celery_app import celery_app

        self.assertEqual(
            celery_app.conf.beat_schedule["injury-watch-poller"],
            {"task": "app.tasks.watch_injuries", "schedule": 900.0},
        )


if __name__ == "__main__":
    unittest.main()
