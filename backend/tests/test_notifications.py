"""Offline unit tests for the best-effort Discord event publisher (QT-1).

These tests NEVER touch a live Redis. The publisher exposes a small
``_redis_client()`` factory seam that each test monkeypatches:

* success case  -> a fake client records (channel, payload) so we can assert the
  channel name is ``pickem:events`` and that the JSON payload round-trips back to
  the original event dict;
* Redis-down    -> the factory (or its ``publish``) raises, and we assert
  ``publish_event`` swallows the error (returns normally, never propagates).

Run with: ``backend/.venv/bin/python -m unittest tests.test_notifications -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from app.models import PickType
from app.services import notifications
from app.services.notifications import (
    EVENTS_CHANNEL,
    admin_pick_cleared_event,
    admin_pick_set_event,
    claim_cooldown,
    freeze_week_event,
    game_final_event,
    ingest_season_event,
    login_event,
    misc_graded_event,
    misc_picked_event,
    pick_cleared_event,
    pick_event,
    GameFinalImpact,
    pick_log_detail,
    player_registered_event,
    publish_event,
    roster_complete_event,
    to_game_final_impacts,
    week_recap_event,
    window_closed_event,
    window_opened_event,
)


class LoginEventBuilderTests(unittest.TestCase):
    def test_login_event_exact_shape(self) -> None:
        event = login_event("alice")
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["type"], "user.login")
        self.assertEqual(event["targets"], ["logger"])
        self.assertEqual(event["actor"], "alice")
        # No extra/sensitive fields leak into the v1 payload.
        self.assertEqual(set(event.keys()), {"v", "type", "targets", "actor"})

    def test_login_event_is_pure(self) -> None:
        # Distinct display names produce distinct actors; same input is stable.
        self.assertEqual(login_event("bob")["actor"], "bob")
        self.assertEqual(login_event("alice"), login_event("alice"))


class PickLogDetailTests(unittest.TestCase):
    """The pure side/team resolver — every branch, offline (no DB)."""

    def test_favorite_cover_uses_favorite_abbr(self) -> None:
        detail = pick_log_detail(
            PickType.FAVORITE_COVER,
            False,
            None,
            favorite_abbr="KC",
            underdog_abbr="LAR",
            home_abbr="KC",
            away_abbr="LAR",
        )
        self.assertEqual(detail, "Favorite (KC)")

    def test_underdog_cover_uses_underdog_abbr(self) -> None:
        detail = pick_log_detail(
            PickType.UNDERDOG_COVER,
            False,
            None,
            favorite_abbr="KC",
            underdog_abbr="LAR",
            home_abbr="KC",
            away_abbr="LAR",
        )
        self.assertEqual(detail, "Underdog (LAR)")

    def test_over_uses_matchup(self) -> None:
        detail = pick_log_detail(
            PickType.OVER,
            False,
            None,
            favorite_abbr="KC",
            underdog_abbr="LAR",
            home_abbr="KC",
            away_abbr="LAR",
        )
        self.assertEqual(detail, "OVER LAR@KC")

    def test_under_uses_matchup(self) -> None:
        detail = pick_log_detail(
            PickType.UNDER,
            False,
            None,
            favorite_abbr="KC",
            underdog_abbr="LAR",
            home_abbr="KC",
            away_abbr="LAR",
        )
        self.assertEqual(detail, "UNDER LAR@KC")

    def test_misc_uses_misc_text(self) -> None:
        detail = pick_log_detail(
            PickType.MISC,
            False,
            "Mahomes throws 3 TDs",
            favorite_abbr="KC",
            underdog_abbr="LAR",
            home_abbr="KC",
            away_abbr="LAR",
        )
        self.assertEqual(detail, "Mahomes throws 3 TDs")

    def test_mortal_lock_annotates_detail(self) -> None:
        detail = pick_log_detail(
            PickType.FAVORITE_COVER,
            True,
            None,
            favorite_abbr="KC",
            underdog_abbr="LAR",
            home_abbr="KC",
            away_abbr="LAR",
        )
        self.assertEqual(detail, "Favorite (KC) (ML)")

    def test_over_mortal_lock_annotates_detail(self) -> None:
        detail = pick_log_detail(
            PickType.OVER,
            True,
            None,
            favorite_abbr="KC",
            underdog_abbr="LAR",
            home_abbr="KC",
            away_abbr="LAR",
        )
        self.assertEqual(detail, "OVER LAR@KC (ML)")

    def test_favorite_abbr_missing_falls_back_to_label(self) -> None:
        # A true pick'em has no favorite/underdog abbr — resolver must not crash.
        detail = pick_log_detail(
            PickType.FAVORITE_COVER,
            False,
            None,
            favorite_abbr=None,
            underdog_abbr=None,
            home_abbr="KC",
            away_abbr="LAR",
        )
        self.assertEqual(detail, "Favorite")


_PICK_EVENT_KEYS = {"v", "type", "targets", "actor", "week", "detail"}


class PickEventBuilderTests(unittest.TestCase):
    def test_pick_created_shape(self) -> None:
        event = pick_event("pick.created", actor="bob", week=3, detail="OVER KC")
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["type"], "pick.created")
        self.assertEqual(event["targets"], ["logger"])
        self.assertEqual(event["actor"], "bob")
        self.assertEqual(event["week"], 3)
        self.assertEqual(event["detail"], "OVER KC")
        self.assertEqual(set(event.keys()), _PICK_EVENT_KEYS)

    def test_pick_changed_shape(self) -> None:
        event = pick_event("pick.changed", actor="bob", week=3, detail="OVER KC")
        self.assertEqual(event["type"], "pick.changed")
        self.assertEqual(set(event.keys()), _PICK_EVENT_KEYS)

    def test_pick_cleared_shape(self) -> None:
        event = pick_cleared_event(actor="bob", week=3, detail="OVER KC")
        self.assertEqual(event["type"], "pick.cleared")
        self.assertEqual(event["targets"], ["logger"])
        self.assertEqual(event["actor"], "bob")
        self.assertEqual(event["week"], 3)
        self.assertEqual(event["detail"], "OVER KC")
        self.assertEqual(set(event.keys()), _PICK_EVENT_KEYS)


class AdminPickEventBuilderTests(unittest.TestCase):
    def test_admin_pick_set_shape(self) -> None:
        event = admin_pick_set_event(target="alice", week=3, detail="Favorite (KC)")
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["type"], "admin.pick_set")
        self.assertEqual(event["targets"], ["logger"])
        self.assertEqual(event["target"], "alice")
        self.assertEqual(event["week"], 3)
        self.assertEqual(event["detail"], "Favorite (KC)")
        self.assertEqual(set(event.keys()), {"v", "type", "targets", "target", "week", "detail"})

    def test_admin_pick_cleared_shape(self) -> None:
        event = admin_pick_cleared_event(target="alice", week=3, slot="FAVORITE_COVER")
        self.assertEqual(event["type"], "admin.pick_cleared")
        self.assertEqual(event["targets"], ["logger"])
        self.assertEqual(event["target"], "alice")
        self.assertEqual(event["week"], 3)
        self.assertEqual(event["slot"], "FAVORITE_COVER")
        self.assertEqual(set(event.keys()), {"v", "type", "targets", "target", "week", "slot"})


class PlayerRegisteredEventBuilderTests(unittest.TestCase):
    def test_player_registered_shape(self) -> None:
        event = player_registered_event("newbie")
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["type"], "player.registered")
        self.assertEqual(event["targets"], ["logger"])
        self.assertEqual(event["actor"], "newbie")
        # HARD RULE: display_name ONLY — exactly these keys, never a password/token/email.
        self.assertEqual(set(event.keys()), {"v", "type", "targets", "actor"})

    def test_player_registered_carries_no_password(self) -> None:
        event = player_registered_event("newbie")
        for forbidden in ("password", "plain_password", "token", "email", "secret"):
            self.assertNotIn(forbidden, event)


class OpsEventBuilderTests(unittest.TestCase):
    def test_ingest_season_shape(self) -> None:
        event = ingest_season_event(season=2026, weeks=18, games=272, failed=1)
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["type"], "ingest.season")
        self.assertEqual(event["targets"], ["logger"])
        self.assertEqual(event["season"], 2026)
        self.assertEqual(event["weeks"], 18)
        self.assertEqual(event["games"], 272)
        self.assertEqual(event["failed"], 1)
        self.assertEqual(
            set(event.keys()),
            {"v", "type", "targets", "season", "weeks", "games", "failed"},
        )

    def test_freeze_week_shape(self) -> None:
        event = freeze_week_event(week=3)
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["type"], "freeze.week")
        self.assertEqual(event["targets"], ["logger"])
        self.assertEqual(event["week"], 3)
        self.assertEqual(set(event.keys()), {"v", "type", "targets", "week"})


class AllBuildersTargetLoggerTests(unittest.TestCase):
    def test_every_new_builder_targets_logger_only(self) -> None:
        events = [
            pick_event("pick.created", actor="a", week=1, detail="d"),
            pick_event("pick.changed", actor="a", week=1, detail="d"),
            pick_cleared_event(actor="a", week=1, detail="d"),
            admin_pick_set_event(target="a", week=1, detail="d"),
            admin_pick_cleared_event(target="a", week=1, slot="OVER"),
            player_registered_event("a"),
            ingest_season_event(season=2026, weeks=1, games=1, failed=0),
            freeze_week_event(week=1),
        ]
        for event in events:
            self.assertEqual(event["targets"], ["logger"])
            self.assertEqual(event["v"], 1)


# --------------------------------------------------------------------------- #
# QT-3 — five player-facing pickem-CHAT event builders (targets ["chat"]).
# --------------------------------------------------------------------------- #


class ChatEventBuilderTests(unittest.TestCase):
    """Each QT-3 builder targets EXACTLY ["chat"] and carries DISPLAY data only."""

    def test_roster_complete_shape(self) -> None:
        event = roster_complete_event(actor="bob", week=3)
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["type"], "roster.complete")
        self.assertEqual(event["targets"], ["chat"])
        self.assertEqual(event["actor"], "bob")
        self.assertEqual(event["week"], 3)
        self.assertEqual(set(event.keys()), {"v", "type", "targets", "actor", "week"})

    def test_window_opened_shape(self) -> None:
        event = window_opened_event(week=3)
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["type"], "window.opened")
        self.assertEqual(event["targets"], ["chat"])
        self.assertEqual(event["week"], 3)
        self.assertEqual(set(event.keys()), {"v", "type", "targets", "week"})

    def test_window_closed_shape(self) -> None:
        event = window_closed_event(week=3)
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["type"], "window.closed")
        self.assertEqual(event["targets"], ["chat"])
        self.assertEqual(event["week"], 3)
        self.assertEqual(set(event.keys()), {"v", "type", "targets", "week"})

    def test_game_final_shape(self) -> None:
        event = game_final_event(
            week=3, away_abbr="LAC", home_abbr="KC", away_score=20, home_score=27
        )
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["type"], "game.final")
        self.assertEqual(event["targets"], ["chat"])
        self.assertEqual(event["week"], 3)
        self.assertEqual(event["away"], "LAC")
        self.assertEqual(event["home"], "KC")
        self.assertEqual(event["away_score"], 20)
        self.assertEqual(event["home_score"], 27)
        # Omitting impacts yields an empty list (back-compatible default).
        self.assertEqual(event["impacts"], [])
        self.assertEqual(
            set(event.keys()),
            {
                "v",
                "type",
                "targets",
                "week",
                "away",
                "home",
                "away_score",
                "home_score",
                "impacts",
            },
        )

    def test_game_final_carries_impacts(self) -> None:
        impacts: list[GameFinalImpact] = [
            {"username": "Alice", "outcome": "busted", "was_mortal_lock": True},
            {"username": "Bob", "outcome": "cashed", "was_mortal_lock": False},
        ]
        event = game_final_event(
            week=3,
            away_abbr="LAC",
            home_abbr="KC",
            away_score=20,
            home_score=27,
            impacts=impacts,
        )
        self.assertEqual(event["impacts"], impacts)
        # The whole event must survive a JSON round-trip unchanged (it is
        # json.dumps'd to Redis downstream).
        self.assertEqual(json.loads(json.dumps(event)), event)

    def test_to_game_final_impacts_win_and_loss_mapping(self) -> None:
        result = to_game_final_impacts(
            [
                {"display_name": "Alice", "is_mortal_lock": False, "outcome": "WIN"},
                {"display_name": "Bob", "is_mortal_lock": False, "outcome": "LOSS"},
            ]
        )
        self.assertEqual(
            result,
            [
                {"username": "Alice", "outcome": "cashed", "was_mortal_lock": False},
                {"username": "Bob", "outcome": "busted", "was_mortal_lock": False},
            ],
        )

    def test_to_game_final_impacts_drops_non_point_outcomes(self) -> None:
        for dropped in ("PUSH", "INELIGIBLE", "UNGRADEABLE"):
            result = to_game_final_impacts(
                [{"display_name": "Alice", "is_mortal_lock": False, "outcome": dropped}]
            )
            self.assertEqual(result, [], f"{dropped} should be dropped")

    def test_to_game_final_impacts_preserves_order_and_mortal_lock(self) -> None:
        # Mortal-lock-first order (as the context builder emits it) is preserved,
        # and was_mortal_lock carries through from is_mortal_lock.
        result = to_game_final_impacts(
            [
                {"display_name": "Zed", "is_mortal_lock": True, "outcome": "WIN"},
                {"display_name": "Alice", "is_mortal_lock": False, "outcome": "LOSS"},
            ]
        )
        self.assertEqual(
            result,
            [
                {"username": "Zed", "outcome": "cashed", "was_mortal_lock": True},
                {"username": "Alice", "outcome": "busted", "was_mortal_lock": False},
            ],
        )

    def test_to_game_final_impacts_skips_missing_display_name(self) -> None:
        result = to_game_final_impacts(
            [
                {"display_name": None, "is_mortal_lock": False, "outcome": "WIN"},
                {"display_name": "", "is_mortal_lock": False, "outcome": "LOSS"},
                {"display_name": "Bob", "is_mortal_lock": False, "outcome": "WIN"},
            ]
        )
        self.assertEqual(
            result,
            [{"username": "Bob", "outcome": "cashed", "was_mortal_lock": False}],
        )

    def test_week_recap_shape(self) -> None:
        event = week_recap_event(
            week=3, winner="Carol", winner_score=6, leader="Dave", leader_score=18
        )
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["type"], "week.recap")
        self.assertEqual(event["targets"], ["chat"])
        self.assertEqual(event["week"], 3)
        self.assertEqual(event["winner"], "Carol")
        self.assertEqual(event["winner_score"], 6)
        self.assertEqual(event["leader"], "Dave")
        self.assertEqual(event["leader_score"], 18)
        self.assertEqual(
            set(event.keys()),
            {
                "v",
                "type",
                "targets",
                "week",
                "winner",
                "winner_score",
                "leader",
                "leader_score",
            },
        )

    def test_misc_graded_shape(self) -> None:
        event = misc_graded_event(
            actor="bob",
            week=3,
            prediction="Mahomes throws 4 TDs",
            verdict="correct",
            points=3,
            grader="admin_zach",
        )
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["type"], "misc.graded")
        self.assertEqual(event["targets"], ["chat"])
        self.assertEqual(event["actor"], "bob")
        self.assertEqual(event["week"], 3)
        self.assertEqual(event["prediction"], "Mahomes throws 4 TDs")
        self.assertEqual(event["verdict"], "correct")
        self.assertEqual(event["points"], 3)
        self.assertEqual(event["grader"], "admin_zach")
        self.assertEqual(
            set(event.keys()),
            {"v", "type", "targets", "actor", "week", "prediction", "verdict", "points", "grader"},
        )

    def test_misc_graded_grader_defaults_to_none(self) -> None:
        # grader is optional/back-compatible; omitting it carries None.
        event = misc_graded_event(actor="bob", week=3, prediction="p", verdict="correct", points=3)
        self.assertIsNone(event["grader"])

    def test_misc_graded_carries_negative_points(self) -> None:
        event = misc_graded_event(
            actor="bob", week=4, prediction="a bold call", verdict="incorrect", points=-2
        )
        self.assertEqual(event["verdict"], "incorrect")
        self.assertEqual(event["points"], -2)

    def test_every_chat_builder_targets_chat_only_and_leaks_nothing(self) -> None:
        events = [
            roster_complete_event(actor="a", week=1),
            window_opened_event(week=1),
            window_closed_event(week=1),
            game_final_event(
                week=1,
                away_abbr="A",
                home_abbr="B",
                away_score=0,
                home_score=0,
                impacts=[{"username": "a", "outcome": "cashed", "was_mortal_lock": False}],
            ),
            week_recap_event(week=1, winner="a", winner_score=0, leader="b", leader_score=0),
            misc_graded_event(actor="a", week=1, prediction="p", verdict="correct", points=1),
        ]
        for event in events:
            self.assertEqual(event["targets"], ["chat"])
            self.assertEqual(event["v"], 1)
            # No sensitive/user-identifying fields ever cross into chat.
            for forbidden in (
                "user_id",
                "password",
                "plain_password",
                "token",
                "email",
                "secret",
            ):
                self.assertNotIn(forbidden, event)


class _FakeRedis:
    """Records the last PUBLISH so the test can assert channel + payload."""

    def __init__(self) -> None:
        self.published: tuple[str, str] | None = None

    def publish(self, channel: str, payload: str) -> int:
        self.published = (channel, payload)
        return 1


class _BoomRedis:
    """A client whose publish always raises — the Redis-down path."""

    def publish(self, channel: str, payload: str) -> int:
        raise RuntimeError("redis is down")


class PublishEventTests(unittest.TestCase):
    def test_publish_sends_json_to_events_channel(self) -> None:
        fake = _FakeRedis()
        event = login_event("alice")
        with patch.object(notifications, "_redis_client", return_value=fake):
            self.assertIsNone(publish_event(event))
        self.assertIsNotNone(fake.published)
        channel, payload = fake.published  # type: ignore[misc]
        self.assertEqual(channel, EVENTS_CHANNEL)
        self.assertEqual(channel, "pickem:events")
        # Payload is JSON that round-trips back to the original event.
        self.assertEqual(json.loads(payload), event)

    def test_publish_swallows_redis_publish_error(self) -> None:
        with patch.object(notifications, "_redis_client", return_value=_BoomRedis()):
            # Must NOT raise even though publish() blows up.
            self.assertIsNone(publish_event(login_event("alice")))

    def test_publish_swallows_client_construction_error(self) -> None:
        def _boom() -> object:
            raise ConnectionError("cannot connect to redis")

        with patch.object(notifications, "_redis_client", side_effect=_boom):
            self.assertIsNone(publish_event(login_event("alice")))


# --------------------------------------------------------------------------- #
# claim_cooldown (260628-itg) — SET NX EX dedup helper, fail-open.
# --------------------------------------------------------------------------- #


class _NxFakeRedis:
    """A tiny in-memory client honoring SET NX semantics on a single key set.

    ``set(key, value, nx=True, ex=...)`` returns ``True`` the FIRST time a key is
    seen and ``None`` (redis-py's NX-rejected return) on any repeat — exactly the
    contract :func:`claim_cooldown` relies on.
    """

    def __init__(self) -> None:
        self._keys: set[str] = set()
        self.calls: list[tuple] = []

    def set(self, key, value, nx=False, ex=None):  # noqa: A002
        self.calls.append((key, value, nx, ex))
        if nx and key in self._keys:
            return None
        self._keys.add(key)
        return True


class _BoomSetRedis:
    """A client whose ``set`` always raises — the cooldown FAIL-OPEN path."""

    def set(self, *args, **kwargs):
        raise RuntimeError("redis is down")


class ClaimCooldownTests(unittest.TestCase):
    def test_first_claim_true_repeat_false(self) -> None:
        fake = _NxFakeRedis()
        with patch.object(notifications, "_redis_client", return_value=fake):
            self.assertTrue(claim_cooldown("pickem:cd:1:2:3", 300))
            # Immediate repeat within the window — suppressed.
            self.assertFalse(claim_cooldown("pickem:cd:1:2:3", 300))
        # A DIFFERENT key is independently claimable (still True).
        with patch.object(notifications, "_redis_client", return_value=fake):
            self.assertTrue(claim_cooldown("pickem:cd:9:9:9", 300))
        # It ran SET NX EX with the supplied ttl.
        self.assertTrue(all(c[2] is True and c[3] == 300 for c in fake.calls))

    def test_fail_open_when_set_raises(self) -> None:
        with patch.object(notifications, "_redis_client", return_value=_BoomSetRedis()):
            # FAIL-OPEN: returns True, never raises.
            self.assertTrue(claim_cooldown("pickem:cd:k", 300))

    def test_fail_open_when_client_construction_raises(self) -> None:
        def _boom() -> object:
            raise ConnectionError("cannot connect to redis")

        with patch.object(notifications, "_redis_client", side_effect=_boom):
            self.assertTrue(claim_cooldown("pickem:cd:k", 300))


# --------------------------------------------------------------------------- #
# misc.picked (260628-itg) — leak-safe chat event: actor + week ONLY.
# --------------------------------------------------------------------------- #


class MiscPickedEventBuilderTests(unittest.TestCase):
    def test_misc_picked_exact_shape(self) -> None:
        event = misc_picked_event(actor="bob", week=3)
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["type"], "misc.picked")
        self.assertEqual(event["targets"], ["chat"])
        self.assertEqual(event["actor"], "bob")
        self.assertEqual(event["week"], 3)
        # Exactly these keys — no user_id, no misc_text.
        self.assertEqual(set(event.keys()), {"v", "type", "targets", "actor", "week"})

    def test_misc_picked_carries_no_user_id_or_misc_text(self) -> None:
        secret_text = "Mahomes throws 5 TDs and it rains"
        event = misc_picked_event(actor="bob", week=7)
        for forbidden in ("user_id", "misc_text", "prediction", "detail"):
            self.assertNotIn(forbidden, event)
        # The actual prediction text appears NOWHERE in the event (LEAK-SAFE).
        self.assertNotIn(secret_text, repr(event))
        for value in event.values():
            self.assertNotEqual(value, secret_text)


if __name__ == "__main__":
    unittest.main()
