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

from app.services import notifications
from app.services.notifications import EVENTS_CHANNEL, login_event, publish_event


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


if __name__ == "__main__":
    unittest.main()
