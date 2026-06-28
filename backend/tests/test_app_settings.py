"""Offline service-layer tests for the keyed app-settings store (260627-xbb).

Fully OFFLINE (in-memory SQLite, no Postgres, no network). They pin the generic
upsert contract and the bot-personality wrappers:

* ``set_setting`` INSERTS when absent then UPDATES in place (one row per key);
* ``get_setting`` returns the stored value or ``None``;
* ``get_bot_personality`` returns the sarcastic DEFAULT when unset/blank;
* ``set_bot_personality`` upserts a valid id and rejects an unknown id with a
  ``ValueError`` whose leading token is the stable code ``unknown_personality``.

Run from backend/ with ``.venv/bin/python -m unittest`` (unittest, NOT pytest).
"""

from __future__ import annotations

import unittest

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.bot.personality import DEFAULT_PERSONALITY_ID, available_personality_ids
from app.models import AppSetting
from app.services import app_settings


class AppSettingsServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _session(self) -> Session:
        return Session(self.engine)

    # -- generic upsert ----------------------------------------------------

    def test_get_setting_returns_none_when_absent(self) -> None:
        with self._session() as session:
            self.assertIsNone(app_settings.get_setting(session, "nope"))

    def test_set_setting_inserts_then_updates_single_row(self) -> None:
        with self._session() as session:
            app_settings.set_setting(session, "k", "v1")
            self.assertEqual(app_settings.get_setting(session, "k"), "v1")
            # Update in place — no second row for the same key.
            app_settings.set_setting(session, "k", "v2")
            self.assertEqual(app_settings.get_setting(session, "k"), "v2")
            rows = session.exec(
                select(AppSetting).where(AppSetting.setting_key == "k")
            ).all()
            self.assertEqual(len(rows), 1, "upsert must keep exactly one row per key")

    # -- bot-personality wrappers -----------------------------------------

    def test_get_bot_personality_defaults_to_sarcastic_when_unset(self) -> None:
        with self._session() as session:
            self.assertEqual(
                app_settings.get_bot_personality(session), DEFAULT_PERSONALITY_ID
            )

    def test_get_bot_personality_defaults_when_blank(self) -> None:
        with self._session() as session:
            app_settings.set_setting(
                session, app_settings.BOT_PERSONALITY_KEY, "   "
            )
            self.assertEqual(
                app_settings.get_bot_personality(session), DEFAULT_PERSONALITY_ID
            )

    def test_set_bot_personality_persists_valid_id(self) -> None:
        # Pick a non-default valid id so the round-trip is meaningful.
        target = next(
            pid for pid in available_personality_ids() if pid != DEFAULT_PERSONALITY_ID
        )
        with self._session() as session:
            returned = app_settings.set_bot_personality(session, target)
            self.assertEqual(returned, target)
            self.assertEqual(app_settings.get_bot_personality(session), target)

    def test_set_bot_personality_rejects_unknown_id_with_stable_code(self) -> None:
        with self._session() as session:
            with self.assertRaises(ValueError) as ctx:
                app_settings.set_bot_personality(session, "totally_made_up")
            self.assertTrue(str(ctx.exception).startswith("unknown_personality"))
            # The rejected write left no setting behind -> still the default.
            self.assertEqual(
                app_settings.get_bot_personality(session), DEFAULT_PERSONALITY_ID
            )


if __name__ == "__main__":
    unittest.main()
