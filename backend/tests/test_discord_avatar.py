"""Offline tests for the Discord avatar-hash capture/refresh backend (260629-kdj).

Fully OFFLINE (in-memory SQLite + StaticPool, no Postgres, no network, no Discord
objects). Three concerns are pinned here:

* the ``User.discord_avatar_hash`` column round-trips ``None`` and a string value,
  and ``UserRead`` serializes it (Task 1);
* ``provision_user`` captures an avatar hash inline at register time, and
  ``upsert_avatar_hash_by_discord_id`` updates / clears an existing row and no-ops
  on an unknown discord_id (Task 2);
* migration 0011 emits a PLAIN nullable ADD COLUMN on ``users`` and a reversing
  DROP COLUMN — asserted by inspecting the DDL the migration emits against the
  PostgreSQL dialect (mirrors test_migration_0008_misc.py); the SQLite-only suite
  does NOT run migrations, so the column itself is exercised via the model field.

Run from backend/ with ``.venv/bin/python -m unittest`` (unittest, NOT pytest).
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.models import User
from app.schemas.admin import AdminUserRead
from app.schemas.auth import UserRead
from app.schemas.results import SeasonStandingRow, UserWeekResult
from app.services.auth import provision_user, upsert_avatar_hash_by_discord_id


class DiscordAvatarModelTests(unittest.TestCase):
    """Task 1 — the column round-trips and UserRead exposes it."""

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

    def test_default_avatar_hash_is_none(self) -> None:
        with self._session() as session:
            user = User(discord_id=1, display_name="no_avatar")
            session.add(user)
            session.commit()
            session.refresh(user)
            self.assertIsNone(user.discord_avatar_hash)

    def test_avatar_hash_round_trips_string(self) -> None:
        with self._session() as session:
            user = User(
                discord_id=2,
                display_name="has_avatar",
                discord_avatar_hash="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            self.assertEqual(user.discord_avatar_hash, "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4")

    def test_user_read_serializes_avatar_hash(self) -> None:
        with self._session() as session:
            user = User(
                discord_id=3,
                display_name="serialized",
                discord_avatar_hash="deadbeef",
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            read = UserRead.model_validate(user, from_attributes=True)
            self.assertEqual(read.discord_avatar_hash, "deadbeef")

    def test_user_read_serializes_none_avatar_hash(self) -> None:
        with self._session() as session:
            user = User(discord_id=4, display_name="serialized_none")
            session.add(user)
            session.commit()
            session.refresh(user)
            read = UserRead.model_validate(user, from_attributes=True)
            self.assertIsNone(read.discord_avatar_hash)


def _load_migration_0011():
    path = (
        Path(__file__).resolve().parent.parent
        / "alembic"
        / "versions"
        / "0011_user_discord_avatar_hash.py"
    )
    spec = importlib.util.spec_from_file_location("migration_0011", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Migration0011AvatarHashTest(unittest.TestCase):
    """Task 1 — migration 0011 is a plain nullable ADD COLUMN, reversible.

    Mirrors test_migration_0008_misc.py: capture the migration's ``op`` calls in
    ``as_sql=True`` mode against the PostgreSQL dialect, asserting the upgrade adds
    a nullable ``discord_avatar_hash`` column to ``users`` (and NOT an enum / CREATE
    TYPE), and the downgrade drops it. Fully offline, no DB.
    """

    def _capture(self, fn_name: str):
        import sqlalchemy as sa
        from alembic.operations import Operations
        from alembic.runtime.migration import MigrationContext

        module = _load_migration_0011()

        add_columns: list[tuple[str, sa.Column]] = []
        drop_columns: list[tuple[str, str]] = []
        executes: list[str] = []

        real_add_column = Operations.add_column
        real_drop_column = Operations.drop_column
        real_execute = Operations.execute

        def capture_add_column(self, table_name, column, *a, **kw):  # noqa: ANN001
            add_columns.append((table_name, column))
            return None

        def capture_drop_column(self, table_name, column_name, *a, **kw):  # noqa: ANN001
            drop_columns.append((table_name, column_name))
            return None

        def capture_execute(self, sqltext, *a, **kw):  # noqa: ANN001
            executes.append(str(sqltext))
            return None

        ctx = MigrationContext.configure(dialect_name="postgresql", opts={"as_sql": True})
        Operations.add_column = capture_add_column  # type: ignore[assignment]
        Operations.drop_column = capture_drop_column  # type: ignore[assignment]
        Operations.execute = capture_execute  # type: ignore[assignment]
        try:
            with Operations.context(Operations(ctx)):
                getattr(module, fn_name)()
        finally:
            Operations.add_column = real_add_column  # type: ignore[assignment]
            Operations.drop_column = real_drop_column  # type: ignore[assignment]
            Operations.execute = real_execute  # type: ignore[assignment]

        return add_columns, drop_columns, executes

    def test_upgrade_adds_nullable_avatar_hash_column(self) -> None:
        add_columns, _, executes = self._capture("upgrade")
        matched = [
            (table, col)
            for table, col in add_columns
            if table == "users" and col.name == "discord_avatar_hash"
        ]
        self.assertEqual(
            len(matched),
            1,
            "0011 must add exactly one discord_avatar_hash column to users.",
        )
        _, col = matched[0]
        self.assertTrue(
            col.nullable,
            "users.discord_avatar_hash must be nullable.",
        )
        # No enum involved — must NOT emit a CREATE TYPE.
        self.assertNotIn(
            "CREATE TYPE",
            " ".join(executes).upper(),
            "0011 is a plain nullable ADD COLUMN — no enum / CREATE TYPE.",
        )

    def test_downgrade_drops_the_column(self) -> None:
        _, drop_columns, _ = self._capture("downgrade")
        self.assertIn(
            ("users", "discord_avatar_hash"),
            drop_columns,
            "0011 downgrade must drop users.discord_avatar_hash.",
        )

    def test_revision_chain(self) -> None:
        module = _load_migration_0011()
        self.assertEqual(module.revision, "0011")
        self.assertEqual(module.down_revision, "0010")


class ProvisionAndUpsertAvatarTests(unittest.TestCase):
    """Task 2 — capture at register + service upsert/clear/no-op."""

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

    def _get_user(self, session: Session, discord_id: int) -> User:
        user = session.exec(select(User).where(User.discord_id == discord_id)).one_or_none()
        assert user is not None
        return user

    def test_provision_user_persists_avatar_hash(self) -> None:
        with self._session() as session:
            user_id, _name, _plain = provision_user(
                session, 100, "AvatarUser", avatar_hash="cafef00d"
            )
            user = session.get(User, user_id)
            assert user is not None
            self.assertEqual(user.discord_avatar_hash, "cafef00d")

    def test_provision_user_without_avatar_hash_leaves_null(self) -> None:
        with self._session() as session:
            user_id, _name, _plain = provision_user(session, 101, "NoAvatarUser")
            user = session.get(User, user_id)
            assert user is not None
            self.assertIsNone(user.discord_avatar_hash)

    def test_upsert_updates_existing_row(self) -> None:
        with self._session() as session:
            provision_user(session, 102, "ToUpdate")
            changed = upsert_avatar_hash_by_discord_id(session, 102, "newhash123")
            self.assertTrue(changed)
            self.assertEqual(self._get_user(session, 102).discord_avatar_hash, "newhash123")

    def test_upsert_clears_to_none(self) -> None:
        with self._session() as session:
            provision_user(session, 103, "ToClear", avatar_hash="willbecleared")
            changed = upsert_avatar_hash_by_discord_id(session, 103, None)
            self.assertTrue(changed)
            self.assertIsNone(self._get_user(session, 103).discord_avatar_hash)

    def test_upsert_unknown_discord_id_is_noop_returns_false(self) -> None:
        with self._session() as session:
            changed = upsert_avatar_hash_by_discord_id(session, 999999, "orphan")
            self.assertFalse(changed)


class DiscordIdSnowflakePrecisionTests(unittest.TestCase):
    """Snowflake ids serialize to an EXACT string via the JSON path (260629-kor).

    Discord snowflakes (e.g. 302924379799683073) exceed JS Number.MAX_SAFE_INTEGER
    (2**53). If emitted as a JSON number, the browser's JSON.parse rounds them to
    the nearest double — ...683073 -> ...683100 — corrupting the avatar CDN URL.
    Every schema that exposes discord_id MUST emit it as a string. These pin that
    via model_dump(mode="json") (the path FastAPI uses) for a value LARGER than
    2**53, asserting the exact string and that no precision is lost.
    """

    # A real Discord snowflake, comfortably above 2**53 (9007199254740992).
    SNOWFLAKE = 302924379799683073
    SNOWFLAKE_STR = "302924379799683073"

    def test_snowflake_exceeds_js_safe_integer(self) -> None:
        # Guard the premise: the fixture id is genuinely unsafe as a JS number.
        self.assertGreater(self.SNOWFLAKE, 2**53)

    def test_user_read_serializes_snowflake_to_exact_string(self) -> None:
        from datetime import datetime, timezone

        read = UserRead(
            id=1,
            discord_id=self.SNOWFLAKE,
            discord_avatar_hash="deadbeef",
            display_name="snowflake_user",
            is_admin=False,
            is_active=True,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        dumped = read.model_dump(mode="json")
        self.assertEqual(dumped["discord_id"], self.SNOWFLAKE_STR)
        self.assertIsInstance(dumped["discord_id"], str)

    def test_admin_user_read_serializes_snowflake_to_exact_string(self) -> None:
        from datetime import datetime, timezone

        read = AdminUserRead(
            id=1,
            display_name="snowflake_admin",
            discord_id=self.SNOWFLAKE,
            discord_avatar_hash="deadbeef",
            is_admin=True,
            is_active=True,
            is_protected=False,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            pick_count=0,
        )
        dumped = read.model_dump(mode="json")
        self.assertEqual(dumped["discord_id"], self.SNOWFLAKE_STR)
        self.assertIsInstance(dumped["discord_id"], str)

    def test_user_week_result_serializes_snowflake_to_exact_string(self) -> None:
        result = UserWeekResult(
            display_name="snowflake_user",
            weekly_score=0,
            picks=[],
            discord_id=self.SNOWFLAKE,
            discord_avatar_hash="deadbeef",
        )
        dumped = result.model_dump(mode="json")
        self.assertEqual(dumped["discord_id"], self.SNOWFLAKE_STR)
        self.assertIsInstance(dumped["discord_id"], str)

    def test_season_standing_row_serializes_snowflake_to_exact_string(self) -> None:
        row = SeasonStandingRow(
            display_name="snowflake_user",
            season_total=0,
            weekly_scores={},
            discord_id=self.SNOWFLAKE,
            discord_avatar_hash="deadbeef",
        )
        dumped = row.model_dump(mode="json")
        self.assertEqual(dumped["discord_id"], self.SNOWFLAKE_STR)
        self.assertIsInstance(dumped["discord_id"], str)

    def test_none_discord_id_serializes_to_null(self) -> None:
        # The web-bootstrap admin / web-origin accounts carry no snowflake.
        from datetime import datetime, timezone

        read = UserRead(
            id=1,
            discord_id=None,
            discord_avatar_hash=None,
            display_name="web_admin",
            is_admin=True,
            is_active=True,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        self.assertIsNone(read.model_dump(mode="json")["discord_id"])


if __name__ == "__main__":
    unittest.main()
