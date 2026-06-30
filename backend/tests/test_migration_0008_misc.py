"""Regression guard: migration 0008 must EXTEND the existing ``picktype`` enum
(``ALTER TYPE ... ADD VALUE``) and add a nullable ``misc_text`` column — offline,
no DB.

Why this offline test exists (same rationale as test_migration_0007_enum):
the SQLite-only unittest suite builds its schema from
``SQLModel.metadata.create_all`` (which renders ``picktype`` as a plain VARCHAR)
and runs NO migration, so it can never catch a Postgres enum-reuse regression. If
0008 ever switched to a generic ``sa.Enum`` (which would re-emit
``CREATE TYPE picktype`` and fail with DuplicateObject on a live Postgres
upgrade), the SQLite suite would stay green. This test pins the fix by inspecting
the DDL the migration emits, with no database and no network.

It captures the migration's ``op`` calls in ``as_sql=True`` mode against the
PostgreSQL dialect and asserts:

* the upgrade emits an ``ALTER TYPE picktype ADD VALUE`` statement carrying the
  new ``'MISC'`` member (via ``op.execute``);
* the upgrade adds a ``misc_text`` column to ``pick`` that is nullable.

Fully offline: imports the migration module and inspects the captured operations.
Run from backend/ with ``.venv/bin/python -m unittest``.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_migration_0008():
    path = (
        Path(__file__).resolve().parent.parent / "alembic" / "versions" / "0008_misc_pick_type.py"
    )
    spec = importlib.util.spec_from_file_location("migration_0008", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Migration0008MiscTest(unittest.TestCase):
    def _captured_upgrade(self):
        """Run the migration's ``upgrade()`` against a captured Operations proxy
        (PostgreSQL dialect, ``as_sql=True``) and return the captured calls."""
        import sqlalchemy as sa
        from alembic.operations import Operations
        from alembic.runtime.migration import MigrationContext

        module = _load_migration_0008()

        executes: list[str] = []
        add_columns: list[tuple[str, sa.Column]] = []

        real_execute = Operations.execute
        real_add_column = Operations.add_column

        def capture_execute(self, sqltext, *a, **kw):  # noqa: ANN001
            executes.append(str(sqltext))
            return None

        def capture_add_column(self, table_name, column, *a, **kw):  # noqa: ANN001
            add_columns.append((table_name, column))
            return None

        ctx = MigrationContext.configure(dialect_name="postgresql", opts={"as_sql": True})
        Operations.execute = capture_execute  # type: ignore[assignment]
        Operations.add_column = capture_add_column  # type: ignore[assignment]
        try:
            with Operations.context(Operations(ctx)):
                module.upgrade()
        finally:
            Operations.execute = real_execute  # type: ignore[assignment]
            Operations.add_column = real_add_column  # type: ignore[assignment]

        return executes, add_columns

    def test_upgrade_adds_misc_enum_value_not_recreate(self) -> None:
        executes, _ = self._captured_upgrade()
        joined = " ".join(executes).upper()
        self.assertIn(
            "ALTER TYPE PICKTYPE ADD VALUE",
            joined,
            "0008 must EXTEND the existing picktype enum via ALTER TYPE ADD VALUE "
            "(not recreate it with a generic sa.Enum, which would re-emit CREATE "
            "TYPE and fail with DuplicateObject on Postgres).",
        )
        self.assertIn(
            "MISC",
            joined,
            "the ALTER TYPE statement must add the 'MISC' enum value.",
        )
        # Belt-and-suspenders: the migration must NOT emit a CREATE TYPE for
        # picktype (that would be the recreate regression).
        self.assertNotIn(
            "CREATE TYPE PICKTYPE",
            joined,
            "0008 must NOT recreate the picktype enum.",
        )

    def test_upgrade_adds_nullable_misc_text_column(self) -> None:
        _, add_columns = self._captured_upgrade()
        matched = [
            (table, col)
            for table, col in add_columns
            if table == "pick" and col.name == "misc_text"
        ]
        self.assertEqual(
            len(matched),
            1,
            "0008 must add exactly one misc_text column to the pick table.",
        )
        _, col = matched[0]
        self.assertTrue(
            col.nullable,
            "pick.misc_text must be nullable (NULL for every non-MISC pick).",
        )


if __name__ == "__main__":
    unittest.main()
