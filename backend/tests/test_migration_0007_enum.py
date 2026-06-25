"""Regression guard: migration 0007 must REUSE the existing ``picktype`` enum,
not recreate it (offline, no DB).

QT-1 (260625-m66) originally wrote the ``before_pick_type`` / ``after_pick_type``
columns as ``sa.Enum(PickType, name="picktype", create_type=False)``. But
``create_type`` is a *PostgreSQL-dialect* parameter — the GENERIC ``sa.Enum``
silently ignores it, so the PG-adapted type defaulted to ``create_type=True`` and
``alembic upgrade head`` emitted ``CREATE TYPE picktype ...`` again, failing with
``DuplicateObject`` against the type 0004 already created. The offline unittest
suite never caught this because it builds the schema from
``SQLModel.metadata.create_all`` on SQLite (which renders enums as VARCHAR and
runs no migration). The fix uses ``postgresql.ENUM(..., create_type=False)``,
which actually honors the flag.

This test pins the fix at the dialect level — it asserts the PG-adapted enum
columns carry ``create_type=False`` — so a regression to the generic ``sa.Enum``
fails here instead of only blowing up on a live Postgres upgrade.

Fully offline: imports the migration module and inspects column types; no DB,
no network. Run from backend/ with ``.venv/bin/python -m unittest``.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from sqlalchemy.dialects import postgresql


def _load_migration_0007():
    path = (
        Path(__file__).resolve().parent.parent
        / "alembic"
        / "versions"
        / "0007_pick_edit_audit.py"
    )
    spec = importlib.util.spec_from_file_location("migration_0007", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Migration0007EnumTest(unittest.TestCase):
    def _enum_columns(self):
        """Build the pick_edit_audit table the migration would create and return
        the two picktype enum columns, without touching a database."""
        import sqlalchemy as sa
        from alembic.operations import Operations
        from alembic.runtime.migration import MigrationContext

        module = _load_migration_0007()

        captured = {}
        real_create_table = Operations.create_table

        def capture_create_table(self, name, *columns, **kw):  # noqa: ANN001
            captured["table"] = sa.Table(name, sa.MetaData(), *columns)
            # Do not actually emit DDL — we only want the constructed columns.
            return captured["table"]

        ctx = MigrationContext.configure(
            dialect_name="postgresql", opts={"as_sql": True}
        )
        Operations.create_table = capture_create_table  # type: ignore[assignment]
        try:
            with Operations.context(Operations(ctx)):
                module.upgrade()
        finally:
            Operations.create_table = real_create_table  # type: ignore[assignment]

        table = captured["table"]
        return [
            table.c.before_pick_type.type,
            table.c.after_pick_type.type,
        ]

    def test_picktype_columns_do_not_recreate_the_enum(self):
        for col_type in self._enum_columns():
            pg_type = col_type.dialect_impl(postgresql.dialect())
            self.assertFalse(
                getattr(pg_type, "create_type", True),
                "0007 picktype enum columns must use postgresql.ENUM with "
                "create_type=False so they REUSE 0004's existing type instead "
                "of re-emitting CREATE TYPE (DuplicateObject on Postgres).",
            )


if __name__ == "__main__":
    unittest.main()
