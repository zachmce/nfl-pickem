"""users.session_version — invalidate sessions on password change

Revision ID: 0014
Revises: 0013
Create Date: 2026-06-30 00:00:00.000000

Hand-written (mirrors 0011's style). Adds a single ``session_version`` integer
column to ``users`` with a server default of 0. The signed session cookie carries
this value as ``sv``; every authenticated request compares the cookie's ``sv`` to
this column and rejects a stale cookie. Every password-write site bumps the
counter, so a password change logs out all previously-issued cookies.

This is a PLAIN additive ADD COLUMN (NOT NULL with a server_default 0, so existing
rows backfill to 0): there is NO enum involved, so the Postgres enum-reuse caveat
(``postgresql.ENUM(create_type=False)``, see 0007/0008) does NOT apply here. The
server_default 0 also makes the missing-``sv`` == 0 backward-compat rule exact —
a legacy cookie with no ``sv`` decodes to 0 and matches the backfilled column, so
the deploy does NOT force-log-out existing sessions. ``upgrade()`` adds the column;
``downgrade()`` drops it.

Note: the SQLite-only OFFLINE tests do not run migrations — they build the schema
from the SQLModel metadata (``SQLModel.metadata.create_all``), so this column is
exercised in tests via the new ``User.session_version`` model field, and by this
migration only on the live Postgres path.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel  # noqa: F401  (kept for parity with 0004..0013)

# revision identifiers, used by Alembic.
revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("session_version", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("users", "session_version")
