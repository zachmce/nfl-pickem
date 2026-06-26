"""misc pick type — extend the picktype enum + add pick.misc_text

Revision ID: 0008
Revises: 0007
Create Date: 2026-06-26 00:00:00.000000

Hand-written (mirrors 0006/0007's style) so the MISC pick type is added
deterministically and — critically — REUSES the EXISTING ``picktype`` native
enum (created by 0004) instead of trying to recreate it.

Two upgrade steps:

(1) Extend the EXISTING ``picktype`` enum with the new ``MISC`` value via
    ``ALTER TYPE picktype ADD VALUE IF NOT EXISTS 'MISC'``. This is the Postgres
    enum-reuse rule: a generic ``sa.Enum`` would try to recreate the type
    (DuplicateObject against 0004's type). Target is Postgres 17; PG 12+ permits
    ``ADD VALUE`` inside a transaction as long as the new value is not USED in the
    same transaction — which it is not here — so no autocommit dance is required.
    ``IF NOT EXISTS`` makes a re-run idempotent.

(2) Add the nullable ``misc_text`` column on ``pick`` (NULL for every existing
    pick of any other type).

``downgrade()`` drops ONLY the column. Postgres cannot drop an enum value, and
leaving the extra value in place is harmless — this mirrors 0007's "downgrade
does not drop the shared enum" comment.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel  # noqa: F401  (kept for parity with 0004/0005/0006/0007)

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # (1) Extend the EXISTING picktype enum — do NOT recreate it (0004 owns it).
    # ``IF NOT EXISTS`` keeps a re-run idempotent; PG 12+ allows ADD VALUE inside
    # the migration transaction since the value is not USED in the same txn here.
    op.execute("ALTER TYPE picktype ADD VALUE IF NOT EXISTS 'MISC'")

    # (2) Add the free-text column (NULL for every existing pick).
    op.add_column(
        "pick",
        sa.Column("misc_text", sa.String(length=280), nullable=True),
    )


def downgrade() -> None:
    # Drop only the column. The picktype enum value cannot be removed in Postgres
    # and leaving it is harmless (mirrors 0007 leaving the shared enum in place).
    op.drop_column("pick", "misc_text")
