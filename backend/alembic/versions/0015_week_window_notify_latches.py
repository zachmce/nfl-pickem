"""week.window_open_notified / window_close_notified — fire-once notify latches

Revision ID: 0015
Revises: 0014
Create Date: 2026-07-06 00:00:00.000000

Hand-written (mirrors 0014's style). Adds two boolean fire-once notify latches to
the ``week`` table so the ``window.opened`` / ``window.closed`` Discord
notification fires exactly once per week per seed generation:

* ``window_open_notified`` — set True the first poll cycle a week's pick window is
  observed open.
* ``window_close_notified`` — set True the first poll cycle a week's window is
  observed closed (after it was announced open).

These are PLAIN additive booleans (NOT NULL, ``server_default=sa.false()``, so
existing rows backfill to False): there is NO enum involved, so the Postgres
enum-reuse caveat (``postgresql.ENUM(create_type=False)``, see 0007/0008) does NOT
apply here. The ``false`` server_default keeps every EXISTING ``week`` row
un-notified after the migration, so upgrading a live DB emits no spurious
"Picks Opened" / window-closed backfill notification. ``upgrade()`` adds both
columns; ``downgrade()`` drops them.

Note: the SQLite-only OFFLINE tests do not run migrations — they build the schema
from the SQLModel metadata (``SQLModel.metadata.create_all``), so these columns are
exercised in tests via the new ``Week.window_open_notified`` /
``Week.window_close_notified`` model fields, and by this migration only on the live
Postgres path.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel  # noqa: F401  (kept for parity with 0004..0014)

# revision identifiers, used by Alembic.
revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "week",
        sa.Column(
            "window_open_notified", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
    )
    op.add_column(
        "week",
        sa.Column(
            "window_close_notified", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
    )


def downgrade() -> None:
    op.drop_column("week", "window_close_notified")
    op.drop_column("week", "window_open_notified")
