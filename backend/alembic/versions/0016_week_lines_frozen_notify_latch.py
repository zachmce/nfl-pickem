"""week.lines_frozen_notified — fire-once freeze notify latch

Revision ID: 0016
Revises: 0015
Create Date: 2026-07-06 00:00:00.000000

Hand-written (mirrors 0015's style). Adds one boolean fire-once notify latch to
the ``week`` table so the ``freeze.week`` "Lines Locked" Discord notification
fires exactly once per week per seed generation:

* ``lines_frozen_notified`` — set True the first poll cycle a week's odds are
  observed frozen (the COMPUTED noon-ET-Wednesday clock OR the manual admin
  override), so the "Lines Locked" card fires exactly once per week.

This is a PLAIN additive boolean (NOT NULL, ``server_default=sa.false()``, so
existing rows backfill to False): there is NO enum involved, so the Postgres
enum-reuse caveat (``postgresql.ENUM(create_type=False)``, see 0007/0008) does NOT
apply here. ``upgrade()`` adds the column; ``downgrade()`` drops it.

Backfill note: ``refresh_games`` scans EVERY week, so an already-frozen week with
``lines_frozen_notified=False`` WILL emit ``freeze.week`` on the next poll. This is
harmless on this project's deploy paths because seeding re-arms correctly: the
demo/go-live seed calls ``refresh_games(now=...)`` which silently re-latches every
week already frozen at seed-now (accumulating edges without publishing), so those
do NOT re-fire — only freezes the live poller crosses AFTER the seed fire once. The
one path that WOULD emit a one-time catch-up per already-frozen week is applying
this migration to a live, mid-season DB that is never re-seeded — not a path this
project uses (deploys seed).

Note: the SQLite-only OFFLINE tests do not run migrations — they build the schema
from the SQLModel metadata (``SQLModel.metadata.create_all``), so this column is
exercised in tests via the new ``Week.lines_frozen_notified`` model field, and by
this migration only on the live Postgres path.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel  # noqa: F401  (kept for parity with 0004..0015)

# revision identifiers, used by Alembic.
revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "week",
        sa.Column("lines_frozen_notified", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("week", "lines_frozen_notified")
