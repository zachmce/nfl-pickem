"""game.odds_provider_id — persist the chosen provider id alongside its name

Revision ID: 0009
Revises: 0008
Create Date: 2026-06-26 00:00:00.000000

Hand-written (mirrors 0008's style). Adds a single nullable ``odds_provider_id``
column to ``game`` so the ingest/poll paths can persist the chosen betting
provider's ID alongside its NAME (``odds_provider``). Provider ids drift across
ESPN endpoints/time, so the id is captured from the SAME selected odds item the
name comes from — never hardcoded — and stored here for auditability.

This is a PLAIN nullable ADD COLUMN: there is NO enum involved, so the Postgres
enum-reuse caveat (``postgresql.ENUM(create_type=False)``, see 0007/0008) does
NOT apply here. ``upgrade()`` adds the column NULL for every existing row;
``downgrade()`` drops it.

Note: the SQLite-only OFFLINE tests do not run migrations — they build the schema
from the SQLModel metadata (``SQLModel.metadata.create_all``), so this column is
exercised in tests via the new ``Game.odds_provider_id`` model field, and by this
migration only on the live Postgres path.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel  # noqa: F401  (kept for parity with 0004..0008)

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "game",
        sa.Column("odds_provider_id", sa.String(length=50), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("game", "odds_provider_id")
