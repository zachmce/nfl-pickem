"""users.discord_avatar_hash — persist each Discord user's avatar hash

Revision ID: 0011
Revises: 0010
Create Date: 2026-06-29 00:00:00.000000

Hand-written (mirrors 0009's style). Adds a single nullable
``discord_avatar_hash`` column to ``users`` so the bot can capture and refresh
each member's current Discord avatar hash (keyed by discord_id), and so every
personalization surface can read it. The site builds the CDN URL from the hash
downstream — no URL is stored here.

This is a PLAIN nullable ADD COLUMN: there is NO enum involved, so the Postgres
enum-reuse caveat (``postgresql.ENUM(create_type=False)``, see 0007/0008) does
NOT apply here. ``upgrade()`` adds the column NULL for every existing row;
``downgrade()`` drops it.

Note: the SQLite-only OFFLINE tests do not run migrations — they build the schema
from the SQLModel metadata (``SQLModel.metadata.create_all``), so this column is
exercised in tests via the new ``User.discord_avatar_hash`` model field, and by
this migration only on the live Postgres path.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel  # noqa: F401  (kept for parity with 0004..0010)

# revision identifiers, used by Alembic.
revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("discord_avatar_hash", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "discord_avatar_hash")
