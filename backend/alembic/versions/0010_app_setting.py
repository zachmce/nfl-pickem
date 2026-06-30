"""app_setting — a keyed app-wide key/value settings table

Revision ID: 0010
Revises: 0009
Create Date: 2026-06-28 00:00:00.000000

Hand-written (mirrors 0005_demo_state's style) so the tiny ``app_setting`` table
is created/dropped deterministically. The table is the generic keyed settings
store whose first consumer is the admin-selectable bot personality (stored under
key ``bot_personality``). No enum type is involved — the personality id is a
free string validated against the registry in the service — so there is no
PG-enum-reuse concern. Kept tiny and fully reversible.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel  # noqa: F401  (kept for parity with the surrounding migrations)

# revision identifiers, used by Alembic.
revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "app_setting",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("setting_key", sa.String(length=100), nullable=False),
        sa.Column("setting_value", sa.String(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("setting_key", name="uq_app_setting_setting_key"),
    )
    op.create_index(
        op.f("ix_app_setting_setting_key"),
        "app_setting",
        ["setting_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_app_setting_setting_key"), table_name="app_setting")
    op.drop_table("app_setting")
