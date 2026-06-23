"""demo_state — the single persisted demo anchor instant

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-23 00:00:00.000000

Hand-written (mirrors 0004's style) so the tiny single-row ``demo_state`` table
is created/dropped deterministically. The table persists the ONE absolute
real-clock instant captured at demo seed time that both the seed process and the
worker/beat process read to rebuild the SAME ``Demo2025Source(offset)``. Kept
tiny and fully reversible so the demo footprint is cleanly purgeable before
go-live (PROD-LEAK-GUARD).
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel  # noqa: F401  (kept for parity with 0004)

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "demo_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("demo_started_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("demo_state")
