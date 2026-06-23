"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-22 00:00:00.000000

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "task_run",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("message", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_task_run_message"), "task_run", ["message"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_task_run_message"), table_name="task_run")
    op.drop_table("task_run")
