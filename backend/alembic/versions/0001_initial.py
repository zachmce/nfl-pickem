"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-22 00:00:00.000000

"""

from collections.abc import Sequence


# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # No-op base revision. This migration originally created the `task_run`
    # scaffold table (an early wiring-proof artifact); that table and its model
    # were removed, so there is nothing to create here. The revision is kept as
    # the base of the chain — later migrations depend on its id.
    pass


def downgrade() -> None:
    pass
