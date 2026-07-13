"""historical_game — final historical NFL games + consensus lines

Revision ID: 0017
Revises: 0016
Create Date: 2026-07-13 00:00:00.000000

Hand-written (mirrors 0010_app_setting's style) so the ``historical_game`` table
is created/dropped deterministically. This is a SEPARATE table from ``game`` — it
holds a static, append-only corpus of completed games (1999 -> last completed
season) for the bot-picks prior and long-run color facts, and carries NONE of
``game``'s poller/odds-freeze/notify state. Keeping it separate preserves the
active-season ``max(game.season)`` invariant (historical rows can never enter it).

NO enum type is involved (``game_type`` is a plain ``VARCHAR`` — REG/WC/DIV/CON/SB),
so the PG enum-reuse caveat does not apply. The table starts EMPTY and is filled
by the idempotent startup upsert (``app.seeds.historical_games``), so no
``server_default`` backfill is needed here. Fully reversible.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel  # noqa: F401  (kept for parity with the surrounding migrations)

# revision identifiers, used by Alembic.
revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "historical_game",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("nflverse_game_id", sa.String(length=50), nullable=False),
        sa.Column("season", sa.Integer(), nullable=False),
        sa.Column("week", sa.Integer(), nullable=False),
        sa.Column("game_type", sa.String(length=10), nullable=False),
        sa.Column("gameday", sa.Date(), nullable=False),
        sa.Column("home_team_id", sa.Integer(), nullable=False),
        sa.Column("away_team_id", sa.Integer(), nullable=False),
        sa.Column("home_score", sa.Integer(), nullable=False),
        sa.Column("away_score", sa.Integer(), nullable=False),
        sa.Column("result", sa.Integer(), nullable=False),
        sa.Column("spread_line", sa.Numeric(4, 1), nullable=False),
        sa.Column("total_line", sa.Numeric(5, 1), nullable=True),
        sa.ForeignKeyConstraint(["home_team_id"], ["team.id"]),
        sa.ForeignKeyConstraint(["away_team_id"], ["team.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("nflverse_game_id", name="uq_historical_game_nflverse_game_id"),
    )


def downgrade() -> None:
    op.drop_table("historical_game")
