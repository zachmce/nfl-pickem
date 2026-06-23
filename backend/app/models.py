from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import ClassVar

import sqlalchemy as sa
from sqlalchemy import BigInteger, Column
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enums
#
# No enums existed in the codebase before the pick'em domain. Convention
# established here: a Python ``str, Enum`` for type-safety in app code, mapped
# to a *native* Postgres enum type via ``sa.Enum(EnumClass, name=...)`` so the
# database enforces the allowed values too. Members are UPPER_SNAKE.
# ---------------------------------------------------------------------------
class GameStatus(str, Enum):
    SCHEDULED = "SCHEDULED"
    IN_PROGRESS = "IN_PROGRESS"
    FINAL = "FINAL"


class PickType(str, Enum):
    UNDERDOG_COVER = "UNDERDOG_COVER"
    FAVORITE_COVER = "FAVORITE_COVER"
    OVER = "OVER"
    UNDER = "UNDER"


class PickResult(str, Enum):
    PENDING = "PENDING"
    WIN = "WIN"
    LOSS = "LOSS"


class User(SQLModel, table=True):
    __tablename__: ClassVar[str] = "users"

    id: int | None = Field(default=None, primary_key=True)
    discord_id: int | None = Field(
        sa_column=Column(BigInteger, nullable=True, unique=True, index=True),
        ge=0,
        le=18446744073709551615,
        description="The unique Discord User Snowflake ID (NULL for web-bootstrap admin)",
    )
    password_hash: str | None = None
    display_name: str = Field(max_length=100, unique=True)
    is_admin: bool = Field(default=False)
    is_active: bool = Field(default=False)
    created_at: datetime = Field(sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False), default_factory=sa.func.now)


class TaskRun(SQLModel, table=True):
    """A trivial record written by the Celery worker to prove the DB wiring.

    Both FastAPI (to read/trigger) and the worker (to write) operate on this
    table through the shared engine in ``app.db``.
    """

    __tablename__ = "task_run"

    id: int | None = Field(default=None, primary_key=True)
    message: str = Field(index=True)
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)


# ---------------------------------------------------------------------------
# Pick'em domain
#
# Schema authority: .planning/notes/data-model.md. Surrogate int PKs; ESPN ids
# stored as unique columns (never PKs); odds embedded on Game (single frozen
# line, all nullable); Pick stores only (game, type) — the side is derived.
# ---------------------------------------------------------------------------
class Team(SQLModel, table=True):
    """NFL team reference data (~32 rows, seeded once)."""

    __tablename__ = "team"

    id: int | None = Field(default=None, primary_key=True)
    espn_team_id: int = Field(
        sa_column=sa.Column(sa.Integer, nullable=False, unique=True),
        description="ESPN team id — stored as a unique column, never the PK.",
    )
    abbreviation: str = Field(max_length=10, nullable=False)
    display_name: str = Field(max_length=100, nullable=False)


class Week(SQLModel, table=True):
    """A scoring week — owns the pick window and the line-freeze state."""

    __tablename__ = "week"
    __table_args__ = (sa.UniqueConstraint("season", "week", name="uq_week_season_week"),)

    id: int | None = Field(default=None, primary_key=True)
    season: int = Field(nullable=False)
    week: int = Field(nullable=False)
    # Window stored (not derived) so it is explicit: closes at the first
    # kickoff of the week, opens after the previous week's last game.
    window_opens_at: datetime | None = Field(
        default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True)
    )
    window_closes_at: datetime | None = Field(
        default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True)
    )
    lines_frozen: bool = Field(default=False, nullable=False)


class Game(SQLModel, table=True):
    """A single NFL game with its embedded (single, frozen) betting line."""

    __tablename__ = "game"

    id: int | None = Field(default=None, primary_key=True)
    espn_event_id: int = Field(
        sa_column=sa.Column(sa.Integer, nullable=False, unique=True),
        description="ESPN event id — unique column, never the PK.",
    )
    # Nullable to be safe for ingest ordering; the data-model lists it without a
    # uniqueness/null note.
    espn_competition_id: int | None = Field(default=None, nullable=True)
    week_id: int = Field(foreign_key="week.id", nullable=False)
    # Convenience duplicates of the owning week's (season, week) per data-model.
    season: int = Field(nullable=False)
    week: int = Field(nullable=False)
    home_team_id: int = Field(foreign_key="team.id", nullable=False)
    away_team_id: int = Field(foreign_key="team.id", nullable=False)
    kickoff_at: datetime | None = Field(
        default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True)
    )
    status: GameStatus = Field(
        default=GameStatus.SCHEDULED,
        sa_column=sa.Column(sa.Enum(GameStatus, name="gamestatus"), nullable=False),
    )
    home_score: int | None = Field(default=None, nullable=True)
    away_score: int | None = Field(default=None, nullable=True)

    # --- Embedded odds (all nullable: a game may exist before lines are posted) ---
    # ``spread`` is the positive half-point magnitude the favorite must cover.
    spread: Decimal | None = Field(
        default=None, sa_column=sa.Column(sa.Numeric(4, 1), nullable=True)
    )
    total: Decimal | None = Field(
        default=None, sa_column=sa.Column(sa.Numeric(5, 1), nullable=True)
    )
    favorite_team_id: int | None = Field(default=None, foreign_key="team.id")
    underdog_team_id: int | None = Field(default=None, foreign_key="team.id")
    odds_provider: str | None = Field(default=None, max_length=50, nullable=True)
    odds_frozen: bool = Field(default=False, nullable=False)
    odds_captured_at: datetime | None = Field(
        default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True)
    )


class Pick(SQLModel, table=True):
    """A user's pick on a game for a week. Side is derived from game + type.

    Two Postgres partial unique indexes enforce, at the DB level:
      * one of each base pick type per user/week (``is_mortal_lock = false``)
      * one mortal lock per user/week (``is_mortal_lock = true``)
    """

    __tablename__ = "pick"
    __table_args__ = (
        sa.Index(
            "uq_pick_user_week_type_base",
            "user_id",
            "week_id",
            "pick_type",
            unique=True,
            postgresql_where=sa.text("is_mortal_lock = false"),
            sqlite_where=sa.text("is_mortal_lock = false"),
        ),
        sa.Index(
            "uq_pick_user_week_mortal_lock",
            "user_id",
            "week_id",
            unique=True,
            postgresql_where=sa.text("is_mortal_lock = true"),
            sqlite_where=sa.text("is_mortal_lock = true"),
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", nullable=False)
    game_id: int = Field(foreign_key="game.id", nullable=False)
    week_id: int = Field(foreign_key="week.id", nullable=False)
    pick_type: PickType = Field(
        sa_column=sa.Column(sa.Enum(PickType, name="picktype"), nullable=False),
    )
    is_mortal_lock: bool = Field(default=False, nullable=False)
    result: PickResult = Field(
        default=PickResult.PENDING,
        sa_column=sa.Column(sa.Enum(PickResult, name="pickresult"), nullable=False),
    )
    points: int = Field(default=0, nullable=False)
    created_at: datetime = Field(
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False),
        default_factory=_utcnow,
    )
    updated_at: datetime = Field(
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False),
        default_factory=_utcnow,
    )
