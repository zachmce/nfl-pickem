from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # Postgres
    postgres_user: str = "pickem"
    postgres_password: str = "pickem"
    postgres_db: str = "pickem"
    postgres_host: str = "db"
    postgres_port: int = 5432

    # Redis / Celery
    redis_url: str = "redis://redis:6379/0"

    # Auth: session-cookie signing (itsdangerous). The default is a DEV-ONLY
    # placeholder so the local stack runs out of the box — ALWAYS override
    # SECRET_KEY in .env for any non-local deployment.
    secret_key: str = "dev-only-insecure-secret-key-change-me-in-production"
    session_cookie_name: str = "session"
    session_max_age_days: int = 30
    # Set True in prod (HTTPS) so the session cookie is only sent over TLS.
    # Must be False for local http://localhost or the browser drops the cookie.
    session_cookie_secure: bool = False

    # CORS: explicit origins are REQUIRED for credentialed (cookie) requests —
    # a wildcard "*" is rejected by browsers when credentials are sent. The Vite
    # dev proxy makes /api same-origin so this only matters for direct calls.
    cors_allowed_origins: list[str] = ["http://localhost:5173"]

    # Discord bot. Non-bot containers (api, worker, migrate) leave the token and
    # guild id unset; the bot container asserts they are present at startup.
    discord_bot_token: str | None = None
    discord_guild_id: int | None = None
    # REG-07 DM password-change pointer — where the bot tells users to log in.
    app_base_url: str = "http://localhost:5173"

    log_level: str = "INFO"

    @field_validator("discord_bot_token", "discord_guild_id", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: object) -> object:
        """Treat an empty/whitespace env value (e.g. ``DISCORD_GUILD_ID=``) as unset.

        Without this, a placeholder line in .env crashes every container that
        loads Settings — not just the bot.
        """
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @property
    def database_url(self) -> str:
        """SQLAlchemy/psycopg3 connection string."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
