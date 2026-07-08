from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Single source of truth for the DEV-ONLY session-signing key. Referenced both as
# the secret_key field default AND by the production fail-closed validator below,
# so the validator's equality check can never drift from the actual default.
_DEV_SECRET_KEY = "dev-only-insecure-secret-key-change-me-in-production"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # Deploy environment indicator (env APP_ENV) and the fail-closed trigger.
    # The default MUST stay non-production so dev/demo/test boot byte-for-byte
    # unchanged: the _prod_fail_closed validator below is a strict no-op unless
    # this equals "production". On the real server set APP_ENV=production to
    # activate the guard that refuses to boot on insecure dev defaults.
    app_env: Literal["development", "test", "staging", "production"] = "development"

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
    secret_key: str = _DEV_SECRET_KEY
    session_cookie_name: str = "session"
    session_max_age_days: int = 30
    # Set True in prod (HTTPS) so the session cookie is only sent over TLS.
    # Must be False for local http://localhost or the browser drops the cookie.
    session_cookie_secure: bool = False

    # CORS: explicit origins are REQUIRED for credentialed (cookie) requests —
    # a wildcard "*" is rejected by browsers when credentials are sent. The Vite
    # dev proxy makes /api same-origin so this only matters for direct calls.
    #
    # IMPORTANT — cookie auth is same-origin only. The session cookie is
    # SameSite=Lax (see auth._set_session_cookie), so a browser will NOT attach it
    # on cross-site XHR/fetch. Configuring CORS origins does not change that:
    # a separately-hosted SPA calling this API cross-origin would silently 401 even
    # with the origin allow-listed here. It works today because both dev (Vite
    # proxy) and prod (nginx proxy) serve the SPA and API on the SAME origin.
    # Genuine cross-origin cookie auth would require SameSite=None + Secure and is
    # intentionally unsupported — deploy behind the proxy instead.
    cors_allowed_origins: list[str] = ["http://localhost:5173"]

    # Discord bot. Non-bot containers (api, worker, migrate) leave the token and
    # guild id unset; the bot container asserts they are present at startup.
    discord_bot_token: str | None = None
    discord_guild_id: int | None = None
    # Discord notification target channels (QT-1). Resolved by the bot subscriber
    # within DISCORD_GUILD_ID's channels (by numeric id OR by name). Default None so
    # prod is unaffected when unset; a blank env value is coerced to None below.
    discord_chat_log_channel: str | None = None
    discord_chat_channel: str | None = None
    # REG-07 DM password-change pointer — where the bot tells users to log in.
    app_base_url: str = "http://localhost:5173"
    # Cadence (minutes) of the bot's guild avatar sweep, which upserts every
    # member's current Discord avatar hash keyed by discord_id (default hourly).
    # The sweep ALSO runs once at startup (the loop's first tick, after ready), so
    # this only controls the steady-state refresh interval. Plain int with a
    # default — NOT in the _empty_str_to_none validator (str|None fields only).
    discord_avatar_sweep_minutes: int = 60

    # Local OpenAI-compatible LLM for the pickem-chat personality layer (260627-nef).
    # All three must be set for the feature to engage; ANY left None disables it (the
    # bot falls back to the deterministic line). Only the bot container sets these —
    # non-bot containers leave them unset, and a blank env value is coerced to None
    # below so a placeholder line in .env never crashes a container.
    llm_api_server: str | None = None
    llm_api_model: str | None = None
    llm_api_key: str | None = None

    log_level: str = "INFO"

    # Demo-mode gate (DEMO-GATE / PROD-LEAK-GUARD). Read from env IS_DEMO_DATA.
    #   OFF (default) -> the PRODUCTION path: default_scoreboard_source() returns
    #     the real ESPN adapter and NO demo code (seed, anchor, Demo2025Source) is
    #     ever imported or constructed. The prod path is byte-for-byte unchanged.
    #   ON -> the DEMO path: the app logs a loud startup banner (so it can never be
    #     on silently), the env-gated demo seed runs, and default_scoreboard_source()
    #     returns the time-shifted Demo2025Source built from the shared persisted
    #     anchor. This is the ONLY source flag — do not add a second one anywhere.
    is_demo_data: bool = False

    # QT-B admin-bootstrap credentials (consumed by app.seeds.admins). Both must
    # be set for the seed to mint the deterministic admin; either blank => no-op.
    default_admin_username: str | None = None
    default_admin_password: str | None = None

    @field_validator(
        "discord_bot_token",
        "discord_guild_id",
        "discord_chat_log_channel",
        "discord_chat_channel",
        "default_admin_username",
        "default_admin_password",
        "llm_api_server",
        "llm_api_model",
        "llm_api_key",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, v: object) -> object:
        """Treat an empty/whitespace env value (e.g. ``DISCORD_GUILD_ID=``) as unset.

        Without this, a placeholder line in .env crashes every container that
        loads Settings — not just the bot.
        """
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @model_validator(mode="after")
    def _prod_fail_closed(self) -> "Settings":
        """Refuse to boot a production deploy that still carries insecure dev defaults.

        This is the fail-closed mechanism for the go-live cutover: ``config.py``
        ships DEV DEFAULTS (a placeholder ``SECRET_KEY``, plaintext-OK cookies, a
        localhost CORS origin) so the local stack runs out of the box — but nothing
        otherwise stops those defaults reaching production. Because
        ``settings = get_settings()`` runs at import time, a ``ValueError`` raised
        here aborts the import and the process refuses to start.

        STRICT NO-OP outside production: when ``app_env != "production"`` (the
        default ``development``, plus ``test``/``staging``) this returns ``self``
        immediately, so EVERY existing dev/demo/test code path is byte-for-byte
        unchanged. Only ``app_env == "production"`` activates the checks below.

        ALL failing checks are collected into one actionable message (we do not
        raise on the first) so an operator fixing config sees every problem and the
        exact env var to set for each, in a single boot attempt.
        """
        if self.app_env != "production":
            return self

        failures: list[str] = []

        # Session-signing key: reject the shared dev default (T-lkf-01) and any key
        # too short to be a real random secret. Compared against _DEV_SECRET_KEY so
        # the check can never drift from the actual field default.
        if self.secret_key == _DEV_SECRET_KEY or len(self.secret_key) < 32:
            failures.append(
                "SECRET_KEY is the insecure dev default or too short (<32 chars); "
                "set a real random SECRET_KEY"
            )

        # Cookies must be HTTPS-only in production (T-lkf-02).
        if not self.session_cookie_secure:
            failures.append(
                "SESSION_COOKIE_SECURE must be true in production (HTTPS-only "
                "cookies); set SESSION_COOKIE_SECURE=true"
            )

        # Never serve demo data in production (T-lkf-03; belt-and-suspenders with
        # the existing PROD-LEAK-GUARD source seam).
        if self.is_demo_data:
            failures.append(
                "IS_DEMO_DATA must be false in production (never serve demo data); "
                "set IS_DEMO_DATA=false"
            )

        # CORS origins must be real and non-localhost (T-lkf-04). Empty list or any
        # localhost/127.0.0.1 entry is rejected.
        if not self.cors_allowed_origins or any(
            "localhost" in origin or "127.0.0.1" in origin for origin in self.cors_allowed_origins
        ):
            failures.append(
                "CORS_ALLOWED_ORIGINS must list real non-localhost origins; "
                "set CORS_ALLOWED_ORIGINS"
            )

        if failures:
            bullets = "\n".join(f"  - {f}" for f in failures)
            raise ValueError(
                "Refusing to start: APP_ENV=production but insecure configuration "
                f"detected:\n{bullets}"
            )

        return self

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


def default_scoreboard_source(session=None):
    """Resolve the default scoreboard source — the single source seam (D-03).

    This is the ONE place the scoreboard source is constructed. It is a plain
    factory (not a Settings field) so the network-touching ESPN adapter is only
    imported/instantiated by callers that actually need it (the Celery task
    wrapper) — keeping the reconciliation service and the offline test suite free
    of any ESPN/network dependency.

    Gating (PROD-LEAK-GUARD, belt-and-suspenders):

    * ``settings.is_demo_data`` FALSE (default, the prod path): behaves EXACTLY as
      before — imports and returns the real :class:`EspnScoreboardSource`. NO demo
      module is imported, no DemoState row is read. Byte-for-byte the prod path.
    * ``settings.is_demo_data`` TRUE (the demo path): lazily imports the demo
      machinery (so it is never importable/constructed when the flag is off),
      reads the SINGLE shared persisted anchor, rebuilds the SAME
      ``Demo2025Source(offset_from_anchor(anchor))`` the seed used, and returns it.
      A missing anchor raises a clear ``RuntimeError`` (run the demo seed first).

    ``session`` is honored only on the demo branch (the Celery task passes its own
    open session so the demo branch reuses it instead of opening a second one). The
    ESPN branch ignores it entirely.
    """
    if settings.is_demo_data:
        # Lazy imports kept INSIDE the True branch so the demo source is never
        # imported/constructed in the prod path (PROD-LEAK-GUARD T-sf0-01).
        from app.demo.anchor import load_demo_anchor, offset_from_anchor
        from app.scoreboard.demo import Demo2025Source

        if session is not None:
            anchor = load_demo_anchor(session)
        else:
            from app.db import task_session

            with task_session() as own_session:
                anchor = load_demo_anchor(own_session)

        if anchor is None:
            raise RuntimeError(
                "IS_DEMO_DATA is on but no demo anchor is persisted. Run the demo "
                "seed first: `python -m app.seeds.demo`."
            )
        return Demo2025Source(offset_from_anchor(anchor))

    from app.scoreboard.espn import EspnScoreboardSource

    return EspnScoreboardSource()
