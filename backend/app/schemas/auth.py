from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.types import DiscordId


class UserRead(BaseModel):
    id: int
    discord_id: DiscordId  # NULL for the web-bootstrap admin; emitted as a string
    # The Discord avatar hash; None means no custom avatar — downstream falls back
    # to display_name initials (no URL is constructed here).
    discord_avatar_hash: str | None
    display_name: str
    is_admin: bool
    is_active: bool
    # Account join date — non-sensitive, always non-null on the User model. Lets
    # the Profile page show "Joined …". NEVER add password_hash or any secret here.
    created_at: datetime


class UserLoginRequest(BaseModel):
    display_name: str
    password: str


class TokenResponse(BaseModel):
    """OAuth2 password-flow token response (consumed by Swagger's Authorize)."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"


class LogoutResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: Literal["logged_out"]


class ChangePasswordRequest(BaseModel):
    """Request body for POST /api/auth/change-password.

    Length bounds live here so out-of-bounds new_password fails as a 422 before
    service logic runs. argon2id has no bcrypt-style 72-byte cap, so 128 is safe.
    """

    current_password: str = Field(max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class ChangePasswordResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: Literal["password_changed"]
