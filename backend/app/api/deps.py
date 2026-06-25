"""Auth dependencies for cookie-based sessions.

The same signed session token can arrive two ways:
  - the SPA sends it as an HttpOnly cookie (set by POST /api/auth/login)
  - Swagger UI / API clients use the OAuth2 password flow's "Authorize" button,
    which sends it as `Authorization: Bearer <token>`

Declaring the OAuth2 scheme is what makes FastAPI mark these routes as secured
(lock icon) and render the Authorize dialog in the docs. `auto_error=False` lets
us raise our own enveloped 401 instead of FastAPI's default.
"""

from fastapi import Depends, Request, Security
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session

from app.config import settings
from app.db import get_session
from app.exceptions import AuthenticationError, AuthorizationError
from app.models import User
from app.services.auth import decode_session_cookie

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token", auto_error=False)


def get_current_user(
    request: Request,
    bearer_token: str | None = Security(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    """Resolve the authenticated user from the bearer token or session cookie.

    Bearer (Swagger/API clients) takes precedence over the cookie (SPA). Raises
    AuthenticationError (401) when no token is present, it's invalid/expired, or
    the user no longer exists / has been deactivated.
    """
    token = bearer_token or request.cookies.get(settings.session_cookie_name)
    if not token:
        raise AuthenticationError()

    user_id = decode_session_cookie(token)
    if user_id is None:
        raise AuthenticationError()

    user = session.get(User, user_id)
    if user is None or not user.is_active:
        # Deactivated/deleted accounts: their token is no longer valid.
        raise AuthenticationError()

    return user


def get_current_admin(user: User = Depends(get_current_user)) -> User:
    """Require the authenticated user to be an admin (403 otherwise)."""
    if not user.is_admin:
        raise AuthorizationError()
    return user


# Spec name for the admin router (QT-C). A thin alias over the canonical
# get_current_admin so the dependency logic lives in exactly one place — the
# bot/discord path keeps importing get_current_admin, the web admin routes use
# require_admin. Both resolve to the same 403-on-non-admin check.
require_admin = get_current_admin
