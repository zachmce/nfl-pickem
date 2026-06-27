"""Cookie-based authentication routes: login, logout, and the current-user probe.

Login verifies credentials via the shared auth service, then sets a signed,
HttpOnly session cookie the browser sends automatically on subsequent requests.
"""

from fastapi import APIRouter, Depends, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session

from app.api.deps import get_current_user
from app.config import settings
from app.csrf import CSRF_COOKIE_NAME, issue_csrf_token, set_csrf_cookie
from app.db import get_session
from app.exceptions import InvalidCredentialsError
from app.models import User
from app.schemas.auth import LogoutResponse, TokenResponse, UserLoginRequest, UserRead
from app.services.auth import create_session_cookie, login_user
from app.services.notifications import login_event, publish_event

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _to_user_read(user: User) -> UserRead:
    return UserRead(
        id=user.id,
        discord_id=user.discord_id,
        display_name=user.display_name,
        is_admin=user.is_admin,
        is_active=user.is_active,
    )


def _set_session_cookie(response: Response, user_id: int) -> None:
    response.set_cookie(
        key=settings.session_cookie_name,
        value=create_session_cookie(user_id),
        max_age=settings.session_max_age_days * 86400,
        httponly=True,  # not readable by JS — mitigates XSS token theft
        samesite="lax",
        secure=settings.session_cookie_secure,
        path="/",
    )


@router.post("/login", response_model=UserRead)
def login(
    payload: UserLoginRequest,
    response: Response,
    session: Session = Depends(get_session),
) -> UserRead:
    """Verify credentials and set the session cookie.

    Returns 401 (invalid_credentials) for bad username/password, an account whose
    signup never completed, or a deactivated account — without leaking which.
    """
    try:
        user = login_user(session, payload)
    except ValueError as exc:
        raise InvalidCredentialsError() from exc

    if not user.is_active:
        raise InvalidCredentialsError()

    assert user.id is not None  # persisted user always has an id
    _set_session_cookie(response, user.id)
    # Issue a CSRF token so the SPA can protect subsequent cookie-auth mutations.
    set_csrf_cookie(response, issue_csrf_token())
    # Post-commit, best-effort: announce the login to the Discord pipe. This site
    # is the success path only (after is_active/credential checks), and the
    # request's get_session dependency commits when this handler returns — so a
    # rejected/rolled-back login never announces. publish_event is best-effort
    # internally (swallows a Redis outage), so the route needs no extra try/except.
    # Display name only — nothing sensitive crosses to Discord.
    publish_event(login_event(user.display_name))
    return _to_user_read(user)


@router.get("/csrf")
def csrf(response: Response) -> dict:
    """Issue/refresh the CSRF cookie and return the token.

    The SPA calls this on load, then sends the value as the ``X-CSRF-Token``
    header on unsafe (POST/PUT/PATCH/DELETE) cookie-authenticated requests.
    """
    token = issue_csrf_token()
    set_csrf_cookie(response, token)
    return {"csrf_token": token}


@router.post("/token", response_model=TokenResponse)
def token(
    form: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
) -> TokenResponse:
    """OAuth2 password-flow token endpoint — backs Swagger's Authorize button.

    Takes form-encoded `username` (= display_name) + `password` and returns the
    same signed session token the cookie carries, as a bearer token. The SPA does
    not use this — it uses POST /login (cookie). 401 on bad/inactive credentials.
    """
    try:
        user = login_user(session, UserLoginRequest(display_name=form.username, password=form.password))
    except ValueError as exc:
        raise InvalidCredentialsError() from exc
    if not user.is_active:
        raise InvalidCredentialsError()

    assert user.id is not None
    return TokenResponse(access_token=create_session_cookie(user.id))


@router.post("/logout", response_model=LogoutResponse)
def logout(response: Response) -> LogoutResponse:
    """Clear the session + CSRF cookies. Idempotent — safe when not logged in."""
    response.delete_cookie(key=settings.session_cookie_name, path="/")
    response.delete_cookie(key=CSRF_COOKIE_NAME, path="/")
    return LogoutResponse(message="logged_out")


@router.get("/me", response_model=UserRead)
def me(user: User = Depends(get_current_user)) -> UserRead:
    """Return the currently authenticated user (used by the SPA to bootstrap auth state)."""
    return _to_user_read(user)
