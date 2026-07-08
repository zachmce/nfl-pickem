"""Double-submit-cookie CSRF protection for the cookie auth path.

Why this shape:
  - CSRF only threatens **cookie** auth (the browser attaches the session cookie
    automatically on cross-site requests). **Bearer** auth is immune — an
    attacker can't set the Authorization header cross-site. So we enforce CSRF
    only when a request is cookie-authenticated AND carries no bearer token.
    => Swagger's Authorize (bearer) flow is never affected.
  - Double-submit: the server issues a non-HttpOnly ``csrftoken`` cookie; the SPA
    reads it and echoes it in the ``X-CSRF-Token`` header on unsafe requests. The
    middleware requires the two to match. A cross-site attacker can neither read
    the cookie (same-origin policy) nor set the custom header, so it can't forge
    the pair.

The login/token/logout endpoints are exempt: login/token are the pre-auth
bootstrap, and logout is deliberately exempt so the all-cookie Swagger workflow
(POST /login then POST /logout via "Try it out") keeps working — forcing a logout
is a negligible-severity CSRF target.
"""

import secrets

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import settings

CSRF_COOKIE_NAME = "csrftoken"
CSRF_HEADER_NAME = "x-csrf-token"

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
_EXEMPT_PATHS = frozenset(
    {"/api/auth/login", "/api/auth/token", "/api/auth/logout", "/api/auth/csrf"}
)


def issue_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def set_csrf_cookie(response: Response, token: str) -> None:
    """Set the CSRF cookie. NOT HttpOnly — the SPA must read it to echo it back."""
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=token,
        max_age=settings.session_max_age_days * 86400,
        httponly=False,
        samesite="lax",
        secure=settings.session_cookie_secure,
        path="/",
    )


class CSRFMiddleware:
    """Enforce double-submit CSRF on unsafe, cookie-authenticated requests.

    A pure-ASGI class middleware. It needs only the scope's
    method/path/headers/cookies, never the body, so it wraps the app directly
    and avoids the response-buffering and contextvar-propagation caveats of
    Starlette's dispatch-function base middleware. Non-http scopes
    (websocket/lifespan) pass straight through; the 403 envelope is emitted via
    ``JSONResponse`` so status, content-type, and serialized body stay
    byte-identical to the prior behavior.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Request(scope) WITHOUT a receive is body-safe: reading .method,
        # .url.path, .headers, .cookies never consumes the body, and reuses the
        # exact header/cookie parsing the old dispatch used.
        request = Request(scope)
        if request.method not in _SAFE_METHODS and request.url.path not in _EXEMPT_PATHS:
            has_bearer = request.headers.get("authorization", "").lower().startswith("bearer ")
            cookie_session = request.cookies.get(settings.session_cookie_name)
            # Only cookie-authenticated requests need CSRF; bearer is exempt.
            if cookie_session and not has_bearer:
                cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
                header_token = request.headers.get(CSRF_HEADER_NAME)
                if (
                    not cookie_token
                    or not header_token
                    or not secrets.compare_digest(cookie_token, header_token)
                ):
                    response = JSONResponse(
                        status_code=403,
                        content={
                            "error": {
                                "code": "csrf_failed",
                                "message": "CSRF token missing or invalid",
                            }
                        },
                    )
                    await response(scope, receive, send)
                    return

        await self.app(scope, receive, send)
