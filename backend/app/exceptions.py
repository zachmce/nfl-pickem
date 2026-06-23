"""Typed API exception hierarchy for the unified error envelope.

``ApiException`` extends plain ``Exception`` — NOT ``HTTPException`` — so a
global handler can translate it into the envelope shape
``{"error": {"code", "message", "reason"?, "fields"?}}`` without FastAPI's
built-in handler short-circuiting it. Ported alongside the auth service; the
bot only needs ``InvalidCredentialsError`` today, the rest are here for the
forthcoming FastAPI auth routes.
"""

from __future__ import annotations

from typing import ClassVar


class ApiException(Exception):
    """Base class for envelope-emitting API exceptions."""

    code: ClassVar[str] = "internal_error"
    status: ClassVar[int] = 500
    message: ClassVar[str] = "An error occurred"

    def __init__(
        self,
        message: str | None = None,
        *,
        reason: str | None = None,
        fields: dict[str, list[str]] | None = None,
    ) -> None:
        self.message: str = message if message is not None else type(self).message
        self.reason: str | None = reason
        self.fields: dict[str, list[str]] | None = fields
        super().__init__(self.message)


class ValidationError(ApiException):
    """422 — request body / query / path failed schema validation."""

    code: ClassVar[str] = "validation_error"
    status: ClassVar[int] = 422
    message: ClassVar[str] = "Request validation failed"


class InvalidCredentialsError(ApiException):
    """401 — login credentials rejected. No info leak about which check failed."""

    code: ClassVar[str] = "invalid_credentials"
    status: ClassVar[int] = 401
    message: ClassVar[str] = "Invalid login credentials"


class AuthenticationError(ApiException):
    """401 — request requires authentication (no/invalid session cookie)."""

    code: ClassVar[str] = "unauthorized"
    status: ClassVar[int] = 401
    message: ClassVar[str] = "Authentication required"


class AuthorizationError(ApiException):
    """403 — authenticated user lacks permission for this action."""

    code: ClassVar[str] = "forbidden"
    status: ClassVar[int] = 403
    message: ClassVar[str] = "Forbidden"


class NotFoundError(ApiException):
    """404 — requested resource does not exist."""

    code: ClassVar[str] = "not_found"
    status: ClassVar[int] = 404
    message: ClassVar[str] = "Not found"


class ConflictError(ApiException):
    """409 — request conflicts with current state (unique violation, etc.)."""

    code: ClassVar[str] = "conflict"
    status: ClassVar[int] = 409
    message: ClassVar[str] = "Conflict"


class InternalError(ApiException):
    """500 — internal invariant violation caught deliberately."""

    code: ClassVar[str] = "internal_error"
    status: ClassVar[int] = 500
    message: ClassVar[str] = "An internal server error occurred"
