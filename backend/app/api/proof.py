"""Proof endpoints demonstrating the two authorization tiers.

- GET /api/proof/authenticated — any logged-in user (401 otherwise)
- GET /api/proof/admin         — admins only (401 if anon, 403 if non-admin)
"""

from fastapi import APIRouter, Body, Depends

from app.api.deps import get_current_admin, get_current_user
from app.models import User

router = APIRouter(prefix="/api/proof", tags=["proof"])


@router.get("/authenticated")
def authenticated_only(user: User = Depends(get_current_user)) -> dict:
    """Accessible to any authenticated user."""
    return {
        "message": f"Hello {user.display_name}, you are authenticated.",
        "user_id": user.id,
        "is_admin": user.is_admin,
    }


@router.get("/admin")
def admin_only(user: User = Depends(get_current_admin)) -> dict:
    """Accessible only to authenticated admins."""
    return {
        "message": f"Hello {user.display_name}, you are an admin.",
        "user_id": user.id,
    }


@router.post("/echo")
def echo(
    message: str = Body("hello", embed=True),
    user: User = Depends(get_current_user),
) -> dict:
    """Authenticated POST — demonstrates CSRF on the cookie path.

    Via the cookie (SPA): requires a matching ``X-CSRF-Token`` header or returns
    403 ``csrf_failed``. Via a bearer token (Swagger): no CSRF needed.
    """
    return {"echo": message, "from": user.display_name}
