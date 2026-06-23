from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime

import structlog
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.config import get_settings
from app.exceptions import InvalidCredentialsError
from app.models import User
from app.schemas.auth import UserLoginRequest

logger = structlog.get_logger(__name__)

_ph = PasswordHasher()  # module-level singleton, thread-safe
_SESSION_SALT = "session"


def hash_password(plain: str) -> str:
    """Hash plain-text password with argon2id (default PasswordHasher params)."""
    return _ph.hash(plain)


def verify_password(stored_hash: str, plain: str) -> bool:
    """Verify a password against a stored argon2id hash.

    Catches VerifyMismatchError (wrong password), VerificationError (hash verification
    failed), and InvalidHashError (malformed / non-argon2 hash string). argon2-cffi
    25.x raises InvalidHashError (a ValueError subclass) for hashes that don't conform
    to the argon2 format — it is NOT a subclass of VerificationError.
    """
    try:
        return _ph.verify(stored_hash, plain)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_settings().secret_key, salt=_SESSION_SALT)


def create_session_cookie(user_id: int) -> str:
    """Produce a signed itsdangerous session token for the given user_id."""
    return _serializer().dumps({"uid": user_id})


def decode_session_cookie(token: str) -> int | None:
    """Decode + verify a session cookie. Returns user_id on success; None on any failure."""
    s = get_settings()
    try:
        data = _serializer().loads(token, max_age=s.session_max_age_days * 86400)
        return int(data["uid"])
    except (BadSignature, SignatureExpired, KeyError, ValueError, TypeError):
        return None


def login_user(session: Session, login_data: UserLoginRequest) -> User:
    user: User | None = session.exec(
        select(User).where(User.display_name == login_data.display_name)
    ).one_or_none()
    if user is None:
        logger.warning(
            "user_auth_failure",
            reason="invalid user name",
            display_name=login_data.display_name,
        )
        raise ValueError("Invalid Login Credentials")
    if user.password_hash is None:
        logger.warning(
            "user_auth_failure",
            reason="invalid user state (signup incomplete)",
            discord_id=user.discord_id,
            display_name=login_data.display_name,
        )
        raise ValueError("Signup Process not yet completed")
    if not verify_password(user.password_hash, login_data.password):
        logger.warning(
            "user_auth_failure",
            reason="invalid password",
            discord_id=user.discord_id,
            display_name=login_data.display_name,
        )
        raise ValueError("Invalid Login Credentials")

    logger.info("user_logged_in", user_id=user.id)
    return user


def change_password(
    session: Session,
    user_id: int,
    current_password: str,
    new_password: str,
) -> None:
    """Verify current password and replace the hash.

    Raises InvalidCredentialsError (401) if current_password does not verify
    against the stored hash, or if the user row is absent / has no hash.
    Does NOT touch the session cookie — it is a signed user_id, not
    password-bound.
    """
    user: User | None = session.get(User, user_id)
    if user is None or user.password_hash is None:
        raise InvalidCredentialsError()
    if not verify_password(user.password_hash, current_password):
        logger.warning(
            "password_change_failure",
            reason="wrong current password",
            user_id=user_id,
        )
        raise InvalidCredentialsError()
    user.password_hash = hash_password(new_password)
    session.add(user)
    session.commit()
    logger.info("password_changed", user_id=user_id)


def _generate_random_password() -> str:
    """Generate a cryptographically random temporary password (~95-bit entropy).

    Returns a 22-char URL-safe base64 string from secrets.token_urlsafe(16).
    """
    return secrets.token_urlsafe(16)


def _derive_safe_username(session: Session, discord_handle: str, discord_id: int) -> str:
    """Derive a collision-safe display_name from a Discord handle.

    Sanitization: lowercase the handle; replace spaces/dots with '_'; drop any
    char not in [a-z0-9_]; collapse consecutive '_' runs; strip leading/trailing '_'.
    Empty-result guard: if sanitization yields an empty string, use 'user_<last6hex>'.
    Collision loop: tries bare base, then base2..base100 (no base1), then
    base_<last6hex> hex fallback. Raises ValueError if all candidates are taken.
    Right-truncation: before appending any suffix, truncates base so
    base+suffix <= 100 chars total (display_name max_length=100).
    """
    base = discord_handle.lower()
    base = re.sub(r"[ .]", "_", base)
    base = re.sub(r"[^a-z0-9_]", "", base)
    base = re.sub(r"_+", "_", base)
    base = base.strip("_")

    last6hex = format(discord_id, "x")[-6:]

    if not base:
        base = f"user_{last6hex}"

    def _available(candidate: str) -> bool:
        return session.exec(select(User).where(User.display_name == candidate)).one_or_none() is None

    candidate = base[:100]
    if _available(candidate):
        return candidate

    for n in range(2, 101):
        suffix = str(n)
        truncated_base = base[: 100 - len(suffix)]
        candidate = f"{truncated_base}{suffix}"
        if _available(candidate):
            return candidate

    hex_suffix = f"_{last6hex}"
    truncated_base = base[: 100 - len(hex_suffix)]
    candidate = f"{truncated_base}{hex_suffix}"
    if _available(candidate):
        return candidate

    raise ValueError(
        f"Could not derive a unique display_name from handle {discord_handle!r}: "
        f"all 101 candidates taken (base loop + hex fallback)"
    )


def provision_user(session: Session, discord_id: int, discord_handle: str) -> tuple[int, str, str]:
    """Create a new pick'em account for a Discord user.

    Raises ValueError if discord_id already has an account.
    Returns (user_id, display_name, plain_password) — the plain_password is
    returned exactly once for the bot to DM; only the Argon2 hash is persisted.

    is_active=True is set explicitly — the model default is False.
    IntegrityError on 23505 is caught and translated to ValueError to handle
    rare concurrent registration races.
    """
    existing = session.exec(select(User).where(User.discord_id == discord_id)).one_or_none()
    if existing is not None:
        raise ValueError(f"Account already exists for discord_id {discord_id}")

    # Generate and hash the password BEFORE constructing User (atomic assignment;
    # if hash_password raises, the User object is never created).
    plain = _generate_random_password()
    password_hash = hash_password(plain)
    display_name = _derive_safe_username(session, discord_handle, discord_id)

    user = User(
        discord_id=discord_id,
        display_name=display_name,
        password_hash=password_hash,
        is_active=True,  # MUST be explicit — model default is False
        created_at=datetime.now(UTC),
    )
    session.add(user)
    try:
        session.commit()
        session.refresh(user)  # populate user.id after INSERT
    except IntegrityError as e:
        session.rollback()
        if getattr(e.orig, "sqlstate", None) == "23505":
            # Disambiguate which UNIQUE constraint fired. A discord_id collision
            # means this Discord user already has an account (concurrent /register
            # race). A display_name collision is a transient conflict that resolves
            # on retry.
            constraint_name: str = getattr(getattr(e.orig, "diag", None), "constraint_name", "") or ""
            if "discord_id" in constraint_name:
                raise ValueError("You already have a pick'em account — log in instead") from e
            raise ValueError("Username taken by concurrent registration — try again") from e
        raise

    logger.info(
        "user_provisioned",
        user_id=user.id,
        discord_id=discord_id,
        display_name=display_name,
    )
    assert user.id is not None  # type guard: id is populated after commit+refresh
    return (user.id, display_name, plain)


def reset_password_for_discord(session: Session, discord_id: int) -> str:
    """Regenerate the password for an existing active account.

    Raises ValueError if discord_id has no account or if the account is
    deactivated. Returns plain_password (str) — returned once for the bot to DM.
    """
    user: User | None = session.exec(select(User).where(User.discord_id == discord_id)).one_or_none()
    if user is None:
        raise ValueError(f"No account found for discord_id {discord_id}")
    if not user.is_active:
        raise ValueError("Account is deactivated — contact an admin to reactivate")

    plain = _generate_random_password()
    user.password_hash = hash_password(plain)
    session.add(user)
    session.commit()
    logger.info("password_reset_discord", user_id=user.id, discord_id=discord_id)
    return plain


def deactivate_user_by_discord_id(session: Session, discord_id: int) -> None:
    """Deactivate a user account by Discord ID.

    Raises ValueError if discord_id has no account or is already deactivated.
    """
    user: User | None = session.exec(select(User).where(User.discord_id == discord_id)).one_or_none()
    if user is None:
        raise ValueError(f"No account found for discord_id {discord_id}")
    if not user.is_active:
        raise ValueError("Account is already deactivated")

    user.is_active = False
    session.add(user)
    session.commit()
    logger.info("user_deactivated_discord", user_id=user.id, discord_id=discord_id)


def reactivate_user_by_discord_id(session: Session, discord_id: int) -> None:
    """Reactivate a previously deactivated user account by Discord ID.

    Raises ValueError if discord_id has no account or is already active.
    """
    user: User | None = session.exec(select(User).where(User.discord_id == discord_id)).one_or_none()
    if user is None:
        raise ValueError(f"No account found for discord_id {discord_id}")
    if user.is_active:
        raise ValueError("Account is already active")

    user.is_active = True
    session.add(user)
    session.commit()
    logger.info("user_reactivated_discord", user_id=user.id, discord_id=discord_id)


def grant_admin_by_discord_id(session: Session, discord_id: int) -> None:
    """Grant admin privileges to a user by Discord ID.

    Raises ValueError if discord_id has no account or is already an admin.
    """
    user: User | None = session.exec(select(User).where(User.discord_id == discord_id)).one_or_none()
    if user is None:
        raise ValueError(f"No account found for discord_id {discord_id}")
    if user.is_admin:
        raise ValueError("User is already an admin")

    user.is_admin = True
    session.add(user)
    session.commit()
    logger.info("admin_granted_discord", user_id=user.id, discord_id=discord_id)


def revoke_admin_by_discord_id(
    session: Session,
    caller_discord_id: int,
    target_discord_id: int,
) -> None:
    """Revoke admin privileges from a user by Discord ID.

    Self-demote guard: raises ValueError when caller_discord_id == target_discord_id.
    The 'demote last admin' guard is explicitly deferred. Only the self-demote rail
    is implemented. Raises ValueError if target has no account or is not an admin.
    """
    # Self-demote guard — checked BEFORE the DB lookup.
    if caller_discord_id == target_discord_id:
        raise ValueError("Cannot remove your own admin access")

    user: User | None = session.exec(select(User).where(User.discord_id == target_discord_id)).one_or_none()
    if user is None:
        raise ValueError(f"No account found for discord_id {target_discord_id}")
    if not user.is_admin:
        raise ValueError("Target user is not an admin")

    user.is_admin = False
    session.add(user)
    session.commit()
    logger.info(
        "admin_revoked_discord",
        user_id=user.id,
        target_discord_id=target_discord_id,
        caller_discord_id=caller_discord_id,
    )


def get_account_by_discord_id(session: Session, discord_id: int) -> str | None:
    """Return the display_name for an existing account, or None if absent.

    Returns a plain str value — never an ORM User object (so no password_hash can
    leak). Never raises. No is_active gate — the read path only needs to know
    whether an account exists.
    """
    user: User | None = session.exec(select(User).where(User.discord_id == discord_id)).one_or_none()
    return user.display_name if user is not None else None


def is_admin_by_discord_id(session: Session, discord_id: int) -> bool:
    """Return True only when a row with that discord_id has is_admin=True.

    Gates on existence + is_admin ONLY — NOT is_active: a deactivated admin can
    still authorize admin commands. The NULL-discord_id seed admin is unreachable
    here by design — that row's discord_id never equals any integer discord_id.
    """
    user: User | None = session.exec(select(User).where(User.discord_id == discord_id)).one_or_none()
    return bool(user is not None and user.is_admin)


def delete_user_by_id(session: Session, user_id: int) -> None:
    """Hard-DELETE a user row by primary key.

    Raises ValueError if user_id is absent — the rollback path (DM-failure
    compensate) must never silently no-op on a missing row.

    Hard-delete is intentional: soft-deactivation would leave an orphaned row that
    prevents re-registration by the same Discord ID.
    """
    user: User | None = session.get(User, user_id)
    if user is None:
        raise ValueError(f"No account found for user_id {user_id}")
    session.delete(user)
    session.commit()
    logger.info("user_deleted_rollback", user_id=user_id)
