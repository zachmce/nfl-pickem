"""Request schema for the admin pick-override API (QT-1).

Mirrors the Pydantic v2 ``BaseModel`` + ``ConfigDict(extra="forbid")`` style of
:mod:`app.schemas.picks` and :mod:`app.schemas.admin`. Only a request shape is
defined here: responses reuse :class:`app.schemas.picks.PickRead` (the GET returns
``list[PickRead]``, the PUT returns the updated ``PickRead``) so the read contract
lives in exactly one place.

``season`` / ``week`` are deliberately NOT in the body — they are route query
params (mirroring the user-facing picks DELETE shape) so the target slot identity
is unambiguous. The acting admin (caller) and the target user are NOT in the body
either: the admin comes from the verified session, the target from the path
``{user_id}`` — there is no actor field a client can spoof.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models import PickResult, PickType
from app.schemas.picks import MISC_POINTS_MAX, MISC_POINTS_MIN, MISC_TEXT_MAX


class AdminPickSetRequest(BaseModel):
    """An admin set/add/change of one ``{game_id, pick_type, lock}`` slot.

    The slot's ``{season, week}`` are route query params, not body fields.
    """

    model_config = ConfigDict(extra="forbid")

    game_id: int
    pick_type: PickType
    is_mortal_lock: bool = False
    # Carries the free-text prediction on a RETROACTIVE admin create of a MISC
    # pick (so an admin can author the MISC text for a user who missed it). NULL
    # for every other pick type. Validation of "MISC requires text" is the
    # user-facing concern; the retroactive admin path mirrors the column only.
    # Capped at the VARCHAR(280) column width so overlong text is a 422, not a
    # DB error.
    misc_text: str | None = Field(default=None, max_length=MISC_TEXT_MAX)


class AdminMiscGradeRequest(BaseModel):
    """An admin grade of a user's MISC pick: mark correct/incorrect + set points.

    The target user is the route ``{user_id}`` and the ``{season, week}`` are
    route query params, so the body carries only the grade itself. ``result`` is
    deliberately a :class:`~app.models.PickResult`; grading must DECIDE the pick,
    so ``PENDING`` is rejected in the service
    (:func:`app.services.admin_picks.admin_grade_misc`, reason
    ``misc_grade_must_decide``) rather than at the schema level — keeping the
    "must decide" rule in one place next to the mutation it guards.
    """

    model_config = ConfigDict(extra="forbid")

    result: PickResult
    # Anti-abuse bound, NOT a scoring rule: scoring intentionally does not clamp
    # MISC points (any admin-set int is a legitimate grade), so this symmetric
    # range is chosen generously enough to cover every real grade while rejecting
    # absurd/abusive values as a 422 before the DB write.
    points: int = Field(ge=MISC_POINTS_MIN, le=MISC_POINTS_MAX)
