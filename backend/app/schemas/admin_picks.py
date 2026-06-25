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

from pydantic import BaseModel, ConfigDict

from app.models import PickType


class AdminPickSetRequest(BaseModel):
    """An admin set/add/change of one ``{game_id, pick_type, lock}`` slot.

    The slot's ``{season, week}`` are route query params, not body fields.
    """

    model_config = ConfigDict(extra="forbid")

    game_id: int
    pick_type: PickType
    is_mortal_lock: bool = False
