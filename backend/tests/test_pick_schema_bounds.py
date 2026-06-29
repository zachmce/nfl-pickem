"""Pure-schema bounds tests for the pick request models (offline, no DB).

Pins the anti-abuse Field bounds added so overlong / out-of-range / oversized
input is rejected as a 422 at request validation BEFORE any DB write (T-nus-03),
instead of becoming a Postgres 500. These instantiate the Pydantic models
directly and assert ``ValidationError`` — no DB or TestClient needed (faster than
driving the full API), and the same field bounds are what FastAPI surfaces as a
422 envelope at the route boundary.

Covered bounds:
* PickItem.misc_text / AdminPickSetRequest.misc_text: max_length=280 (mirrors the
  VARCHAR(280) Pick.misc_text column).
* PickSubmitRequest.picks: max_length cap (a generous-but-finite batch size).
* AdminMiscGradeRequest.points: an intentional MISC anti-abuse range (NOT a
  scoring rule — scoring intentionally does not clamp MISC points).

At-bound values (280-char text, cap-sized batch, points at ge and le) are ACCEPTED.

> Run from backend/ with ``.venv/bin/python -m unittest`` (unittest, NOT pytest).
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from app.models import PickResult, PickType
from app.schemas.admin_picks import AdminMiscGradeRequest, AdminPickSetRequest
from app.schemas.picks import (
    PICKS_BATCH_MAX,
    MISC_POINTS_MAX,
    MISC_POINTS_MIN,
    MISC_TEXT_MAX,
    PickItem,
    PickSubmitRequest,
)


def _pick_item() -> dict:
    return {"game_id": 1, "pick_type": "FAVORITE_COVER"}


class MiscTextBoundsTests(unittest.TestCase):
    """misc_text is capped at the column width (280) on both request shapes."""

    def test_pick_item_misc_text_at_bound_ok(self) -> None:
        PickItem(game_id=1, pick_type=PickType.MISC, misc_text="x" * MISC_TEXT_MAX)

    def test_pick_item_misc_text_over_bound_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            PickItem(
                game_id=1, pick_type=PickType.MISC, misc_text="x" * (MISC_TEXT_MAX + 1)
            )

    def test_pick_item_misc_text_none_ok(self) -> None:
        PickItem(game_id=1, pick_type=PickType.FAVORITE_COVER, misc_text=None)

    def test_admin_set_misc_text_at_bound_ok(self) -> None:
        AdminPickSetRequest(
            game_id=1, pick_type=PickType.MISC, misc_text="y" * MISC_TEXT_MAX
        )

    def test_admin_set_misc_text_over_bound_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AdminPickSetRequest(
                game_id=1, pick_type=PickType.MISC, misc_text="y" * (MISC_TEXT_MAX + 1)
            )


class SubmitBatchBoundsTests(unittest.TestCase):
    """The submit batch is bounded between 1 and the generous cap."""

    def test_single_item_ok(self) -> None:
        req = PickSubmitRequest(season=2025, week=1, picks=[_pick_item()])
        self.assertEqual(len(req.picks), 1)

    def test_empty_batch_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            PickSubmitRequest(season=2025, week=1, picks=[])

    def test_cap_sized_batch_ok(self) -> None:
        req = PickSubmitRequest(
            season=2025, week=1, picks=[_pick_item() for _ in range(PICKS_BATCH_MAX)]
        )
        self.assertEqual(len(req.picks), PICKS_BATCH_MAX)

    def test_oversized_batch_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            PickSubmitRequest(
                season=2025,
                week=1,
                picks=[_pick_item() for _ in range(PICKS_BATCH_MAX + 1)],
            )


class MiscPointsBoundsTests(unittest.TestCase):
    """MISC grade points are bounded to a sane anti-abuse range (not a scoring rule)."""

    def test_points_at_lower_bound_ok(self) -> None:
        req = AdminMiscGradeRequest(result=PickResult.LOSS, points=MISC_POINTS_MIN)
        self.assertEqual(req.points, MISC_POINTS_MIN)

    def test_points_at_upper_bound_ok(self) -> None:
        req = AdminMiscGradeRequest(result=PickResult.WIN, points=MISC_POINTS_MAX)
        self.assertEqual(req.points, MISC_POINTS_MAX)

    def test_typical_points_ok(self) -> None:
        for p in (-1, 0, 1, 3):
            AdminMiscGradeRequest(result=PickResult.WIN, points=p)

    def test_points_below_bound_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AdminMiscGradeRequest(result=PickResult.WIN, points=MISC_POINTS_MIN - 1)

    def test_points_above_bound_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AdminMiscGradeRequest(result=PickResult.WIN, points=MISC_POINTS_MAX + 1)


if __name__ == "__main__":
    unittest.main()
