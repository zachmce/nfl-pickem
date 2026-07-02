"""Atheris harness for the pure pick-validation service (app.services.pick_validation).

Targets ``validate_roster``, ``check_new_pick`` and the shared predicate
``is_pick_type_eligible``. Every referenced ``game_id`` is seeded into
``games_by_id`` so the documented ``KeyError`` (missing-game programmer error) is
never provoked — any exception here is a genuine finding, so no try/except.
"""

import sys

import atheris

import app.models  # noqa: F401  (pulls the ORM import chain before instrumentation)
from app.models import PickType

with atheris.instrument_imports():
    from app.services.pick_validation import (
        check_new_pick,
        is_pick_type_eligible,
        validate_roster,
    )
    from fuzz._domain import build_game, build_pick


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)

    picks = [build_pick(fdp) for _ in range(fdp.ConsumeIntInRange(0, 8))]
    games_by_id = {p.game_id: build_game(fdp, game_id=p.game_id) for p in picks}
    validate_roster(picks, games_by_id)

    # check_new_pick against that same accepted set (seed the new game_id too).
    new_pick = build_pick(fdp)
    games_by_id.setdefault(new_pick.game_id, build_game(fdp, game_id=new_pick.game_id))
    check_new_pick(new_pick, picks, games_by_id)

    # The shared eligibility predicate across every pick type on one game.
    game = build_game(fdp, game_id=1)
    for pick_type in PickType:
        is_pick_type_eligible(game, pick_type)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
