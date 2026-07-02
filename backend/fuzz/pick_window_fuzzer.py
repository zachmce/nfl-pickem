"""Atheris harness for the pure pick-window logic (app.services.pick_window).

Targets ``compute_window``, ``is_pick_open`` and ``is_game_locked``. These raise a
deliberate, documented ``ValueError`` on a naive datetime or an empty/kickoff-less
current week (via ``_require_aware``) — those are valid domain outcomes and are
caught. ANY OTHER exception (e.g. a bare ``TypeError`` from mixing naive/aware) is
a genuine finding and is allowed to propagate.
"""

import sys

import atheris

import app.models  # noqa: F401  (pulls the ORM import chain before instrumentation)

with atheris.instrument_imports():
    from app.services.pick_window import (
        compute_window,
        is_game_locked,
        is_pick_open,
    )
    from fuzz._domain import build_game, opt_datetime


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)

    week = [build_game(fdp) for _ in range(fdp.ConsumeIntInRange(0, 5))]
    prev = (
        [build_game(fdp) for _ in range(fdp.ConsumeIntInRange(0, 5))] if fdp.ConsumeBool() else None
    )

    window = None
    try:
        window = compute_window(week, prev)
    except ValueError:
        pass  # deliberate: empty week / no kickoff / naive kickoff

    if window is not None:
        now = opt_datetime(fdp)
        if now is not None:
            try:
                is_pick_open(window, now)
            except ValueError:
                pass  # deliberate: naive `now`

    now2 = opt_datetime(fdp)
    if now2 is not None:
        try:
            is_game_locked(build_game(fdp), now2)
        except ValueError:
            pass  # deliberate: naive `now` / naive kickoff


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
