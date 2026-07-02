"""Atheris harness for the pure scoring engine (app.services.scoring).

Targets ``grade_pick`` and ``score_week``. With well-typed inputs (mirroring the
real column types) neither should raise: ``grade_pick`` guards non-FINAL / missing
scores to UNGRADEABLE and passes MISC points through, and ``score_week`` sees every
``game_id`` pre-seeded into ``games_by_id`` so the documented ``KeyError`` path is
never hit. Any exception here is therefore a genuine finding — no try/except.
"""

import sys

import atheris

# Import the heavy third-party chain (sqlalchemy/pydantic via app.models)
# UNINSTRUMENTED — we only want coverage on our own module under test.
import app.models  # noqa: F401  (pulls the ORM import chain before instrumentation)

with atheris.instrument_imports():
    from app.services.scoring import grade_pick, score_week
    from fuzz._domain import build_game, build_pick


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)

    # grade_pick: one game + one pick sharing a game_id.
    grade_pick(build_game(fdp, game_id=1), build_pick(fdp, game_id=1))

    # score_week: a handful of picks with every referenced game_id seeded, so a
    # spurious KeyError can't mask a real crash.
    picks = [build_pick(fdp) for _ in range(fdp.ConsumeIntInRange(0, 6))]
    games_by_id = {p.game_id: build_game(fdp, game_id=p.game_id) for p in picks}
    score_week(games_by_id, picks)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
