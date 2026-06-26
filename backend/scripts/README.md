# `backend/scripts/` — dev/test tooling

Standalone developer tooling that is **deliberately separate from the production
ingest path** in `backend/app/`. Nothing here runs as part of the application,
Celery workers, or any automatic schedule. Scripts are run manually by a
developer.

> **Production seeder lives elsewhere.** The 32-team NFL reference seeder is
> production reference data (`Game` rows FK to `team.id`), so it lives under
> `backend/app/seeds/`, **not** here. See
> [Production team seeder](#production-team-seeder--appseedsteams) below.

---

## `gen_2025_fixture.py` — one-time 2025 NFL test-data fixture generator

### What it is

A one-time generator that pulls the **2025 NFL regular season** (seasontype=2,
weeks 1–18) from public ESPN endpoints and writes a single normalized JSON
fixture. Its purpose is to produce ground-truth 2025 data — real final scores
plus stand-in odds — so the future scoring engine can be tested **before** the
production Game/Odds schema exists.

It uses **only the Python standard library** (`urllib` + `json`); it adds no
dependencies and imports nothing from `backend/app/`.

### Honest labeling of odds — read this

The odds in the fixture are **ESPN BET stand-in lines, NOT real DraftKings
lines.** 2025 has **no DraftKings odds available anywhere**: once a game finals,
the DraftKings line disappears from the site API and was never stored in the core
API (which only ever retained ESPN BET historically). ESPN BET is close enough to
DraftKings to validate scoring logic, but it is labeled honestly in the fixture
metadata (`odds_provider: "ESPN BET"`) and must not be presented as real DK data.

### Data sources

| Field group                              | Source                                  |
| ---------------------------------------- | --------------------------------------- |
| Games / scores / status / teams          | ESPN **site scoreboard** (per week)     |
| Spread / total / favorite-underdog odds  | ESPN **core API** ESPN BET (per game)   |

### Run command

```bash
cd backend
python -m scripts.gen_2025_fixture
```

> Note: on this machine the interpreter is `python3` (there is no bare `python`
> on `PATH`); use `python3 -m scripts.gen_2025_fixture` if `python` is not found.

Optional flags:

| Flag             | Default                                              | Purpose                                          |
| ---------------- | --------------------------------------------------- | ------------------------------------------------ |
| `--weeks 1-2`    | `1-18`                                               | Pull a subset (also accepts `3` or `1,2,5`) for a quick smoke run. |
| `--out PATH`     | `backend/tests/fixtures/nfl_2025_regular_season.json` | Override the output path.                       |
| `--delay 0.4`    | `0.4`                                                | Polite delay (seconds) between per-game odds requests. |

### Output and idempotency

The script writes pretty-printed JSON to
`backend/tests/fixtures/nfl_2025_regular_season.json` by default. The document
has a top-level `metadata` block (`generated_at`, `season`, `season_type`,
`source`, `odds_provider`, the ESPN-BET honesty note, and with/without-odds
counts) and a `games` array. **Re-running overwrites the existing file cleanly —
the operation is idempotent.**

### Cost / runtime

A full run pulls 18 weekly scoreboards and then makes **one odds request per
game** — roughly **~270 events**, so a few hundred polite sequential requests in
total. Expect the full run to take a couple of minutes. Games whose odds are
unavailable simply get `odds: null` and the run continues; the final summary line
reports the with/without-odds counts.

### Tests

The pure normalization logic is covered by offline unit tests (no network):

```bash
cd backend
python -m unittest tests.test_gen_2025_fixture -v
```

---

## `backfill_fixture_odds.py` — one-shot REAL-odds backfill for the 2025 fixture

### What it is

A one-shot, **idempotent** backfill that fills in the **79 odds-less games** of
`backend/app/seeds/data/nfl_2025_regular_season.json` (the wk13 NYG@NE gap game
plus all of weeks 14–18) so the full 18-week 2025 demo season becomes
**pickable and gradeable**.

### Why it's needed

`gen_2025_fixture.py` only accepts the **ESPN BET** provider (name `ESPN BET` /
id `58`). For finaled weeks 14–18 (and one wk13 game) ESPN BET lines have
disappeared — but the **same** core-API odds endpoint still returns **real
DraftKings** lines (provider id `100`) in the exact shape
`normalize_odds` already consumes. This script adds a **permissive selector**
(`select_odds_item_permissive`: ESPN BET → id 58 → first item that yields a
*usable* line) that unblocks those games. It **reuses** `normalize_odds` and
`parse_team_id_from_ref` from `gen_2025_fixture` and does **not** modify the
generator (its ESPN-BET-first behavior and tests stay untouched).

A "usable line" requires **all four** of `spread`, `total`, `favorite_team_id`,
`underdog_team_id` to be non-null. Partial/empty lines are **never written** —
such a game is left `odds: null` and **reported**, so the importer never crashes
on `abs(None)` / `int(None)`.

### Run command

```bash
cd backend
.venv/bin/python -m scripts.backfill_fixture_odds
```

> Performs outbound HTTP to ESPN's core API (one GET per still-null game, with a
> polite `--delay` default 0.4s). `--path PATH` overrides the fixture file.

### Idempotency & safety

- Games that **already** have odds are skipped untouched — never re-fetched,
  never reordered. Re-running after a successful backfill is a no-op for filled
  games.
- The merge edits each filled game's `odds` **in place**; game order, game count
  (272), and weeks 1–13 data are preserved byte-for-byte. The diff is limited to
  the previously-null games + the `metadata` block (recomputed
  `games_with_odds` / `games_without_odds`, a refreshed `note`, and a new
  `backfilled_at` timestamp; the original `generated_at` is left intact).
- Each backfilled odds object is labeled by its **real** provider via
  `provider_label` (e.g. `"DraftKings"`), not a hardcoded constant.

### Accepted provider-label inconsistency (intentional)

The fixture now labels wk1–13 odds `"ESPN BET"` and the backfilled games by their
real provider (mostly `"DraftKings"`). However, **`app.seeds.fixture_2025.py`
still hardcodes `game.odds_provider = "ESPN BET"`** in the DB regardless of the
per-game provider. This is left **as-is intentionally**: `odds_provider` is not
surfaced in the slate API, so it is cosmetic only. Reading the per-game provider
in the importer is an **optional follow-up**, not part of this backfill.

### Tests

The pure selection/merge logic is covered by offline unit tests (no network):

```bash
cd backend
.venv/bin/python -m unittest tests.test_backfill_fixture_odds -v
```

---

## Production team seeder — `app.seeds.teams`

> Not in `scripts/`. Documented here for discoverability; the seeder itself lives
> at `backend/app/seeds/teams.py` because it writes **production** reference data.

### What it is

An **idempotent** seeder that populates the `team` table with all 32 NFL teams
(canonical ESPN ids, abbreviations, and full display names). Unlike the dev-only
generators in this directory, it is production reference data: every `Game` row
FKs to `team.id`, so the team table **must be seeded before any game ingest**.

The seeder upserts each team keyed on the unique `espn_team_id` column (never on
the surrogate PK), so it is **safe to re-run**: a second run leaves exactly 32
rows (no duplicates) and corrects any drifted abbreviation/display_name back to
the canonical value.

### Run command

```bash
cd backend
python -m app.seeds.teams
```

> Note: on this machine the interpreter is `python3` (there is no bare `python`
> on `PATH`); use `python3 -m app.seeds.teams` or the venv interpreter
> `.venv/bin/python -m app.seeds.teams` if `python` is not found.

### Tests

The canonical table and idempotent-upsert behavior are covered by offline unit
tests (no network, no Postgres — in-memory SQLite):

```bash
cd backend
python -m unittest tests.test_seed_teams -v
```
