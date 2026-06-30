# NFL Pick'em

> **Heads up — about this project.** This is a small NFL Pick'em app I built for a
> group of friends, so we'd have a nicer way to make our weekly picks than a group
> chat and a spreadsheet. I'll be honest: an AI/LLM coding agent wrote most of the
> code here — I drove it, reviewed it, and made the calls, but the bulk of the
> typing was the agent's. PRs are always welcome if you want to poke at it. It is
> **not** trying to be a slick, million-user product — it's a simple tool for a
> simple job that probably nobody outside our group actually needs. Take it in that
> spirit.

An NFL Pick'em web app **plus a Discord bot**, for a friend group. Each week you
make up to five main picks (plus an optional misc) within the pick window, they
lock at kickoff, and they're graded into a season-long scoreboard. The Discord bot
handles account signup and posts LLM-flavored recaps and commentary as games and
weeks play out.

## What it is

- **Weekly picks.** Up to **five main picks** a week — one of each of the four bet
  types, plus a Mortal Lock — and one optional misc pick. Every pick is optional.
  - **Favorite cover** — the favorite covers the spread.
  - **Underdog cover** — the underdog covers the spread.
  - **Over** — the two teams' combined score goes over the total.
  - **Under** — the combined score stays under the total.
  - **Mortal Lock** — a fifth "wildcard" main pick on any game and bet type you
    choose (the only same-type duplicate allowed); higher risk, higher reward.
  - **MISC** — a free-text prediction tied to a game, graded by an admin.

  At most one base pick of each of the four types per week, plus one mortal lock —
  enforced both in the domain logic and by DB partial-unique indexes.
- **Weekly slate + pick windows.** Each game has a pick window that **locks at
  kickoff**; you can edit a pick until then, and a locked game is read-only.
- **Scoring + season standings.** Graded picks roll up into a season matrix
  (`Rank | Player | W1…Wn | Total`) with competition ranking and your row
  highlighted.
- **Weekly results.** Per-week, per-player results — with a pick-leak gate so you
  never see another player's pick on a game that hasn't locked yet (you always see
  your own).
- **Calendar, rules, profile.** A season calendar view, a rules page, and a profile
  page with self-serve password change.
- **Admin tools.** Player management (deactivate / reactivate / grant-admin /
  revoke-admin / delete), per-user **pick override** (set/clear any user's pick on
  any game, bypassing the lock while keeping roster integrity, audited), season
  **ingest** + week **odds-freeze**, and a **bot-personality selector**.
- **Discord LLM chat personality.** An admin-selectable personality layer that posts
  recaps and commentary to a chat feed across game / window / week events, with
  team-logo app-emojis. The bot owns the deterministic facts; the LLM only phrases
  them, and there's a deterministic fallback on any LLM/DB failure. It talks to a
  **local / self-hosted OpenAI-compatible** LLM server (no cloud provider, no
  tool-calling). See [Discord bot](#discord-bot) below.
- **Demo mode.** A flag-gated, time-shifted 2025 season with seeded bot users and
  preordained picks, for walking a whole season in (shifted) real time. See
  [Demo mode](#demo-mode).

## The stack

| Service    | Stack                                              | Dev port |
| ---------- | -------------------------------------------------- | -------- |
| `backend`  | FastAPI (Python 3.14) + SQLModel                   | 8000     |
| `worker`   | Celery worker with **embedded beat** (refresh poller) | —     |
| `db`       | Postgres 17                                        | 5432     |
| `redis`    | Redis 7 (Celery broker + result store)             | 6379     |
| `migrate`  | Alembic + seed init container (runs on startup)    | —        |
| `bot`      | Discord bot (discord.py) + local-LLM chat layer    | —        |
| `frontend` | React 19 + Vite (dev) / nginx (prod)               | 5173     |
| `pgadmin`  | pgAdmin 4 (local dev DB console, no-auth)          | 5050     |

The `worker` runs `celery ... worker --beat`, so the `beat_schedule` (the
`refresh_games` poller) fires on its cadence and the season unspools on its own.

```
.
├── backend/            FastAPI app, Celery, SQLModel models, Alembic
│   ├── app/
│   │   ├── config.py        settings (env-driven)
│   │   ├── db.py            shared sync engine: get_session + task_session
│   │   ├── models.py        SQLModel tables (User, Game, Pick, …)
│   │   ├── exceptions.py    API exception hierarchy
│   │   ├── logging_config.py structlog JSON logging
│   │   ├── celery_app.py    Celery app + beat schedule
│   │   ├── tasks.py         Celery tasks (the refresh poller, season ingest, …)
│   │   ├── main.py          FastAPI app + router mounts
│   │   ├── api/            route modules (auth, picks, results, slate, …)
│   │   ├── schemas/         pydantic request/response models
│   │   ├── services/        business logic (auth, scoring, picks, …)
│   │   ├── seeds/          teams / demo / admins seeders
│   │   └── bot/             Discord bot + LLM chat layer (personality, recap, …)
│   └── alembic/        migration environment + versions/
├── frontend/           React + Vite SPA, nginx prod config
├── docker-compose.yml          dev stack
└── docker-compose.prod.yml     prod override (nginx SPA)
```

### One DB access method everywhere

The FastAPI app, the Celery worker/beat, AND the Discord bot all share a single
**synchronous** SQLModel engine (`app/db.py`). There is no async DB layer.

- FastAPI uses the `get_session()` dependency (one Session per request).
- Celery tasks and the bot use the `task_session()` context manager.
- The bot runs an async event loop (discord.py), so it never calls the sync DB
  directly — `app/bot/db_bridge.py` wraps each sync service call in
  `asyncio.to_thread(...)` so DB I/O happens on a worker thread and never blocks
  the gateway. Business logic lives once in `app/services/` and is reused by every
  surface.

## Prerequisites

- Docker + Docker Compose

## Getting started

```bash
cp .env.example .env
docker compose up --build      # or: make up
```

On startup the `migrate` init container runs `alembic upgrade head` and the seeders
against Postgres and exits; `backend` and `worker` wait for it to finish
(`service_completed_successfully`) before booting.

Then open:

- Frontend (Vite dev server): http://localhost:5173
- API docs (Swagger): http://localhost:8000/docs
- Health check: http://localhost:8000/api/health
- pgAdmin (local DB console): http://localhost:5050

First look: log in (accounts come from the Discord `/register` flow, or use the
demo bot users in demo mode), open **My Picks** to see the weekly slate and make
your five picks, then check **Standings** and **Weekly** results.

## Demo mode

Set `IS_DEMO_DATA=true` in `.env` and bring the stack up on an empty DB to seed a
**time-shifted 2025 season**: real games and odds, shifted so the season starts
~24h out and then plays in (shifted) real time. Seeded bot users come with
preordained picks (weeks 1–13), so the standings and weekly screens render real,
non-empty data while you walk the season — make picks now, watch them lock at the
shifted kickoffs, and see the beat poller roll results in.

When the flag is ON the API logs a loud demo banner at startup and demo data is
clearly labeled; when it's OFF (the prod path), the demo seed refuses to run and
the prod path is byte-for-byte unaffected.

## Production-style run (nginx serves the compiled SPA)

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build
# or: make prod
```

The frontend is built with `vite build` and served by nginx on
http://localhost:80, with `/api/*` proxied to the backend.

## API surface

All routes are mounted under `/api`. The main feature routers:

| Prefix              | What it does                                                  |
| ------------------- | ------------------------------------------------------------ |
| `/api/auth`         | login / token / csrf / logout / me / change-password         |
| `/api/picks`        | submit (POST), read (GET), clear (DELETE) your weekly picks   |
| `/api/results`      | `/week` (weekly results), `/standings` (season matrix)       |
| `/api/current-week` | the current pick'em week + window state                      |
| `/api/slate`        | the weekly game slate                                        |
| `/api/calendar`     | season calendar                                             |
| `/api/config`       | client-facing config                                        |
| `/api/admin`        | player mgmt, per-user pick override, ingest-season, freeze-week, bot-personality |

## Authentication (cookie-based)

Cookie sessions backed by the shared `app/services/auth.py`. Login signs a
session token (itsdangerous) and sets it as an **HttpOnly** cookie; the browser
sends it automatically, and FastAPI dependencies resolve the user from it.

| Method & path                    | Auth required | Purpose                                    |
| -------------------------------- | ------------- | ------------------------------------------ |
| `POST /api/auth/login`           | none          | Verify credentials, set session cookie     |
| `POST /api/auth/token`           | none          | OAuth2 password flow — backs Swagger auth  |
| `GET  /api/auth/csrf`            | none          | Issue/refresh the CSRF cookie + token      |
| `POST /api/auth/logout`          | none          | Clear the session + CSRF cookies           |
| `GET  /api/auth/me`              | any user      | Current user — SPA auth-state bootstrap    |
| `POST /api/auth/change-password` | any user      | Self-serve password change (8–128 chars)   |

The same signed token reaches the API two ways, so secured routes are marked
with a lock in Swagger and the **Authorize** button works:

- **SPA** → HttpOnly **cookie** set by `POST /api/auth/login`.
- **Swagger UI / API clients** → **bearer token** via the OAuth2 password flow
  (`POST /api/auth/token`). In `/docs`, click **Authorize**, enter a
  display_name + password, and locked endpoints become callable.

### CSRF protection (cookie path only)

Cookie auth is protected with a **double-submit-cookie** CSRF check
(`app/csrf.py`); bearer auth is exempt (it can't be triggered cross-site), so
**Swagger testing is unaffected**.

- `POST /api/auth/login` issues a readable `csrftoken` cookie (so does
  `GET /api/auth/csrf`, which the SPA calls on load to refresh it).
- On unsafe methods (POST/PUT/PATCH/DELETE), a **cookie-authenticated** request
  must send the token back in the `X-CSRF-Token` header; mismatch/missing → `403`
  `csrf_failed`. Safe methods (GET/HEAD) and bearer requests skip the check.
- Exempt endpoints: `login`, `token`, `logout`, `csrf`. (`change-password` is
  **not** exempt — a cookie-authed change must carry the CSRF header.)

SPA flow: `GET /api/auth/csrf` once on load, read the `csrftoken` cookie, and
send its value as `X-CSRF-Token` on every mutating request.

Failures use a JSON envelope `{"error": {"code", "message"}}`: `401`
(`invalid_credentials` / `unauthorized`) for bad or missing auth, `403`
(`forbidden`) for a non-admin hitting an admin route. Invalid password and
unknown user return the **same** 401 (no account-existence leak).

The dependency guards live in `app/api/deps.py`: `get_current_user` (401 if the
cookie is missing/expired or the account is gone/deactivated) and
`get_current_admin` (403 if not an admin).

```bash
# log in (accounts come from the Discord /register flow; password is DM'd)
curl -c jar.txt -X POST localhost:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"display_name":"your_name","password":"your_password"}'

curl -b jar.txt localhost:8000/api/auth/me        # 200 — current user
curl -b jar.txt -X POST localhost:8000/api/auth/logout
```

Relevant settings (`app/config.py`): `SECRET_KEY` (signs cookies — override in
prod), `SESSION_MAX_AGE_DAYS`, `SESSION_COOKIE_SECURE` (set `true` behind HTTPS),
and `CORS_ALLOWED_ORIGINS` (explicit origins are required for credentialed CORS).

## Discord bot

The bot (`app/bot/`) provides slash commands backed by `app/services/auth.py`:

- `/register` — provision a pick'em account for the invoking member; DMs a
  temporary password.
- `/reset-password` — rotate the member's password and DM the new one.
- `/admin deactivate|reactivate|grant-admin|revoke-admin @member` —
  authorization is gated on the DB `is_admin` column, never Discord roles.

### LLM chat personality

Beyond account commands, the bot runs an **admin-selectable chat-personality
layer** that posts recaps and commentary to a chat feed as games, pick windows,
and weeks play out, with team-logo app-emojis. It's built around a personality
registry (`app/bot/personality.py`, with `compose_prompt` stitching
voice + role + guard) and the `chat_personality.py` / `commentary.py` /
`recap.py` orchestrators on top of `llm_client.py`.

Design intent: the bot computes the **deterministic facts**; the LLM only phrases
them, and there's a deterministic fallback on any LLM or DB failure — so an LLM
outage degrades to plain text rather than breaking. It targets a **local /
self-hosted OpenAI-compatible** LLM server, configured by `llm_api_server`,
`llm_api_model`, and `llm_api_key` in `app/config.py`. There is **no** cloud
provider involved and **no** tool-calling (that tier is shelved). The admin
personality selector lives in the Admin page (`POST /api/admin/bot-personality`).

The bot comes up with the rest of the stack (`docker compose up`). It depends
only on `db` + `migrate`. Set these in `.env` first — the bot **fails fast and
exits** if the token or guild id are missing (empty values are treated as unset),
and because it has `restart: unless-stopped` a missing token becomes a crash-loop:

```dotenv
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_GUILD_ID=123456789012345678
APP_BASE_URL=http://localhost:5173     # link shown in the password DM
SECRET_KEY=<32+ char random string>    # override the dev default in any non-local env
```

The `members` privileged intent must be enabled in the Discord Developer Portal.
To run the stack **without** the bot, comment out the `bot` service or
`docker compose stop bot`.

The bot's health is tracked via a `/tmp/bot_heartbeat` file it touches every 15s
while the gateway is connected — the compose healthcheck goes unhealthy if the
gateway drops, not just if the process dies.

> **Security note:** `SECRET_KEY` has a dev-only default so the stack runs out of
> the box. It signs session cookies — set a real random value in `.env` for
> anything beyond local development. Setting `APP_ENV=production` activates a
> fail-closed startup guard that refuses to boot if `SECRET_KEY`,
> `SESSION_COOKIE_SECURE`, `CORS_ALLOWED_ORIGINS`, or `IS_DEMO_DATA` are left at
> insecure dev values.

## Testing

The **default** backend test run is fully offline — no Postgres, no Docker. It
builds the schema with `SQLModel.metadata.create_all()` on SQLite:

```bash
cd backend && .venv/bin/python -m unittest
```

Two suites are **opt-in** and skip cleanly by default:

- **Live ESPN smoke** (real outbound GET) — set `RUN_ESPN_LIVE`:

  ```bash
  RUN_ESPN_LIVE=1 .venv/bin/python -m unittest tests.test_scoreboard_espn -v
  ```

- **Postgres + Alembic migration smoke** — runs the **real** `alembic upgrade
  head` (the same entrypoint the compose `migrate` service uses) against a
  throwaway Postgres and asserts the load-bearing PG invariants (one-null
  partial unique index, `pick_edit_audit` / `pick` FK cascades, the single
  `picktype` enum, and a scoped down/up round-trip). The default offline suite
  never runs Alembic, so these Postgres-only behaviors are otherwise unverified.

  The runner stands up an **isolated** throwaway Postgres on port `5433` (never
  port 5432, never the dev/demo `db` service or its volume — the dev/demo
  database is never touched), runs the test, and always tears the container
  down. Requires Docker:

  ```bash
  bash backend/scripts/run_pg_smoke.sh
  ```

  To run it against your own disposable Postgres instead, set
  `TEST_DATABASE_URL` and invoke the test directly:

  ```bash
  cd backend && TEST_DATABASE_URL=postgresql+psycopg://u:p@localhost:5433/db \
    .venv/bin/python -m unittest tests.test_pg_migration_smoke -v
  ```

## Database migrations (Alembic)

Migrations live in `backend/alembic/versions/`. The `migrate` init container
applies them automatically on every `docker compose up`.

**Generate a new migration after changing models** in `backend/app/models.py`:

```bash
docker compose run --rm migrate \
  alembic revision --autogenerate --rev-id 0002 -m "add users table"
# or: make revision rev=0002 m="add users table"
```

This writes `backend/alembic/versions/0002_add_users_table.py`.

> **On the file naming:** Alembic identifies revisions by a random hash and, by
> default, names files `<hash>_<slug>.py`. The `000X_` prefix is *not* automatic
> — it just comes from the `--rev-id` you pass. Use zero-padded sequential ids
> (`0002`, `0003`, …) to keep `versions/` ordered. Omit `--rev-id` and you'll get
> the hash form (e.g. `8e26134bb95d_...py`) — still valid, just unordered.

The file appears in `backend/alembic/versions/` (bind-mounted, so it lands on
your host). Review it, then apply:

```bash
docker compose run --rm migrate alembic upgrade head   # or: make migrate
```

Other useful commands (run inside the backend image):

```bash
docker compose run --rm migrate alembic downgrade -1   # roll back one
docker compose run --rm migrate alembic history        # list revisions
docker compose run --rm migrate alembic current        # show current head
```

> Autogenerate diffs the live DB against `SQLModel.metadata`. Always eyeball the
> generated migration — Alembic can miss renames, enums, and server defaults.

## Adding dependencies

- **Backend:** add to `backend/pyproject.toml`, then rebuild
  (`docker compose build backend worker migrate`).
- **Frontend:** `docker compose run --rm frontend npm install <pkg>`, then
  rebuild.

## Local (non-Docker) backend dev

Uses [uv](https://docs.astral.sh/uv/):

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload
```

You'll need Postgres and Redis reachable at the hosts in your `.env`.

## Contributing

PRs and issues are welcome — see above for the spirit of the project. Clone from
`github.com/zachmce/nfl-pickem`.
