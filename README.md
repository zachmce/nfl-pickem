# NFL Pick'em

A monorepo scaffold:

| Service    | Stack                                   | Dev port |
| ---------- | --------------------------------------- | -------- |
| `backend`  | FastAPI (Python 3.14) + SQLModel        | 8000     |
| `worker`   | Celery worker (Redis broker)            | —        |
| `db`       | Postgres 17                             | 5432     |
| `redis`    | Redis 7 (Celery broker + result store)  | 6379     |
| `migrate`  | Alembic init container (runs on startup)| —        |
| `frontend` | React 19 + Vite (dev) / nginx (prod)    | 5173     |
| `bot`      | Discord bot (discord.py)                | —        |

```
.
├── backend/            FastAPI app, Celery, SQLModel models, Alembic
│   ├── app/
│   │   ├── config.py        settings (env-driven)
│   │   ├── db.py            shared sync engine: get_session + task_session
│   │   ├── models.py        SQLModel tables
│   │   ├── exceptions.py    API exception hierarchy
│   │   ├── logging_config.py structlog JSON logging
│   │   ├── celery_app.py
│   │   ├── tasks.py         sample task that writes to Postgres
│   │   ├── main.py          FastAPI routes
│   │   ├── schemas/         pydantic request/response models
│   │   ├── services/        business logic (auth.py)
│   │   └── bot/             Discord bot: client, db_bridge, commands/
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
  the gateway. Business logic lives once in `app/services/auth.py` and is reused
  by every surface.

The sample `app.tasks.ping` task writes a `TaskRun` row from the worker; the API
reads it back — proving both ends are wired to Postgres.

## Prerequisites

- Docker + Docker Compose

## Getting started

```bash
cp .env.example .env
docker compose up --build      # or: make up
```

On startup the `migrate` init container runs `alembic upgrade head` against
Postgres and exits; `backend` and `worker` wait for it to finish
(`service_completed_successfully`) before booting.

Then open:

- Frontend (Vite dev server): http://localhost:5173
- API docs (Swagger): http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

Click **Enqueue ping task** in the UI (or `POST /api/ping`). The worker writes a
row to Postgres; **Refresh** lists it back via `GET /api/task-runs`.

## Production-style run (nginx serves the compiled SPA)

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build
# or: make prod
```

The frontend is built with `vite build` and served by nginx on
http://localhost:80, with `/api/*` proxied to the backend.

## Authentication (cookie-based)

Cookie sessions backed by the shared `app/services/auth.py`. Login signs a
session token (itsdangerous) and sets it as an **HttpOnly** cookie; the browser
sends it automatically, and FastAPI dependencies resolve the user from it.

| Method & path                  | Auth required | Purpose                                  |
| ------------------------------ | ------------- | ---------------------------------------- |
| `POST /api/auth/login`         | none          | Verify credentials, set session cookie   |
| `POST /api/auth/token`         | none          | OAuth2 password flow — backs Swagger auth |
| `GET  /api/auth/csrf`          | none          | Issue/refresh the CSRF cookie + token    |
| `POST /api/auth/logout`        | none          | Clear the session + CSRF cookies         |
| `GET  /api/auth/me`            | any user      | Current user — SPA auth-state bootstrap  |
| `GET  /api/proof/authenticated`| any user      | Proof endpoint: any logged-in user       |
| `GET  /api/proof/admin`        | admin         | Proof endpoint: admins only              |
| `POST /api/proof/echo`         | any user      | Proof: CSRF-protected cookie mutation    |

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
- Exempt endpoints: `login`, `token`, `logout`, `csrf`.

`POST /api/proof/echo` (authenticated) is a demonstration target: it needs the
CSRF header over the cookie, but works with just a bearer token in Swagger.

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

curl -b jar.txt localhost:8000/api/proof/authenticated   # 200
curl -b jar.txt localhost:8000/api/proof/admin           # 200 if admin, else 403
curl -b jar.txt -X POST localhost:8000/api/auth/logout
```

Relevant settings (`app/config.py`): `SECRET_KEY` (signs cookies — override in
prod), `SESSION_MAX_AGE_DAYS`, `SESSION_COOKIE_SECURE` (set `true` behind HTTPS),
and `CORS_ALLOWED_ORIGINS` (explicit origins are required for credentialed CORS).

## Discord bot

The bot (`app/bot/`) provides three slash commands, all backed by
`app/services/auth.py`:

- `/register` — provision a pick'em account for the invoking member; DMs a
  temporary password.
- `/reset-password` — rotate the member's password and DM the new one.
- `/admin deactivate|reactivate|grant-admin|revoke-admin @member` —
  authorization is gated on the DB `is_admin` column, never Discord roles.

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
> anything beyond local development.

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
