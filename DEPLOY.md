# Deploying NFL Pick'em (Option A — image-based, manual pull)

This is the **server runbook** for a real deployment. CI builds the backend +
frontend images and pushes them to GHCR on every push to `main` (and on `vX.Y.Z`
tags); the server just **pulls and runs** them via
[`docker-compose.deploy.yml`](./docker-compose.deploy.yml). No source checkout and
no build toolchain are needed on the box — only that compose file and a `.env`.

> Design notes: `.planning/notes/deployment-architecture.md`. Hands-off auto-deploy
> (CI push-deploy / Watchtower) is intentionally deferred — see the
> `deploy-automation` seed. Cutover guidance (start from a **fresh non-demo DB**):
> `.planning/notes/demo-to-live-cutover.md`.

## 1. Prerequisites

- A host with **Docker + Docker Compose v2**.
- Access to the GHCR images:
  - `ghcr.io/zachmce/nfl-pickem-backend`
  - `ghcr.io/zachmce/nfl-pickem-frontend`
  - If the packages are **public**, no auth is needed to pull.
  - If they're **private**, run `docker login ghcr.io` with a token that has
    `read:packages` first.

## 2. Create `.env` on the server

The backend **refuses to boot** in production if any of these are insecure (the
`APP_ENV` fail-closed guard in `app/config.py`), so set them deliberately:

```dotenv
# Activates the production guard below — REQUIRED.
APP_ENV=production

# Session-cookie signing key. Must be real and >= 32 chars (NOT the dev default).
SECRET_KEY=<paste: openssl rand -hex 32>

# Cookies are HTTPS-only in prod.
SESSION_COOKIE_SECURE=true

# Recommended: give the session cookie the __Host- prefix so the browser ENFORCES
# Secure + Path=/ + no-Domain. Safe without any code change — both cookie setters
# already use path="/" and never send a Domain. Recommended, not required.
SESSION_COOKIE_NAME=__Host-session

# Real origin(s) the SPA is served from — NO localhost. Comma-separated if several.
CORS_ALLOWED_ORIGINS=https://pickem.example.com

# Never serve demo data in prod.
IS_DEMO_DATA=false

# Where the bot's DM tells users to log in.
APP_BASE_URL=https://pickem.example.com

# Database (used by both the db container and the app).
POSTGRES_USER=pickem
POSTGRES_PASSWORD=<a strong password>
POSTGRES_DB=pickem

# Redis auth — REQUIRED. The deploy stack's redis runs with `--requirepass
# ${REDIS_PASSWORD}` and FAIL-CLOSES (docker compose refuses to parse) if this is
# unset. Generate one with: openssl rand -hex 32
REDIS_PASSWORD=<paste: openssl rand -hex 32>
# REDIS_URL must then carry that password in the userless `:password@` form.
# backend/worker/bot/celery all read this one URL (config.py redis_url) — no code change.
REDIS_URL=redis://:<REDIS_PASSWORD>@redis:6379/0

# Discord bot — required, or the `bot` service crash-loops. Remove the bot
# service from docker-compose.deploy.yml if you are not running it.
DISCORD_BOT_TOKEN=<token>
DISCORD_GUILD_ID=<guild id>

# Optional: admin bootstrap (seeded once on first migrate if set).
# DEFAULT_ADMIN_USERNAME=...
# DEFAULT_ADMIN_PASSWORD=...

# Optional: local/self-hosted OpenAI-compatible LLM for the chat personality.
# Leave unset to disable (the bot falls back to deterministic lines).
# LLM_API_SERVER=...
# LLM_API_MODEL=...
# LLM_API_KEY=...
```

> If any production value is left insecure (dev `SECRET_KEY`, `SESSION_COOKIE_SECURE`
> not true, `IS_DEMO_DATA=true`, or a localhost CORS origin), the backend will exit
> at startup with a single error listing every problem. That's by design.

## 3. Deploy

```bash
# Pull the image set (defaults to :latest = current main). Pin a release with
# IMAGE_TAG, e.g. export IMAGE_TAG=v1.2.0  (or IMAGE_TAG=sha-abc1234).
docker compose -f docker-compose.deploy.yml pull

docker compose -f docker-compose.deploy.yml up -d
```

- `migrate` runs `alembic upgrade head` + seeds once, then exits; `backend`,
  `worker`, and `bot` wait for it to finish.
- Only the **frontend** publishes a port (`:80`). `db`, `redis`, and `backend`
  are reachable only on the internal compose network.

### Network segmentation (defense-in-depth)

`docker-compose.deploy.yml` splits the stack across **two** internal bridge
networks instead of one shared default:

- **`frontend_net`** — the internet-facing `frontend` (nginx) joins **only** this
  network. It reaches `backend:8000` here to proxy `/api`, but has **no route** to
  the data plane.
- **`backend_net`** — `db`, `redis`, `worker`, `bot`, and `migrate` live **only**
  here. `backend` is the **only** service that bridges both networks.

Net effect: the only host-exposed container (nginx) cannot resolve or reach
`postgres:5432` or `redis:6379`. A hypothetical frontend compromise has no direct
path to the data store or the celery/bot broker — defense-in-depth for this
single-operator, internal-network deploy. Combined with Redis `requirepass` above,
reaching the broker now also requires the shared secret.

> **Scope / deferral.** This segmentation + Redis auth is applied to
> `docker-compose.deploy.yml` **only** — the actual internet-facing production
> artifact. `docker-compose.prod.yml` is a thin overlay on the base
> `docker-compose.yml` and does not self-contain `redis`/`db`, so mirroring the
> hardening there is a separate, deliberately-deferred follow-up (not a silent
> skip).

Watch it come up:

```bash
docker compose -f docker-compose.deploy.yml logs -f
docker compose -f docker-compose.deploy.yml ps
```

The app is served at `http://<server>/` (front it with TLS — see below).

## 4. Updating to a new build

```bash
docker compose -f docker-compose.deploy.yml pull   # re-pull :latest (or a newer pinned IMAGE_TAG)
docker compose -f docker-compose.deploy.yml up -d   # recreate only what changed
```

`migrate` re-runs `alembic upgrade head` (a no-op when already current), so schema
changes in a new image are applied on update.

**One-time (upgrading an existing deploy — celerybeat volume uid change):**

The backend image runs as a non-root user, and that uid **changed** when the base
moved to Chainguard/Wolfi: the runtime is now the Chainguard `nonroot` user, uid
`65532` (gid `65532`) — previously it was the Debian `appuser` uid `10001` (and
before that, root). A `celerybeat` named volume created under an **older** image
still holds data owned by that older uid (`10001`, or `root:root`), and Docker
mounts that volume *over* the image's build-time `chown` — so the new `65532` worker
can no longer write `/var/lib/celerybeat/celerybeat-schedule` and embedded celery
beat fails to start (`PermissionError: [Errno 13]`).

Because this is a uid **change** (10001 → 65532), the trap triggers on **any host
that already has a celerybeat volume** from a prior deploy — not just the original
root→non-root upgrade. Only a **pre-existing** volume is affected; a fresh deploy is
fine (a first-mount volume inherits the image dir's `65532` ownership). Fix it once
with either remedy below.

First find the real volume name — it is compose-project-prefixed (e.g.
`nfl-pickem_celerybeat`), and the prefix depends on the deploy directory /
`COMPOSE_PROJECT_NAME`:

```bash
docker volume ls | grep celerybeat
```

- **Remedy A — chown in place** (preserves the schedule; recommended when in doubt):

  ```bash
  docker run --rm -v <project>_celerybeat:/data alpine chown -R 65532:65532 /data
  docker compose -f docker-compose.deploy.yml up -d
  ```

  `65532` is the Chainguard `nonroot` uid and gid. If unsure of the group, chown the
  uid alone (`chown -R 65532 /data`) — the worker writes as uid `65532`, which is what
  matters.

- **Remedy B — recreate** (simplest; the schedule is disposable): the beat schedule
  holds only last-run timestamps, and the `refresh_games` / `refresh_odds` pollers are
  idempotent, so losing it just re-fires them once, harmlessly.

  ```bash
  docker compose -f docker-compose.deploy.yml stop worker
  docker volume rm <project>_celerybeat
  docker compose -f docker-compose.deploy.yml up -d
  ```

**Which stacks are affected by the volume trap:**

- **`docker-compose.deploy.yml`** (staging/prod Option A) mounts a `celerybeat`
  named volume → **affected** on any host that already has that volume.
- **`docker-compose.prod.yml`** mounts **no** celerybeat volume (beat writes into
  the image dir, which is `65532`-owned) → **immune**.
- **`docker-compose.yml`** (local dev) mounts a `celerybeat` volume → **same trap**.
  Apply the identical fix with the dev project prefix, e.g.
  `docker volume rm nfl-pickem_celerybeat` (the prefix follows the deploy directory /
  `COMPOSE_PROJECT_NAME`).

## 5. Still to do (not in this compose)

- **TLS / HTTPS.** This compose serves plain HTTP on `:80`. Put a real reverse
  proxy (Caddy / nginx / Traefik) with a certificate in front, and point
  `CORS_ALLOWED_ORIGINS` / `APP_BASE_URL` at the `https://` origin.
- These are the remaining Codex audit **Theme 1** items; the host-port exposure and
  pgAdmin items are already handled by this file.
