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

## 5. Still to do (not in this compose)

- **TLS / HTTPS.** This compose serves plain HTTP on `:80`. Put a real reverse
  proxy (Caddy / nginx / Traefik) with a certificate in front, and point
  `CORS_ALLOWED_ORIGINS` / `APP_BASE_URL` at the `https://` origin.
- These are the remaining Codex audit **Theme 1** items; the host-port exposure and
  pgAdmin items are already handled by this file.
