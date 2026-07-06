# syntax=docker/dockerfile:1
# Two-stage Chainguard/Wolfi build. glibc-minimal, distroless runtime — drops
# perl-base (CVE-2026-8376 / CVE-2026-42496, both unfixed) and the bulk of the
# Debian OS-package CVE surface the old python:3.14-slim base carried.
#
# This one image is shared by all four backend roles — uvicorn API, the celery
# worker+beat, the Discord bot, and the one-shot migrate/seed container. All run
# unprivileged (Trivy DS-0002): the runtime is Chainguard's nonroot user (uid
# 65532) by default, uvicorn binds :8000 (>1024), deps live in a read-only venv,
# and PYTHONDONTWRITEBYTECODE=1 means no .pyc writes into /app.
#
# Both FROM lines are digest-pinned for reproducibility; Dependabot's docker
# ecosystem (/backend) refreshes the pinned digest so it can't silently go stale.

# ---- builder: :latest-dev has uv + pip + apk + a shell, same 3.14.6 as runtime ----
FROM cgr.dev/chainguard/python:latest-dev@sha256:d7212e2da67f2d04db5fee5e03bb9e500cd92eb7e90397fd1bdcea895584f7ac AS builder

# :latest-dev defaults to the nonroot user, which cannot write /opt/venv (a
# root-owned root-level dir) nor chown the celerybeat dir below. This is a
# throwaway build stage, so become root for the whole stage — only the runtime
# stage's privilege posture ships.
USER root

# UV_PYTHON_DOWNLOADS=never is load-bearing: it stops uv fetching its own
# python-build-standalone interpreter (which would NOT be copied to the runtime,
# leaving the venv's interpreter symlink dangling). Forcing the Chainguard system
# 3.14.6 — the same build present in the runtime stage — is what makes the
# venv-copy resolve.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first for better layer caching. A placeholder package lets
# the project build for its deps before the real source is copied in. The venv is
# built at /opt/venv — deliberately OUTSIDE /app — because the dev + prod-overlay
# compose services bind-mount ./backend:/app, which would shadow a venv under /app
# and every command would lose uvicorn/celery/watchmedo at runtime.
# uv is already bundled in :latest-dev (no COPY --from=astral-sh/uv needed).
# Keep `uv pip install` (NOT `uv sync`) — there is no committed uv.lock; deps are
# tracked via the pip Dependabot ecosystem on pyproject.toml.
COPY pyproject.toml ./
RUN mkdir -p app && touch app/__init__.py \
    && uv venv --python python3.14 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache-dir .

# Application source (overwrites the placeholder package).
COPY . .

# Home for Celery beat's persistent schedule DB. In dev/staging a named volume is
# mounted here (keeping the file off the host bind-mount); in prod no volume is
# mounted, so the dir must already ship in the image for
# `--schedule=/var/lib/celerybeat/...` to work. Create + chown it here as the
# RUNTIME nonroot uid (65532) — the distroless runtime has no shell to chown at
# runtime. The .keep file guards against the empty-dir `COPY --from` edge case.
# (This stage already runs as root — see the USER root at the top.)
RUN mkdir -p /var/lib/celerybeat \
    && touch /var/lib/celerybeat/.keep \
    && chown -R 65532:65532 /var/lib/celerybeat

# ---- runtime: :latest is distroless Wolfi, glibc, nonroot 65532, NO shell ----
FROM cgr.dev/chainguard/python:latest@sha256:676e6960252ff1cf3a3bd003d3f9b7b419c6e2391ae31e7697e4b48ce71d5414

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# The venv resolves because builder and runtime share the identical 3.14.6
# Chainguard interpreter at the same path (see UV_PYTHON_DOWNLOADS=never above).
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=65532:65532 /var/lib/celerybeat /var/lib/celerybeat
COPY --from=builder /app /app

# CRITICAL: Chainguard's python image defaults ENTRYPOINT to ["/usr/bin/python"],
# so a compose `command:` would be passed as ARGUMENTS to python (e.g.
# `python uvicorn app.main:app ...` → crash). Reset it so the exec-form,
# shell-free role commands (uvicorn / celery / watchmedo -> celery /
# python -m app.bot.client / python -m app.bootstrap) run as bare executables.
ENTRYPOINT []

# Restate the nonroot user explicitly even though the base already defaults to it:
# `trivy config` (the iac-scan job) reads the Dockerfile TEXT, not the base image's
# metadata, so an absent USER instruction reads as root and re-raises DS-0002 — the
# exact reason frontend/Dockerfile restates `USER nginx`. Numeric 65532 is valid
# without a passwd entry and matches the celerybeat chown above. No HEALTHCHECK:
# the runtime ships no shell/curl; the bot's healthcheck is a compose-level
# CMD-form probe that execs the venv's python3 directly.
USER 65532

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
