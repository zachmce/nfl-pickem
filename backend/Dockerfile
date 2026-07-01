# syntax=docker/dockerfile:1
FROM python:3.14-slim

# uv: fast, modern Python package manager.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first for better layer caching. A placeholder package
# lets the project build for its deps before the real source is copied in.
COPY pyproject.toml ./
RUN mkdir -p app && touch app/__init__.py \
    && uv pip install --system --no-cache-dir .

# Application source (overwrites the placeholder package).
COPY . .

# Home for Celery beat's persistent schedule DB. In dev a named volume is mounted
# here (keeping the file off the host bind-mount); in prod no volume is mounted,
# so the dir must already exist for `--schedule=/var/lib/celerybeat/...` to work.
RUN mkdir -p /var/lib/celerybeat

# Run as a non-root user (Trivy DS-0002). This image is shared by uvicorn, the
# celery worker+beat, the bot, and the one-shot migrate/seed container — all run
# fine unprivileged: uvicorn binds :8000 (>1024); deps are installed --system to
# /usr/local (read-only at runtime); PYTHONDONTWRITEBYTECODE=1 means no .pyc writes
# into /app; the only runtime-writable path is the celerybeat schedule dir, handed
# to appuser here (a first-mount named volume inherits this ownership).
RUN useradd --system --uid 10001 --create-home appuser \
    && chown -R appuser:appuser /var/lib/celerybeat
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
