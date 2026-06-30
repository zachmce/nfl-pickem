#!/usr/bin/env bash
#
# run_pg_smoke.sh — OPT-IN Postgres + Alembic migration smoke run (Codex Theme 6).
#
# One-line invocation (from anywhere):
#
#     bash backend/scripts/run_pg_smoke.sh
#
# Stands up a FULLY ISOLATED, throwaway Postgres container (its own credentials,
# its own name, a non-default host port, no persisted volume), points
# TEST_DATABASE_URL at it, runs the real `alembic upgrade head` smoke test
# (tests/test_pg_migration_smoke.py), and ALWAYS tears the container down — even
# on failure or Ctrl-C. It NEVER uses the compose `db` service, the dev/demo
# pgdata volume, or port 5432, so the dev/demo database can never be touched.
#
# Requires: docker + the existing backend venv (.venv). No new dependencies.
set -euo pipefail

CONTAINER="pickem-pg-smoke"
HOST_PORT="5433" # non-5432 so a running dev stack is never disturbed.
PG_USER="smoke"
PG_PASSWORD="smoke"
PG_DB="smoke"

# Resolve the backend/ dir from this script's location so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Always remove the throwaway container on exit (success, failure, or interrupt):
#   docker rm -f pickem-pg-smoke
cleanup() {
  docker rm -f pickem-pg-smoke >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Remove any leftover container from a previous interrupted run before starting.
docker rm -f pickem-pg-smoke >/dev/null 2>&1 || true

echo ">> starting throwaway Postgres container '${CONTAINER}' on host port ${HOST_PORT}..."
docker run --rm -d \
  --name "${CONTAINER}" \
  -e POSTGRES_USER="${PG_USER}" \
  -e POSTGRES_PASSWORD="${PG_PASSWORD}" \
  -e POSTGRES_DB="${PG_DB}" \
  -p "${HOST_PORT}:5432" \
  postgres:17-alpine >/dev/null

echo ">> waiting for Postgres to accept connections..."
ready=""
for _ in $(seq 1 30); do
  if docker exec "${CONTAINER}" pg_isready -U "${PG_USER}" >/dev/null 2>&1; then
    ready="yes"
    break
  fi
  sleep 1
done
if [[ -z "${ready}" ]]; then
  echo "!! Postgres never became ready after ~30s; aborting." >&2
  exit 1
fi

# psycopg3 driver suffix: the test strips it for psycopg.connect and derives
# POSTGRES_* for the alembic subprocess by parsing this URL.
export TEST_DATABASE_URL="postgresql+psycopg://${PG_USER}:${PG_PASSWORD}@localhost:${HOST_PORT}/${PG_DB}"

echo ">> running migration smoke test against ${TEST_DATABASE_URL}"
# Propagate the unittest exit code as the script's exit code (CI/caller sees
# pass/fail) while the EXIT trap still removes the container.
rc=0
(
  cd "${BACKEND_DIR}"
  .venv/bin/python -m unittest tests.test_pg_migration_smoke -v
) || rc=$?

exit "${rc}"
