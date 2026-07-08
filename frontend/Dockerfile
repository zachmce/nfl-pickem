# syntax=docker/dockerfile:1
# Production image: compile the SPA with Vite, then serve it from nginx.

FROM node:26-alpine@sha256:725aeba2364a9b16beae49e180d83bd597dbd0b15c47f1f28875c290bfd255b9 AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
RUN npm run build

# ---- runtime: cgr.dev/chainguard/nginx:latest — distroless, glibc, nonroot ----
# Chainguard's distroless nginx: runs as the nonroot nginx user (uid 65532) and
# listens on 8080 by default, so no root is needed to bind the port (Trivy
# DS-0002). Same config layout as the old nginx-unprivileged image — server blocks
# load from /etc/nginx/conf.d/*.conf and the docroot is /usr/share/nginx/html — so
# both COPY lines and the whole nginx.conf drop in unchanged. The minimal image
# ships none of the Alpine/nginx OS-package advisories the old base carried, so the
# released-digest Trivy findings drop to ~0. Mirrors the merged backend Chainguard
# swap. Digest-pinned for reproducibility; Dependabot's docker ecosystem (/frontend)
# refreshes the pin. Caveat (same as backend): Dependabot bumping a third-party
# cgr.dev `:latest@sha256` digest is less battle-tested than Docker Hub semver —
# confirm on the first weekly PR, Renovate/manual bump as fallback.
FROM cgr.dev/chainguard/nginx:latest@sha256:76cc70e5e8da88ae76ff585525c340fc09d54018c5c902551d61a24c2b3ac0d2 AS runtime
# OVERWRITE the base image's stock conf.d/nginx.default.conf (a `server_name
# localhost; listen 8080` block with no SPA fallback, no /api proxy, and no
# security headers). The main nginx.conf `include /etc/nginx/conf.d/*.conf` loads
# every *.conf, so copying to a DIFFERENT filename (the old default.conf) left the
# stock block in place — and its exact `server_name localhost` match beat our
# `server_name _` default_server for `Host: localhost`, silently shadowing our
# config (headers, SPA fallback, proxy) on the local-access path and re-leaking the
# nginx version. Copying to nginx.default.conf replaces it, leaving exactly one
# server block (ours) that handles every Host. Distroless-safe: a COPY needs no
# shell, unlike a `RUN rm` (this base ships none).
COPY nginx.conf /etc/nginx/conf.d/nginx.default.conf
COPY --from=build /app/dist /usr/share/nginx/html
# Restate the nonroot user explicitly even though the base already defaults to it:
# `trivy config` (iac-scan) reads the Dockerfile TEXT, not the base image metadata,
# so an absent USER instruction reads as root and re-raises DS-0002. Numeric 65532
# is valid without a passwd entry and matches the image's nginx user.
USER 65532
EXPOSE 8080
# No CMD/ENTRYPOINT: the base image already starts nginx correctly on its own
# (ENTRYPOINT ["/usr/sbin/nginx"] + CMD to run with daemon off). Re-adding a
# positional `nginx` arg would double the entrypoint and crash — the same trap the
# backend fixed. Unlike the python image we do NOT reset ENTRYPOINT: nginx's default
# entrypoint is exactly what we want. No container health probe either: distroless
# ships no shell/wget to probe with, and nothing depends on frontend health in
# compose (prod/deploy use a plain depends_on: [backend], no service_healthy).
# Mirrors the backend runtime.
