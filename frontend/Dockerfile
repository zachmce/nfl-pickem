# syntax=docker/dockerfile:1
# Production image: compile the SPA with Vite, then serve it from nginx.

FROM node:26-alpine@sha256:725aeba2364a9b16beae49e180d83bd597dbd0b15c47f1f28875c290bfd255b9 AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
RUN npm run build

# nginxinc/nginx-unprivileged:stable-alpine (digest-pinned) — the official NGINX
# Inc rootless image: runs as uid 101 (nginx) and listens on 8080 so no root is
# needed to bind the port (Trivy DS-0002). It is the same upstream nginx on the
# same Alpine as nginx:stable-alpine, so the vuln posture is identical — 0 fixable
# HIGH/CRITICAL, with only the same unfixed MEDIUM/UNKNOWN advisories — i.e. the
# swap adds no new Trivy findings. Digest pin keeps builds reproducible;
# Dependabot's docker ecosystem refreshes it.
FROM nginxinc/nginx-unprivileged:stable-alpine@sha256:fd3314e343bad2de4e1127ef58be122abbfa7e09572fa46ae62fcddb6b3f21c5 AS runtime
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
# Explicit non-root USER: `trivy config` reads the Dockerfile text (not the base
# image metadata), so the base's built-in USER 101 must be restated to clear DS-0002.
USER nginx
EXPOSE 8080
# Rootless nginx serves on 8080. busybox wget is already in the image (no new
# packages) — probe the SPA root so the container reports health (Trivy DS-0026).
# Use 127.0.0.1, not localhost: the unprivileged image's listen-on-ipv6 entrypoint
# script can't patch our read-only config as non-root, so nginx is IPv4-only and a
# `localhost` probe would resolve to ::1 and fail.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/ >/dev/null 2>&1 || exit 1
CMD ["nginx", "-g", "daemon off;"]
