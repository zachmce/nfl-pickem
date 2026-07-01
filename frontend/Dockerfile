# syntax=docker/dockerfile:1
# Production image: compile the SPA with Vite, then serve it from nginx.

FROM node:26-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

# nginx:stable-alpine (digest-pinned) — bumped from 1.27-alpine. stable-alpine
# tracks nginx stable (1.29) on the freshest Alpine (3.23.5): scans clean of
# fixable HIGH/CRITICAL, whereas 1.27-alpine (Alpine 3.21) and even the pinned
# 1.29.x-alpine patch tags (Alpine 3.22) still carry openssl/expat CVEs. Digest
# pin keeps builds reproducible; Dependabot's docker ecosystem refreshes it.
FROM nginx:stable-alpine@sha256:0d3b80406a13a767339fbe2f41406d6c7da727ab89cf8fae399e81f780f814d1 AS runtime
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
